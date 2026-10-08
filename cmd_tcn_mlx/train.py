"""Train the CMD TCN in MLX: one Mac, or data-parallel across both under mlx.launch.

    .venv/bin/python -m cmd_tcn_mlx.train --data-root ~/data/cmd --epochs 1 --test-forward
    ./scripts/run_tcn.sh --cluster --data-root ~/data/cmd --epochs 1 --test-forward --rank-weights 2,1
    .venv/bin/python -m cmd_tcn_mlx.train --data-root <smoke-out> --max-steps 5 --batch-size 32

Same network, loss, optimizer and schedule as cmd_tcn/train.py (PyTorch Lightning). Every rank
owns a contiguous member range and reads one block of its batch size per step straight from the
memory-mapped .npy files (thread prefetch); gradients are averaged with nn.average_gradients each
step; rank 0 prints every --log-every steps and writes results/tcn/<run>.summary.json in the same
schema as the PyTorch run so cmd_tcn_mlx.report can put them side by side.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_map

from cmd_tcn_mlx.data import (X_SEQ_BYTES_PER_MEMBER, BlockReader, Normalizer, make_blocks, n_members,
                              plan_blocks, split_dir)
from cmd_tcn_mlx.model import DEFAULT_DILATIONS, CMDTCNRegressor, count_params, masked_huber
from mlx_cluster.comm import backend_name, gather_json
from mlx_train.train import broadcast_params

DTYPES = {"float32": mx.float32, "bfloat16": mx.bfloat16}


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="output-root given to generate_synthetic_cmd_arrays.py (same path on every Mac)")
    ap.add_argument("--batch-size", type=int, default=1024, help="GLOBAL batch: members per step summed over ranks")
    ap.add_argument("--rank-weights", default=None, help="per-rank share of each batch, e.g. 2,1 (Studio twice the MacBook)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--limit-members", type=int, default=None)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--val-batches", type=int, default=20, help="validation blocks per rank per epoch (0 = all)")
    ap.add_argument("--test-forward", action="store_true", help="evaluate on one_year_forward_test__CMD at the end")
    ap.add_argument("--test-batches", type=int, default=0, help="forward-test blocks per rank (0 = all)")
    ap.add_argument("--channels", type=int, default=512)
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--dilations", default=",".join(map(str, DEFAULT_DILATIONS)))
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--dtype", choices=list(DTYPES), default="float32", help="parameter/compute dtype (bfloat16 = pure bf16)")
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True, help="mx.compile the training step")
    ap.add_argument("--prefetch", type=int, default=4, help="blocks read ahead by the loader threads")
    ap.add_argument("--all-reduce-size", type=int, default=32 * 1024 * 1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--out-dir", default="results/tcn")
    ap.add_argument("--name", default="tcn-mlx")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    group = mx.distributed.init()
    rank, size = group.rank(), group.size()
    backend = backend_name(group)
    is_main = rank == 0

    def say(*a):
        if is_main:
            print(*a, flush=True)

    weights = [float(w) for w in args.rank_weights.split(",")] if args.rank_weights else None
    dilations = tuple(int(d) for d in args.dilations.split(","))
    run_id = f"{args.name}-{size}n-{backend}-{time.strftime('%Y%m%d-%H%M%S')}"

    # ---- model ------------------------------------------------------------------------------
    mx.random.seed(args.seed)
    model = CMDTCNRegressor(channels=args.channels, kernel=args.kernel, dilations=dilations, dropout=args.dropout)
    if args.dtype != "float32":
        model.set_dtype(DTYPES[args.dtype])
    mx.eval(model.parameters())
    if size > 1:
        broadcast_params(model, group)
    n_params = count_params(model)

    # ---- data -------------------------------------------------------------------------------
    tdir = split_dir(args.data_root, "train__CMD")
    norm = Normalizer.fit(tdir, 4096)  # deterministic: first 4096 members, identical files on every Mac
    seq_mean, seq_std = mx.array(norm.seq_mean), mx.array(norm.seq_std)
    sta_mean, sta_std = mx.array(norm.static_mean), mx.array(norm.static_std)
    clip = norm.clip
    n = n_members(args.data_root, "train__CMD")
    if args.limit_members:
        n = min(n, args.limit_members)
    plans, batch_sizes = plan_blocks(n, args.batch_size, size, weights, args.val_fraction, args.seed)
    train_blocks, val_blocks = plans[rank]
    b = batch_sizes[rank]
    grad_weight = b * size / args.batch_size  # weighted average == exact global-batch mean
    reader = BlockReader(tdir, args.prefetch)
    steps_per_epoch = len(train_blocks)
    max_steps = args.max_steps if args.max_steps else steps_per_epoch * args.epochs
    train_members = sum(e - s for s, e in train_blocks)
    val_members = sum(e - s for s, e in val_blocks)
    totals = gather_json({"train": train_members, "val": val_members, "host": socket.gethostname(), "batch": b}, group)
    say(f"== {run_id}: TCN channels={args.channels} kernel={args.kernel} dilations={dilations} params={n_params:,} "
        f"| receptive field {model.tcn.receptive_field} bins | mlx {mx.__version__} {args.dtype} compile={args.compile}")
    say(f"   data {args.data_root}: train {sum(t['train'] for t in totals):,} members / {steps_per_epoch} steps per epoch, "
        f"val {sum(t['val'] for t in totals):,} members | global batch {args.batch_size} = {[t['batch'] for t in totals]} "
        f"on {[t['host'] for t in totals]} | max_steps={max_steps} backend={backend}")

    # ---- step --------------------------------------------------------------------------------
    warm = max(1, args.warmup_steps)
    total = max(max_steps, warm + 1)
    schedule = optim.join_schedules(
        [optim.linear_schedule(args.lr / warm, args.lr, warm), optim.cosine_decay(args.lr, total - warm, 0.1 * args.lr)], [warm])
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=args.weight_decay)

    def prepare(x_seq, x_static, y, out_mask):
        x_seq = mx.clip((mx.nan_to_num(x_seq) - seq_mean) / seq_std, -clip, clip)
        x_static = mx.clip((mx.nan_to_num(x_static) - sta_mean) / sta_std, -clip, clip)
        mask = out_mask * mx.isfinite(y)
        return x_seq, x_static, mx.nan_to_num(y), mask

    def loss_fn(model, x_seq, enroll, x_static, y, mask):
        pred = model(x_seq, enroll, x_static).astype(mx.float32)
        return masked_huber(pred, y, mask)

    loss_and_grad = nn.value_and_grad(model, lambda m, *a: loss_fn(m, *a)[0])

    def step(x_seq_raw, enroll, x_static_raw, y_raw, out_mask):
        x_seq, x_static, y, mask = prepare(x_seq_raw, x_static_raw, y_raw, out_mask)
        if args.dtype != "float32":
            x_seq, enroll, x_static = (t.astype(DTYPES[args.dtype]) for t in (x_seq, enroll, x_static))
        loss, grads = loss_and_grad(model, x_seq, enroll, x_static, y, mask)
        if size > 1:
            if grad_weight != 1.0:
                grads = tree_map(lambda g: g * grad_weight, grads)
                loss = loss * grad_weight
            grads = nn.average_gradients(grads, group, all_reduce_size=args.all_reduce_size)
            loss = mx.distributed.all_sum(loss, group=group) / size
        optimizer.update(model, grads)
        return loss

    def eval_step(x_seq_raw, enroll, x_static_raw, y_raw, out_mask):
        x_seq, x_static, y, mask = prepare(x_seq_raw, x_static_raw, y_raw, out_mask)
        if args.dtype != "float32":
            x_seq, enroll, x_static = (t.astype(DTYPES[args.dtype]) for t in (x_seq, enroll, x_static))
        loss, mae_log, mae_usd = loss_fn(model, x_seq, enroll, x_static, y, mask)
        d = mask.sum()
        return mx.stack([loss * d, mae_log * d, mae_usd * d, d])

    state = [model.state, optimizer.state, mx.random.state]
    if args.compile:
        step = mx.compile(step, inputs=state, outputs=state)

    def to_mx(raw):
        return (mx.array(raw["X_seq"]), mx.array(raw["enrollment_mask"]), mx.array(raw["X_static"]),
                mx.array(raw["y"]), mx.array(raw["outcome_enrollment_mask"]))

    def evaluate(split_path, blocks, limit, label):
        model.eval()
        acc = mx.zeros((4,))
        blocks = blocks[:limit] if limit else blocks
        for raw in BlockReader(split_path, args.prefetch).iter(blocks):
            acc = acc + eval_step(*to_mx(raw))
            mx.eval(acc)
        if size > 1:
            acc = mx.distributed.all_sum(acc, group=group, stream=mx.cpu)
            mx.eval(acc)
        model.train()
        a = acc.tolist()
        d = max(a[3], 1.0)
        return {f"{label}/loss": a[0] / d, f"{label}/mae_log1p": a[1] / d, f"{label}/mae_usd": a[2] / d,
                f"{label}/blocks_per_rank": len(blocks)}

    # ---- loop --------------------------------------------------------------------------------
    skip = 5
    mx.reset_peak_memory()
    t_fit = time.perf_counter()
    gstep, measured, t_meas, members_meas = 0, 0, None, 0
    epoch_losses: list[float] = []
    metrics: dict = {}
    done = False
    for epoch in range(args.epochs if not args.max_steps else 10**9):
        order = np.random.default_rng([args.seed, epoch]).permutation(len(train_blocks))  # same on all ranks
        blocks = [train_blocks[i] for i in order]
        losses = []
        for raw in reader.iter(blocks):
            t0 = time.perf_counter()
            loss = step(*to_mx(raw))
            mx.eval(state, loss)
            lv = float(loss.item())
            losses.append(lv)
            gstep += 1
            if gstep == skip:
                t_meas, measured, members_meas = time.perf_counter(), 0, 0
            elif gstep > skip:
                measured += 1
                members_meas += args.batch_size
            if gstep % args.log_every == 0 or gstep == 1 or gstep == max_steps:
                el = (time.perf_counter() - t_meas) if t_meas and measured else 0.0
                sps = measured / el if el else 0.0
                say(f"step {gstep:5d}  epoch {epoch}  loss {lv:.4f}  lr {float(optimizer.learning_rate.item()):.2e}  "
                    f"{sps:6.2f} steps/s  {sps * args.batch_size:9,.0f} members/s  "
                    f"{sps * args.batch_size * X_SEQ_BYTES_PER_MEMBER / 2**30:5.2f} GiB/s X_seq  "
                    f"peak {mx.get_peak_memory() / 2**30:.2f} GiB  ({time.perf_counter() - t0:.3f}s)")
            if gstep >= max_steps:
                done = True
                break
        if losses:
            epoch_losses.append(float(np.mean(losses)))
        if val_blocks:  # validate at the end of every epoch (and when --max-steps stops mid-epoch)
            metrics.update(evaluate(tdir, val_blocks, args.val_batches, "val"))
            say(f"epoch {epoch} done: train loss {epoch_losses[-1]:.4f} | val loss {metrics['val/loss']:.4f} "
                f"mae_log1p {metrics['val/mae_log1p']:.4f} mae_usd {metrics['val/mae_usd']:,.0f}")
        if done:
            break
    fit_s = time.perf_counter() - t_fit
    metrics["train/loss_epoch"] = epoch_losses[-1] if epoch_losses else float("nan")

    test_metrics: dict = {}
    if args.test_forward:
        try:
            fdir = split_dir(args.data_root, "one_year_forward_test__CMD")
            nf = n_members(args.data_root, "one_year_forward_test__CMD")
            if args.limit_members:
                nf = min(nf, args.limit_members)
            fplans, _ = plan_blocks(nf, args.batch_size, size, weights, 0.0, args.seed)
            test_metrics = evaluate(fdir, fplans[rank][0], args.test_batches, "test")
            say(f"forward test: loss {test_metrics['test/loss']:.4f} mae_log1p {test_metrics['test/mae_log1p']:.4f} "
                f"mae_usd {test_metrics['test/mae_usd']:,.0f}")
        except FileNotFoundError as e:
            say(f"forward test skipped: {e}")

    # ---- summary -------------------------------------------------------------------------------
    el = (time.perf_counter() - t_meas) if t_meas and measured else 0.0
    sps = measured / el if el else 0.0
    peaks = gather_json({"host": socket.gethostname(), "peak": int(mx.get_peak_memory())}, group)
    summary = {
        "run_id": run_id, "framework": "mlx", "mlx": mx.__version__, "hosts": [p["host"] for p in peaks],
        "ranks": size, "backend": backend, "device": str(mx.default_device()),
        "config": vars(args) | {"dilations": list(dilations), "max_steps": max_steps, "global_batch": args.batch_size,
                                "rank_batch_sizes": batch_sizes, "precision": args.dtype},
        "params": n_params, "receptive_field": model.tcn.receptive_field,
        "train_members": sum(t["train"] for t in totals), "val_members": sum(t["val"] for t in totals),
        "steps_per_epoch": steps_per_epoch, "fit_wall_s": fit_s,
        "measured_steps": measured, "wall_s": el, "steps_per_sec": sps, "members_per_sec": sps * args.batch_size,
        "peak_device_bytes": max(p["peak"] for p in peaks), "peak_per_rank": peaks,
        "gib_per_sec_x_seq": sps * args.batch_size * X_SEQ_BYTES_PER_MEMBER / 2**30,
        "metrics": metrics, "test_metrics": test_metrics,
    }
    if is_main:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f"{run_id}.summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\n== {run_id} ==")
        print(f"params {n_params:,} | {size} rank(s) {backend} on {summary['hosts']} | {measured} measured steps | "
              f"{sps:.2f} steps/s | {sps * args.batch_size:,.0f} members/s ({summary['gib_per_sec_x_seq']:.2f} GiB/s of X_seq) | "
              f"peak device mem {summary['peak_device_bytes'] / 2**30:.2f} GiB | fit wall {fit_s:.1f}s")
        for k in ("train/loss_epoch", "val/loss", "val/mae_log1p", "val/mae_usd"):
            if k in metrics:
                print(f"  {k:18s} {metrics[k]:.4f}")
        for k, v in test_metrics.items():
            if not k.endswith("blocks_per_rank"):
                print(f"  {k:18s} {v:.4f}")
        print(f"summary: {out_dir / (run_id + '.summary.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
