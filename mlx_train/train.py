"""Training loop: single node, or distributed data-parallel under ``mlx.launch``.

    python -m mlx_train.train --config configs/10m.yaml
    mlx.launch -n 2 --backend ring -- python -m mlx_train.train --config configs/10m.yaml
    mlx.launch --hostfile hosts.json -- python -m mlx_train.train --config configs/10m.yaml

Every step: forward/backward -> (distributed) nn.average_gradients -> clip -> AdamW update
-> mx.eval(model, optimizer state, loss). Rank 0 logs JSONL + a summary and appends one line
to results/benchmarks.jsonl for ``python -m mlx_train.report``.

Verified against mlx 0.32.3: nn.value_and_grad, nn.average_gradients(all_reduce_size=...),
optim.AdamW / clip_grad_norm / join_schedules, mx.compile(inputs=..., outputs=...),
mx.get_peak_memory / reset_peak_memory.
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
from mlx.utils import tree_flatten, tree_map

from mlx_cluster.comm import backend_name, gather_json
from mlx_train.config import load_config, to_dict
from mlx_train.data import build_dataset
from mlx_train.models import build_model, count_params, param_bytes, param_checksum

DTYPES = {"float32": mx.float32, "bfloat16": mx.bfloat16, "float16": mx.float16}


def rank_batch_sizes(global_batch: int, size: int, weights: list[float] | None) -> list[int]:
    """Split the global batch across ranks (optionally unevenly). Sum is exactly global_batch."""
    w = [float(x) for x in (weights or [1.0] * size)]
    if len(w) != size:
        raise ValueError(f"rank_batch_weights has {len(w)} entries but there are {size} ranks")
    tot = sum(w)
    exact = [global_batch * wi / tot for wi in w]
    sizes = [int(e) for e in exact]
    for i in sorted(range(size), key=lambda i: -(exact[i] - sizes[i]))[: global_batch - sum(sizes)]:
        sizes[i] += 1
    if any(s <= 0 for s in sizes):
        raise ValueError(f"global_batch_size={global_batch} is too small for {size} ranks with weights {w}")
    return sizes


def broadcast_params(model: nn.Module, group: mx.distributed.Group) -> None:
    """Make every rank's parameters identical to rank 0's (all_sum of rank-0 values and zeros)."""
    rank = group.rank()

    def bc(p: mx.array) -> mx.array:
        return mx.distributed.all_sum(p if rank == 0 else mx.zeros_like(p), group=group)

    model.update(tree_map(bc, model.parameters()))
    mx.eval(model.parameters())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="YAML config (configs/*.yaml)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override, e.g. model.d_model=512")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--tag", default="", help="extra label for the log file / benchmark row (e.g. backend)")
    ap.add_argument("--log-dir")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.set)
    if args.steps:
        cfg.steps = args.steps
    if args.compile is not None:
        cfg.compile = args.compile
    if args.log_dir:
        cfg.log_dir = args.log_dir

    group = mx.distributed.init()
    rank, size = group.rank(), group.size()
    backend = backend_name(group)
    is_main = rank == 0

    def say(*a):
        if is_main:
            print(*a, flush=True)

    # ---- model (identical init on every rank) -------------------------------------------
    mx.random.seed(cfg.seed)
    model = build_model(cfg.model)
    if cfg.dtype != "float32":
        model.set_dtype(DTYPES[cfg.dtype])
    mx.eval(model.parameters())
    n_params = count_params(model)
    init_report = {"checksum_seed_init": param_checksum(model)}
    if size > 1:
        cks = gather_json(init_report["checksum_seed_init"], group)
        init_report["seed_init_identical_across_ranks"] = max(cks) - min(cks) <= 1e-6 * max(1.0, abs(cks[0]))
        broadcast_params(model, group)
        cks2 = gather_json(param_checksum(model), group)
        init_report["checksums_after_broadcast"] = cks2
        init_report["identical_after_broadcast"] = len(set(cks2)) == 1
        if not init_report["identical_after_broadcast"]:
            say("FATAL: parameters differ across ranks after broadcast:", cks2)
            return 1
    say(f"model={cfg.model.type} params={n_params:,} ({param_bytes(model) / 2**20:.1f} MiB {cfg.dtype}) "
        f"ranks={size} backend={backend} compile={cfg.compile}")
    if size > 1:
        say("init check:", json.dumps(init_report))

    # ---- data / batch split --------------------------------------------------------------
    batch_sizes = rank_batch_sizes(cfg.global_batch_size, size, cfg.rank_batch_weights)
    b = batch_sizes[rank]
    grad_weight = b * size / cfg.global_batch_size  # == 1.0 for an even split
    dataset = build_dataset(cfg.data, cfg.model).shard(rank, size)
    batches = dataset.batches(b, seed=cfg.seed)
    tokens_per_step = cfg.global_batch_size * dataset.tokens_per_sample
    unit = "tokens" if dataset.tokens_per_sample > 1 else "samples"
    say(f"global batch={cfg.global_batch_size} per-rank batches={batch_sizes} {unit}/step={tokens_per_step}")

    # ---- optimizer / step ----------------------------------------------------------------
    warmup = max(1, cfg.warmup_steps)
    schedule = optim.join_schedules(
        [optim.linear_schedule(0.0, cfg.lr, warmup),
         optim.cosine_decay(cfg.lr, max(1, cfg.steps - warmup), cfg.lr * cfg.min_lr_ratio)],
        [warmup],
    )
    optimizer = optim.AdamW(learning_rate=schedule, weight_decay=cfg.weight_decay)

    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(logits.astype(mx.float32), y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    def step(x, y):
        loss, grads = loss_and_grad(model, x, y)
        if size > 1:
            if grad_weight != 1.0:  # weighted average == exact global-batch mean
                grads = tree_map(lambda g: g * grad_weight, grads)
                loss = loss * grad_weight
            grads = nn.average_gradients(grads, group, all_reduce_size=cfg.all_reduce_size)
            loss = mx.distributed.all_sum(loss, group=group) / size
        if cfg.grad_clip > 0:
            grads, _ = optim.clip_grad_norm(grads, cfg.grad_clip)
        optimizer.update(model, grads)
        return loss

    state = [model.state, optimizer.state, mx.random.state]
    if cfg.compile:
        step = mx.compile(step, inputs=state, outputs=state)

    # ---- loop -----------------------------------------------------------------------------
    ts = time.strftime("%Y%m%d-%H%M%S")
    tag = f"-{args.tag}" if args.tag else ""
    run_id = f"{cfg.name}{tag}-{size}n-{backend}-{ts}"
    log_path = Path(cfg.log_dir) / f"{run_id}.jsonl"
    if is_main:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "w") if is_main else None

    mx.reset_peak_memory()
    step_times, losses = [], []
    t_start = time.perf_counter()
    for s in range(cfg.steps):
        x, y = next(batches)
        t0 = time.perf_counter()
        loss = step(x, y)
        mx.eval(state, loss)
        dt = time.perf_counter() - t0
        lv = float(loss.item())
        step_times.append(dt)
        losses.append(lv)
        rec = {
            "step": s, "loss": lv, "lr": float(optimizer.learning_rate.item()), "step_time_s": dt,
            "steps_per_sec": 1.0 / dt, f"{unit}_per_sec": tokens_per_step / dt,
            "peak_mem_bytes": int(mx.get_peak_memory()), "active_mem_bytes": int(mx.get_active_memory()),
        }
        if log_f:
            log_f.write(json.dumps(rec) + "\n")
        if s % cfg.log_every == 0 or s == cfg.steps - 1:
            say(f"step {s:5d}  loss {lv:.4f}  {1 / dt:7.2f} steps/s  {tokens_per_step / dt:12,.0f} {unit}/s  "
                f"peak {mx.get_peak_memory() / 2**30:.2f} GiB")
    wall = time.perf_counter() - t_start
    if log_f:
        log_f.close()

    # ---- summary --------------------------------------------------------------------------
    kept = step_times[cfg.skip_first:] or step_times
    avg_dt = sum(kept) / len(kept)
    per_rank = gather_json({"host": socket.gethostname(), "rank": rank, "batch": b,
                            "peak_mem_bytes": int(mx.get_peak_memory())}, group)
    summary = {
        "run_id": run_id, "name": cfg.name, "tag": args.tag, "framework": "mlx", "mlx_version": mx.__version__,
        "config": to_dict(cfg), "params": n_params, "param_bytes": param_bytes(model),
        "ranks": size, "backend": backend, "hosts": [r["host"] for r in per_rank],
        "global_batch_size": cfg.global_batch_size, "rank_batch_sizes": batch_sizes,
        "seq_len": cfg.model.seq_len if cfg.model.type == "transformer" else None,
        "compile": cfg.compile, "dtype": cfg.dtype, "steps": cfg.steps, "skip_first": cfg.skip_first,
        "steps_per_sec": 1.0 / avg_dt, f"{unit}_per_sec": tokens_per_step / avg_dt, "throughput_unit": unit,
        "wall_seconds": wall, "first_loss": losses[0], "final_loss": losses[-1],
        "min_loss": min(losses), "peak_mem_bytes_per_rank": [r["peak_mem_bytes"] for r in per_rank],
        "init_check": init_report, "log": str(log_path), "timestamp": ts,
    }
    if is_main:
        (Path(cfg.log_dir) / f"{run_id}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        bf = Path(cfg.benchmarks_file)
        bf.parent.mkdir(parents=True, exist_ok=True)
        with open(bf, "a") as f:
            f.write(json.dumps(summary) + "\n")
        print(f"\n== {run_id} ==")
        print(f"params {n_params:,} | ranks {size} ({backend}) | global batch {cfg.global_batch_size} | compile {cfg.compile}")
        print(f"avg {summary['steps_per_sec']:.2f} steps/s | {summary[unit + '_per_sec']:,.0f} {unit}/s | "
              f"loss {losses[0]:.4f} -> {losses[-1]:.4f} | wall {wall:.1f}s")
        print("peak memory per rank:", ", ".join(f"{r['host']}: {r['peak_mem_bytes'] / 2**30:.2f} GiB" for r in per_rank))
        print(f"log: {log_path}")
        if cfg.checkpoint:
            Path(cfg.checkpoint).parent.mkdir(parents=True, exist_ok=True)
            mx.save_safetensors(cfg.checkpoint, dict(tree_flatten(model.parameters())))
            print(f"checkpoint: {cfg.checkpoint}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
