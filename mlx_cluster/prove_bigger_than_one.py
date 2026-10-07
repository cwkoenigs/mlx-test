"""PROOF: the two-Mac cluster holds, runs and trains a model larger than either Mac's memory.

    mlx.launch --hostfile hosts.json -- python -m mlx_cluster.prove_bigger_than_one
    mlx.launch --hostfile hosts.json -- python -m mlx_cluster.prove_bigger_than_one --plan-only
    mlx.launch -n 2 --backend ring -- python -m mlx_cluster.prove_bigger_than_one \
        --d-model 512 --n-heads 8 --total-gb 0.3 --criterion none      # local 2-process smoke

What it does
------------
1. Every rank reports its memory (physical RAM, Metal working set). The ranks agree on a plan:
   a decoder-only transformer (same ``Block`` as mlx_train.models) whose total parameter bytes
   exceed the largest single node's memory, split *pipeline-parallel* across the ranks in
   proportion to each node's memory (so a 48 GB MacBook and a 128 GB Studio both contribute).
2. Each rank builds only its own layers (rank 0 also the embedding, the last rank the norm +
   LM head). Weights are measured with mx.get_active_memory(), not estimated.
3. Forward: activations flow rank 0 -> 1 -> ... -> last via mx.distributed.send / recv_like.
   The last rank computes the cross-entropy loss on a fixed synthetic batch.
4. Backward + SGD, pipeline-parallel: cotangents flow back last -> ... -> 0. Each rank runs
   layer-by-layer VJPs (mx.vjp) and applies the SGD update immediately, so only ONE layer's
   gradients exist at any time. Peak extra memory is about one layer + activations, which is
   what lets a model that fills ~80% of both nodes still take real training steps.
5. PASS when: total parameter bytes > the largest node's memory (criterion), every rank's
   resident memory fits its own node, the loss is finite, and it decreases over the steps.

Verified against mlx 0.32.3: mx.distributed.send / recv_like / all_gather, mx.vjp with flat
array primals, nn.Module.update / set_dtype, mx.get_active_memory / get_peak_memory.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

from mlx_cluster import nodeinfo
from mlx_cluster.comm import backend_name, barrier, gather_json, human_bytes
from mlx_train.data import SyntheticLM
from mlx_train.models import Block, block_param_count, param_bytes

DTYPES = {"float32": mx.float32, "bfloat16": mx.bfloat16, "float16": mx.float16}
GIB = 2**30


# ----------------------------------------------------------------------------------------------
# planning
# ----------------------------------------------------------------------------------------------
def plan(usable: list[int], fill: float, headroom: int, layer_bytes: int, act_bytes: int,
         extra: list[int], n_layers: int | None) -> dict:
    """Decide how many layers each rank holds.

    budget_r  = fill * usable_r - headroom - two layers (one layer's gradients + its updated
                 parameters coexist briefly during the SGD update) - rank-specific extras
    capacity_r = budget_r // (layer_bytes + per-layer activation bytes)
    With n_layers=None every rank is filled to capacity; otherwise n_layers are distributed in
    proportion to the budgets (each rank >= 1, <= capacity).
    """
    budgets = [max(0.0, fill * u - headroom - 2 * layer_bytes - e) for u, e in zip(usable, extra)]
    capacity = [int(b // (layer_bytes + act_bytes)) for b in budgets]
    if n_layers is None:
        layers = capacity[:]
    else:
        tot = sum(budgets) or 1.0
        exact = [n_layers * b / tot for b in budgets]
        layers = [int(e) for e in exact]
        for i in sorted(range(len(layers)), key=lambda i: -(exact[i] - layers[i]))[: n_layers - sum(layers)]:
            layers[i] += 1
        layers = [min(max(1, l), c) for l, c in zip(layers, capacity)]
    feasible = all(1 <= l <= c for l, c in zip(layers, capacity)) and (n_layers is None or sum(layers) == n_layers)
    return {"budgets": budgets, "capacity": capacity, "layers": layers, "feasible": feasible}


# ----------------------------------------------------------------------------------------------
# the local pipeline stage
# ----------------------------------------------------------------------------------------------
class Head(nn.Module):
    def __init__(self, d_model: int, vocab: int):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def __call__(self, h: mx.array) -> mx.array:
        return self.head(self.norm(h))


class Stage:
    def __init__(self, rank: int, size: int, n_local: int, d: int, heads: int, vocab: int, dtype, log):
        self.rank, self.size, self.first, self.last = rank, size, rank == 0, rank == size - 1
        self.embed = self.headmod = None
        if self.first:
            self.embed = nn.Embedding(vocab, d)
            self.embed.set_dtype(dtype)
            mx.eval(self.embed.parameters())
        self.blocks: list[Block] = []
        t0 = time.perf_counter()
        report_every = max(1, n_local // 10)
        for i in range(n_local):
            blk = Block(d, heads)
            blk.set_dtype(dtype)
            mx.eval(blk.parameters())
            self.blocks.append(blk)
            if (i + 1) % report_every == 0 or i + 1 == n_local:
                log(f"  built layer {i + 1}/{n_local}  resident {human_bytes(mx.get_active_memory())}  "
                    f"({time.perf_counter() - t0:.0f}s)")
        if self.last:
            self.headmod = Head(d, vocab)
            self.headmod.set_dtype(dtype)
            mx.eval(self.headmod.parameters())

    def param_bytes(self) -> int:
        n = sum(param_bytes(b) for b in self.blocks)
        if self.embed is not None:
            n += param_bytes(self.embed)
        if self.headmod is not None:
            n += param_bytes(self.headmod)
        return n

    def param_count(self) -> int:
        mods = self.blocks + [m for m in (self.embed, self.headmod) if m is not None]
        return sum(v.size for m in mods for _, v in tree_flatten(m.parameters()))


def sgd_update(module: nn.Module, keys, vals, grads: list, lr: float) -> None:
    """p <- p - lr * g, consuming the gradient list as it goes to cap transient memory."""
    new = []
    for v in vals:
        g = grads.pop(0)
        nv = v - (lr * g).astype(v.dtype)
        mx.eval(nv)
        new.append(nv)
        del g
    module.update(tree_unflatten(list(zip(keys, new))))


def forward(stage: Stage, x: mx.array, y: mx.array, shape, dtype, group):
    """Returns (layer inputs, output hidden state, loss-or-None)."""
    if stage.first:
        h = stage.embed(x)
    else:
        h = mx.distributed.recv_like(mx.zeros(shape, dtype=dtype), stage.rank - 1, group=group)
    mx.eval(h)
    acts = []
    for blk in stage.blocks:
        acts.append(h)
        h = blk(h)
        mx.eval(h)
    if stage.last:
        logits = stage.headmod(h).astype(mx.float32)
        loss = nn.losses.cross_entropy(logits, y, reduction="mean")
        mx.eval(loss)
        return acts, h, loss
    mx.eval(mx.distributed.send(h, stage.rank + 1, group=group))
    return acts, h, None


def backward(stage: Stage, acts, h_out: mx.array, x: mx.array, y: mx.array, lr: float, group) -> None:
    """Layer-by-layer VJP with immediate SGD update; cotangent flows to the previous rank."""
    if stage.last:
        keys, vals = zip(*tree_flatten(stage.headmod.parameters()))

        def f_head(*args):
            stage.headmod.update(tree_unflatten(list(zip(keys, args[:-1]))))
            logits = stage.headmod(args[-1]).astype(mx.float32)
            return nn.losses.cross_entropy(logits, y, reduction="mean")

        _, vjps = mx.vjp(f_head, [*vals, h_out], [mx.array(1.0, dtype=mx.float32)])
        dh, grads = vjps[-1], vjps[:-1]
        del vjps
        mx.eval(dh)
        sgd_update(stage.headmod, keys, vals, grads, lr)
    else:
        dh = mx.distributed.recv_like(h_out, stage.rank + 1, group=group)
        mx.eval(dh)

    for i in reversed(range(len(stage.blocks))):
        blk = stage.blocks[i]
        keys, vals = zip(*tree_flatten(blk.parameters()))

        def f_block(*args, blk=blk, keys=keys):
            blk.update(tree_unflatten(list(zip(keys, args[:-1]))))
            return blk(args[-1])

        _, vjps = mx.vjp(f_block, [*vals, acts[i]], [dh])
        dh, grads = vjps[-1], vjps[:-1]
        del vjps
        mx.eval(dh)
        sgd_update(blk, keys, vals, grads, lr)

    if stage.first:
        keys, vals = zip(*tree_flatten(stage.embed.parameters()))

        def f_embed(*args):
            stage.embed.update(tree_unflatten(list(zip(keys, args))))
            return stage.embed(x)

        _, vjps = mx.vjp(f_embed, list(vals), [dh])
        sgd_update(stage.embed, keys, vals, list(vjps), lr)
    else:
        mx.eval(mx.distributed.send(dh, stage.rank - 1, group=group))


# ----------------------------------------------------------------------------------------------
def write_report(path_base: Path, rep: dict) -> None:
    path_base.parent.mkdir(parents=True, exist_ok=True)
    path_base.with_suffix(".json").write_text(json.dumps(rep, indent=2, default=str) + "\n")
    n = rep["nodes"]
    md = [f"# Bigger-than-one-Mac proof: **{rep['verdict']}**", "",
          f"Run: {rep['timestamp']}  |  backend: {rep['backend']}  |  ranks: {rep['ranks']}  |  criterion: {rep['criterion']}", "",
          "## Model", "",
          f"- Decoder-only transformer, d_model={rep['model']['d_model']}, n_heads={rep['model']['n_heads']}, "
          f"layers={rep['model']['n_layers']} (split {rep['model']['split']}), vocab={rep['model']['vocab']}, dtype={rep['model']['dtype']}",
          f"- Total parameters: **{rep['total_params']:,}** = **{rep['total_param_bytes'] / GIB:.2f} GiB** (measured, summed over ranks)",
          f"- Batch {rep['model']['batch']} x seq {rep['model']['seq_len']} tokens per step", "",
          "## Nodes", "",
          "| Rank | Host | Chip | RAM | Usable (criterion) | Layers | Params held | Resident after build | Peak during training |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in n:
        md.append(f"| {r['rank']} | {r['hostname']} | {r.get('chip') or '-'} | {r['ram_bytes'] / GIB:.1f} GiB | "
                  f"{r['usable_bytes'] / GIB:.1f} GiB | {r['layers']} | {r['param_bytes'] / GIB:.2f} GiB | "
                  f"{r['active_after_build'] / GIB:.2f} GiB | {r['peak_bytes'] / GIB:.2f} GiB |")
    md += ["", "## Claim", ""]
    for c in rep["checks"]:
        md.append(f"- {'PASS' if c['ok'] else 'FAIL'}: {c['what']}")
    md += ["", "## Training", "",
           f"- Losses per step (last rank): {', '.join(f'{l:.4f}' for l in rep['losses'])}",
           f"- Step wall time (rank 0, forward + backward across the pipeline): "
           f"{', '.join(f'{t:.1f}s' for t in rep['step_seconds'])}", ""]
    path_base.with_suffix(".md").write_text("\n".join(md) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--total-gb", type=float, default=None,
                    help="target model size in GiB (default: fill every node to --fill)")
    ap.add_argument("--criterion", choices=["ram", "working-set", "none"], default="ram",
                    help="what 'a node's memory' means: physical RAM (default), Metal max recommended working set, "
                         "or none (skip the size claim; for local smoke runs)")
    ap.add_argument("--fill", type=float, default=0.80, help="max fraction of each node's usable memory for parameters")
    ap.add_argument("--headroom-gb", type=float, default=2.0, help="per-node reserve for temporaries")
    ap.add_argument("--split", default="auto", help="layers per rank, e.g. 30,10 (default: proportional to memory)")
    ap.add_argument("--d-model", type=int, default=8192)
    ap.add_argument("--n-heads", type=int, default=64)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--dtype", choices=list(DTYPES), default="float32")
    ap.add_argument("--steps", type=int, default=4, help="pipeline-parallel SGD steps on a fixed batch (0 = forward only)")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plan-only", action="store_true", help="print the plan and exit without allocating")
    ap.add_argument("--out-dir", default="results/bigger_than_one")
    args = ap.parse_args()

    group = mx.distributed.init()
    rank, size = group.rank(), group.size()
    backend = backend_name(group)
    is_main = rank == 0

    def say(*a):
        if is_main:
            print(*a, flush=True)

    if args.d_model % args.n_heads:
        say(f"FATAL: --d-model {args.d_model} must be divisible by --n-heads {args.n_heads}")
        return 1
    dtype = DTYPES[args.dtype]
    itemsize = mx.zeros((1,), dtype=dtype).nbytes
    d, V, B, L = args.d_model, args.vocab, args.batch, args.seq_len

    # ---- 1. who is here, how much memory ---------------------------------------------------
    infos = gather_json(nodeinfo.collect(), group)
    crit = "ram" if args.criterion == "none" else args.criterion
    usable = [nodeinfo.usable_bytes(i, crit) for i in infos]
    say(f"== bigger-than-one proof: {size} rank(s), backend={backend}, criterion={args.criterion} ==")
    for i, u in zip(infos, usable):
        say(f"  rank {i['hostname']}: {i.get('chip') or '-'} | RAM {i['ram_bytes'] / GIB:.1f} GiB | "
            f"working set {(i.get('gpu_working_set_bytes') or 0) / GIB:.1f} GiB | usable {u / GIB:.1f} GiB | mlx {i['mlx_version']}")
    if size < 2:
        say("WARNING: a single rank cannot prove anything about a cluster; launch with mlx.launch.")

    # ---- 2. plan ---------------------------------------------------------------------------
    layer_bytes = block_param_count(d) * itemsize
    act_bytes = B * L * d * itemsize  # one stored layer input per layer for the backward pass
    headroom = int(args.headroom_gb * GIB)
    embed_bytes, head_bytes = V * d * itemsize, V * d * itemsize + d * itemsize
    extra = [(embed_bytes if r == 0 else 0) + (head_bytes if r == size - 1 else 0) for r in range(size)]
    if args.split != "auto":
        layers = [int(v) for v in args.split.split(",")]
        if len(layers) != size:
            say(f"FATAL: --split has {len(layers)} entries for {size} ranks")
            return 1
        p = plan(usable, args.fill, headroom, layer_bytes, act_bytes, extra, None)
        p["layers"] = layers
        p["feasible"] = all(1 <= l <= c for l, c in zip(layers, p["capacity"]))
    else:
        n_layers = None if args.total_gb is None else max(size, math.ceil((args.total_gb * GIB - sum(extra)) / layer_bytes))
        p = plan(usable, args.fill, headroom, layer_bytes, act_bytes, extra, n_layers)
    layers = p["layers"]
    n_layers = sum(layers)
    total_bytes_planned = n_layers * layer_bytes + sum(extra)
    max_usable = max(usable)
    say(f"plan: d_model={d} heads={args.n_heads} vocab={V} dtype={args.dtype} -> {human_bytes(layer_bytes)} per layer, "
        f"{human_bytes(act_bytes)} activations per layer (batch {B} x seq {L})")
    for r in range(size):
        say(f"  rank {r} ({infos[r]['hostname']}): budget {p['budgets'][r] / GIB:.1f} GiB -> capacity {p['capacity'][r]} layers, "
            f"assigned {layers[r]} layers = {human_bytes(layers[r] * layer_bytes + extra[r])}")
    say(f"  total: {n_layers} layers = {total_bytes_planned / GIB:.1f} GiB of parameters "
        f"vs largest single node usable {max_usable / GIB:.1f} GiB "
        f"({'BIGGER' if total_bytes_planned > max_usable else 'NOT bigger'})")
    if not p["feasible"]:
        say("FATAL: this plan does not fit. Lower --total-gb, raise --fill, or pass --split explicitly.")
        return 2
    if args.criterion != "none" and total_bytes_planned <= max_usable:
        say("FATAL: even filling every node to --fill, the model would NOT exceed the largest node's memory.\n"
            "       Options: raise --fill (close other apps first), use --criterion working-set, or accept that\n"
            "       this pair of machines cannot demonstrate the claim against physical RAM.")
        return 2
    if args.plan_only:
        say("plan only; exiting.")
        return 0

    # ---- 3. build the local stage -----------------------------------------------------------
    mx.random.seed(args.seed * 1000 + rank)
    mx.reset_peak_memory()
    say(f"building stages ...")
    t0 = time.perf_counter()
    stage = Stage(rank, size, layers[rank], d, args.n_heads, V, dtype, say)
    barrier(group)
    build_s = time.perf_counter() - t0
    local = {"rank": rank, "hostname": infos[rank]["hostname"], "chip": infos[rank].get("chip"),
             "ram_bytes": infos[rank]["ram_bytes"], "usable_bytes": usable[rank], "layers": layers[rank],
             "param_bytes": stage.param_bytes(), "param_count": stage.param_count(),
             "active_after_build": int(mx.get_active_memory()), "rss_after_build": nodeinfo.rss_bytes()}
    say(f"built in {build_s:.0f}s")

    # ---- 4. forward / backward steps ---------------------------------------------------------
    ds = SyntheticLM(1 << 16, L, V, args.seed)
    x, y = next(ds.batches(B, seed=args.seed))  # identical fixed batch on every rank
    mx.eval(x, y)
    shape = (B, L, d)
    losses, step_seconds = [], []
    n_steps = max(1, args.steps)
    for s in range(n_steps):
        barrier(group)
        t0 = time.perf_counter()
        acts, h, loss = forward(stage, x, y, shape, dtype, group)
        if args.steps > 0:
            backward(stage, acts, h, x, y, args.lr, group)
        del acts, h
        barrier(group)
        dt = time.perf_counter() - t0
        step_seconds.append(dt)
        lv = gather_json(None if loss is None else float(loss.item()), group)[size - 1]
        losses.append(lv)
        say(f"step {s}: loss {lv:.4f}  ({dt:.1f}s forward{'+backward+sgd' if args.steps > 0 else ''} across {size} ranks)")
    local["peak_bytes"] = int(mx.get_peak_memory())
    local["rss_peak"] = nodeinfo.rss_bytes()
    nodes = gather_json(local, group)

    # ---- 5. verdict ----------------------------------------------------------------------------
    total_param_bytes = sum(n["param_bytes"] for n in nodes)
    total_params = sum(n["param_count"] for n in nodes)
    checks = []
    if args.criterion != "none":
        checks.append({"ok": total_param_bytes > max_usable,
                       "what": f"model parameters ({total_param_bytes / GIB:.1f} GiB, measured) exceed the largest node's "
                               f"{args.criterion} ({max_usable / GIB:.1f} GiB on {infos[usable.index(max_usable)]['hostname']})"})
        for n in nodes:
            checks.append({"ok": n["param_bytes"] <= n["usable_bytes"] and n["peak_bytes"] <= n["ram_bytes"],
                           "what": f"{n['hostname']} holds {n['param_bytes'] / GIB:.1f} GiB of parameters, peak {n['peak_bytes'] / GIB:.1f} GiB, "
                                   f"within its {n['ram_bytes'] / GIB:.1f} GiB RAM"})
    else:
        checks.append({"ok": True, "what": f"size claim skipped (--criterion none); model is {total_param_bytes / GIB:.2f} GiB over {size} ranks"})
    checks.append({"ok": all(math.isfinite(l) for l in losses), "what": f"forward pass across the pipeline gives a finite loss ({losses[0]:.4f})"})
    if args.steps >= 2:
        checks.append({"ok": losses[-1] < losses[0],
                       "what": f"pipeline-parallel backward + SGD decreases the loss over {args.steps} steps ({losses[0]:.4f} -> {losses[-1]:.4f})"})
    verdict = "PASS" if all(c["ok"] for c in checks) else "FAIL"
    rep = {"verdict": verdict, "timestamp": time.strftime("%Y%m%d-%H%M%S"), "backend": backend, "ranks": size,
           "criterion": args.criterion, "fill": args.fill,
           "model": {"d_model": d, "n_heads": args.n_heads, "n_layers": n_layers, "split": layers, "vocab": V,
                     "dtype": args.dtype, "batch": B, "seq_len": L, "lr": args.lr, "steps": args.steps},
           "total_params": total_params, "total_param_bytes": total_param_bytes, "max_usable_bytes": max_usable,
           "nodes": nodes, "checks": checks, "losses": losses, "step_seconds": step_seconds, "build_seconds": build_s}
    if is_main:
        base = Path(args.out_dir) / f"proof-{rep['timestamp']}"
        write_report(base, rep)
        print()
        for c in checks:
            print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['what']}")
        print(f"\nPROOF: {verdict}   (report: {base.with_suffix('.md')})")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
