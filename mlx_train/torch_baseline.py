"""Optional Phase 1 comparison: the same transformer and synthetic data in PyTorch on MPS.

    uv sync --frozen --extra torch
    .venv/bin/python -m mlx_train.torch_baseline --config configs/10m.yaml

Writes the same JSONL/summary schema as mlx_train.train with ``framework: torch`` so
``python -m mlx_train.report`` can show the comparison. Single device only.

NOTE: written to match the MLX model one-to-one but not executed in the authoring environment
(no PyTorch there); run configs/tiny.yaml first on the Mac and fix anything it reports.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import time
from pathlib import Path

import numpy as np

from mlx_train.config import load_config, to_dict
from mlx_train.data import build_dataset
from mlx_train.models import ffn_dim


def main() -> int:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--steps", type=int)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--tag", default="torch")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    if args.steps:
        cfg.steps = args.steps
    if cfg.model.type != "transformer":
        raise SystemExit("torch baseline implements the transformer only")
    dev = torch.device(args.device)
    torch.manual_seed(cfg.seed)
    m = cfg.model

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.h, self.hd = m.n_heads, m.d_model // m.n_heads
            self.wqkv = nn.Linear(m.d_model, 3 * m.d_model, bias=False)
            self.wo = nn.Linear(m.d_model, m.d_model, bias=False)
            inv = 1.0 / (10000 ** (torch.arange(0, self.hd, 2).float() / self.hd))
            t = torch.arange(m.seq_len).float()
            freqs = torch.outer(t, inv)
            self.register_buffer("cos", freqs.cos()[None, None], persistent=False)
            self.register_buffer("sin", freqs.sin()[None, None], persistent=False)

        def rope(self, x):  # non-traditional RoPE (halves), matches nn.RoPE(traditional=False)
            L = x.shape[2]
            x1, x2 = x[..., : self.hd // 2], x[..., self.hd // 2 :]
            c, s = self.cos[:, :, :L], self.sin[:, :, :L]
            return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)

        def forward(self, x):
            B, L, _ = x.shape
            q, k, v = self.wqkv(x).split(m.d_model, dim=-1)
            q = self.rope(q.view(B, L, self.h, self.hd).transpose(1, 2))
            k = self.rope(k.view(B, L, self.h, self.hd).transpose(1, 2))
            v = v.view(B, L, self.h, self.hd).transpose(1, 2)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return self.wo(o.transpose(1, 2).reshape(B, L, -1))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            ff = ffn_dim(m.d_model, m.d_ff)
            self.attn_norm, self.mlp_norm = nn.RMSNorm(m.d_model), nn.RMSNorm(m.d_model)
            self.attn = Attention()
            self.gate, self.up, self.down = nn.Linear(m.d_model, ff, bias=False), nn.Linear(m.d_model, ff, bias=False), nn.Linear(ff, m.d_model, bias=False)

        def forward(self, x):
            x = x + self.attn(self.attn_norm(x))
            h = self.mlp_norm(x)
            return x + self.down(F.silu(self.gate(h)) * self.up(h))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(m.vocab_size, m.d_model)
            self.layers = nn.ModuleList([Block() for _ in range(m.n_layers)])
            self.norm = nn.RMSNorm(m.d_model)
            self.head = nn.Linear(m.d_model, m.vocab_size, bias=False)

        def forward(self, t):
            h = self.embed(t)
            for l in self.layers:
                h = l(h)
            return self.head(self.norm(h))

    model = Model().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    dataset = build_dataset(cfg.data, cfg.model)
    batches = dataset.batches(cfg.global_batch_size, seed=cfg.seed)
    warm = max(1, cfg.warmup_steps)

    def lr_at(s):
        if s < warm:
            return cfg.lr * s / warm
        p = (s - warm) / max(1, cfg.steps - warm)
        return cfg.lr * cfg.min_lr_ratio + (cfg.lr - cfg.lr * cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    tokens_per_step = cfg.global_batch_size * m.seq_len
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_id = f"{cfg.name}-{args.tag}-1n-torch-{dev.type}-{ts}"
    log_path = Path(cfg.log_dir) / f"{run_id}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"torch {torch.__version__} device={dev} params={n_params:,}")
    losses, times = [], []
    peak = 0
    with open(log_path, "w") as log:
        for s in range(cfg.steps):
            x, y = next(batches)
            x = torch.from_numpy(np.array(x)).long().to(dev)
            y = torch.from_numpy(np.array(y)).long().to(dev)
            for g in opt.param_groups:
                g["lr"] = lr_at(s)
            t0 = time.perf_counter()
            loss = F.cross_entropy(model(x).float().view(-1, m.vocab_size), y.view(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            if dev.type == "mps":
                torch.mps.synchronize()
                peak = max(peak, int(torch.mps.driver_allocated_memory()))
            dt = time.perf_counter() - t0
            lv = float(loss.item())
            losses.append(lv)
            times.append(dt)
            log.write(json.dumps({"step": s, "loss": lv, "lr": lr_at(s), "step_time_s": dt, "steps_per_sec": 1 / dt,
                                  "tokens_per_sec": tokens_per_step / dt, "peak_mem_bytes": peak}) + "\n")
            if s % cfg.log_every == 0 or s == cfg.steps - 1:
                print(f"step {s:5d}  loss {lv:.4f}  {1 / dt:7.2f} steps/s  {tokens_per_step / dt:12,.0f} tokens/s")
    kept = times[cfg.skip_first:] or times
    avg = sum(kept) / len(kept)
    summary = {"run_id": run_id, "name": cfg.name, "tag": args.tag, "framework": "torch", "torch_version": torch.__version__,
               "config": to_dict(cfg), "params": n_params, "ranks": 1, "backend": f"torch-{dev.type}",
               "hosts": [socket.gethostname()], "global_batch_size": cfg.global_batch_size, "seq_len": m.seq_len,
               "compile": False, "dtype": "float32", "steps": cfg.steps, "skip_first": cfg.skip_first,
               "steps_per_sec": 1 / avg, "tokens_per_sec": tokens_per_step / avg, "throughput_unit": "tokens",
               "first_loss": losses[0], "final_loss": losses[-1], "min_loss": min(losses),
               "peak_mem_bytes_per_rank": [peak], "log": str(log_path), "timestamp": ts}
    (Path(cfg.log_dir) / f"{run_id}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with open(cfg.benchmarks_file, "a") as f:
        f.write(json.dumps(summary) + "\n")
    print(f"avg {1 / avg:.2f} steps/s | {tokens_per_step / avg:,.0f} tokens/s | loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
