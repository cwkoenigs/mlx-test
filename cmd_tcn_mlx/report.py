"""Side-by-side table of every TCN run in results/tcn/*.summary.json (PyTorch and MLX).

    .venv/bin/python -m cmd_tcn_mlx.report [--out results/tcn/comparison.md]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="results/tcn")
    ap.add_argument("--out", default="results/tcn/comparison.md")
    args = ap.parse_args(argv)
    rows = []
    for p in sorted(Path(args.dir).glob("*.summary.json")):
        s = json.loads(p.read_text())
        c = s.get("config", {})
        m, t = s.get("metrics", {}), s.get("test_metrics", {})
        rows.append({
            "run": s["run_id"], "framework": s.get("framework", "?"), "ranks": s.get("ranks", 1),
            "backend": s.get("backend", "-"), "hosts": ", ".join(s.get("hosts", [s.get("host", "?")])),
            "channels": c.get("channels"), "batch": c.get("batch_size") or c.get("global_batch"),
            "precision": c.get("precision") or c.get("dtype"), "params": s.get("params"),
            "steps_s": s.get("steps_per_sec", 0.0), "members_s": s.get("members_per_sec", 0.0),
            "gib_s": s.get("gib_per_sec_x_seq", 0.0), "peak": s.get("peak_device_bytes", 0) / 2**30,
            "steps": s.get("measured_steps"), "val": m.get("val/loss"), "val_mae_usd": m.get("val/mae_usd"),
            "test": t.get("test/loss"), "test_mae_usd": t.get("test/mae_usd"),
        })
    if not rows:
        print(f"no summaries in {args.dir}")
        return 0
    f = lambda v, fmt: (fmt.format(v) if v is not None else "-")
    lines = ["# CMD TCN: PyTorch vs MLX, one Mac vs two", "",
             "| Run | Framework | Macs | Backend | Channels | Batch | Precision | Params | Steps/s | Members/s | X_seq GiB/s | Peak mem | Val loss | Val MAE $ | Fwd-test loss | Fwd-test MAE $ |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["framework"], r["ranks"], r["channels"] or 0, r["batch"] or 0)):
        lines.append(f"| {r['run']} | {r['framework']} | {r['ranks']} ({r['hosts']}) | {r['backend']} | {r['channels']} | {r['batch']} | "
                     f"{r['precision']} | {f(r['params'], '{:,}')} | {r['steps_s']:.2f} | {r['members_s']:,.0f} | {r['gib_s']:.2f} | "
                     f"{r['peak']:.1f} GiB | {f(r['val'], '{:.4f}')} | {f(r['val_mae_usd'], '{:,.0f}')} | {f(r['test'], '{:.4f}')} | {f(r['test_mae_usd'], '{:,.0f}')} |")
    text = "\n".join(lines) + "\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(text)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
