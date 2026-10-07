"""Train the CMD TCN with PyTorch Lightning.

    .venv/bin/python -m cmd_tcn.train --data-root ~/data/cmd --max-steps 500 --batch-size 256
    .venv/bin/python -m cmd_tcn.train --data-root ~/data/cmd --epochs 1 --num-workers 4 --test-forward
    .venv/bin/python -m cmd_tcn.train --data-root <smoke-out> --max-steps 5 --batch-size 32 --accelerator cpu

Writes results/tcn/<run>.summary.json (config, parameter count, steps/s, members/s, peak device
memory, final metrics). Lightning logs go to results/tcn/lightning_logs/.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import lightning as L
import torch

from cmd_tcn.lit import CMDDataModule, LitTCN, ThroughputCallback
from cmd_tcn.model import DEFAULT_DILATIONS


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="output-root given to generate_synthetic_cmd_arrays.py")
    ap.add_argument("--batch-size", type=int, default=256, help="members per step (= contiguous block read)")
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many optimizer steps")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=0, help="DataLoader workers (try 2-4 on the Studio)")
    ap.add_argument("--limit-members", type=int, default=None, help="use only the first N members of each split")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--val-batches", type=int, default=20, help="validation blocks per epoch (0 = all)")
    ap.add_argument("--test-forward", action="store_true", help="evaluate on one_year_forward_test__CMD at the end")
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--kernel", type=int, default=3)
    ap.add_argument("--dilations", default=",".join(map(str, DEFAULT_DILATIONS)))
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--warmup-steps", type=int, default=50)
    ap.add_argument("--accelerator", default="auto", help="auto | mps | cpu")
    ap.add_argument("--precision", default="32-true", help="32-true | bf16-mixed | 16-mixed")
    ap.add_argument("--compile", action="store_true", help="torch.compile the model (experimental on MPS)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--out-dir", default="results/tcn")
    ap.add_argument("--name", default="tcn")
    ap.add_argument("--checkpoint", action="store_true", help="save a Lightning checkpoint at the end")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    L.seed_everything(args.seed, workers=True)
    torch.set_float32_matmul_precision("high")
    dilations = tuple(int(d) for d in args.dilations.split(","))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{args.name}-{time.strftime('%Y%m%d-%H%M%S')}"

    dm = CMDDataModule(args.data_root, batch_size=args.batch_size, num_workers=args.num_workers,
                       val_fraction=args.val_fraction, seed=args.seed, limit_members=args.limit_members)
    dm.setup()
    steps_per_epoch = len(dm.train_ds)
    max_steps = args.max_steps if args.max_steps else steps_per_epoch * args.epochs
    model = LitTCN(channels=args.channels, kernel=args.kernel, dilations=dilations, dropout=args.dropout,
                   lr=args.lr, weight_decay=args.weight_decay, warmup_steps=args.warmup_steps, max_steps=max_steps)
    if args.compile:
        model.model = torch.compile(model.model)

    perf = ThroughputCallback(skip=5)
    trainer = L.Trainer(
        accelerator=args.accelerator, devices=1, precision=args.precision,
        max_steps=max_steps, max_epochs=args.epochs if not args.max_steps else -1,
        limit_val_batches=args.val_batches or 1.0, num_sanity_val_steps=0,
        log_every_n_steps=args.log_every, enable_checkpointing=args.checkpoint,
        default_root_dir=str(out_dir), callbacks=[perf], enable_model_summary=False,
    )
    print(f"== {run_id}: TCN channels={args.channels} kernel={args.kernel} dilations={dilations} "
          f"params={model.n_params:,} | receptive field {model.model.tcn.receptive_field} bins")
    print(f"   data {args.data_root}: train {dm.n_train_members:,} members / {steps_per_epoch} blocks of {args.batch_size}, "
          f"val {dm.n_val_members:,} members | max_steps={max_steps} workers={args.num_workers} "
          f"precision={args.precision} accelerator={trainer.accelerator.__class__.__name__}")
    t0 = time.perf_counter()
    trainer.fit(model, datamodule=dm)
    fit_s = time.perf_counter() - t0
    metrics = {k: float(v) for k, v in trainer.callback_metrics.items()}
    test_metrics = {}
    if args.test_forward and dm.test_ds is not None:
        res = trainer.test(model, datamodule=dm, verbose=False)
        test_metrics = {k: float(v) for k, v in (res[0] if res else {}).items()}
    s = perf.summary()
    summary = {
        "run_id": run_id, "framework": "torch-lightning", "torch": torch.__version__, "lightning": L.__version__,
        "host": socket.gethostname(), "device": str(model.device),
        "config": vars(args) | {"dilations": list(dilations), "max_steps": max_steps},
        "params": model.n_params, "receptive_field": model.model.tcn.receptive_field,
        "train_members": dm.n_train_members, "val_members": dm.n_val_members, "steps_per_epoch": steps_per_epoch,
        "fit_wall_s": fit_s, **s,
        "gib_per_sec_x_seq": s["members_per_sec"] * 105 * 324 * 4 / 2**30,
        "metrics": metrics, "test_metrics": test_metrics,
    }
    (out_dir / f"{run_id}.summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n== {run_id} ==")
    print(f"params {model.n_params:,} | {s['measured_steps']} measured steps | {s['steps_per_sec']:.2f} steps/s | "
          f"{s['members_per_sec']:,.0f} members/s ({summary['gib_per_sec_x_seq']:.2f} GiB/s of X_seq) | "
          f"peak device mem {s['peak_device_bytes'] / 2**30:.2f} GiB | fit wall {fit_s:.1f}s")
    for k in ("train/loss_epoch", "val/loss", "val/mae_log1p", "val/mae_usd"):
        if k in metrics:
            print(f"  {k:18s} {metrics[k]:.4f}")
    for k, v in test_metrics.items():
        print(f"  {k:18s} {v:.4f}")
    print(f"summary: {out_dir / (run_id + '.summary.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
