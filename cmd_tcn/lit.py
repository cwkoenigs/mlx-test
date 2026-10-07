"""LightningModule + LightningDataModule for the CMD TCN."""

from __future__ import annotations

import time
from pathlib import Path

import lightning as L
import torch
from torch import nn
from torch.utils.data import DataLoader

from cmd_tcn.data import BlockDataset, Normalizer, make_blocks, n_members, split_dir, train_val_blocks
from cmd_tcn.model import DEFAULT_DILATIONS, CMDTCNRegressor, count_params


class CMDDataModule(L.LightningDataModule):
    def __init__(self, data_root: str, batch_size: int = 256, num_workers: int = 0, val_fraction: float = 0.2,
                 seed: int = 0, limit_members: int | None = None, norm_sample: int = 4096,
                 train_split: str = "train__CMD", test_split: str = "one_year_forward_test__CMD"):
        super().__init__()
        self.save_hyperparameters()
        self.norm: Normalizer | None = None

    def setup(self, stage: str | None = None):
        hp = self.hparams
        tdir = split_dir(hp.data_root, hp.train_split)
        if self.norm is None:
            self.norm = Normalizer.fit(tdir, hp.norm_sample)
        n = n_members(hp.data_root, hp.train_split)
        if hp.limit_members:
            n = min(n, hp.limit_members)
        tr, va = train_val_blocks(n, hp.batch_size, hp.val_fraction, hp.seed)
        self.train_ds = BlockDataset(tdir, tr, self.norm)
        self.val_ds = BlockDataset(tdir, va, self.norm)
        self.n_train_members = sum(e - s for s, e in tr)
        self.n_val_members = sum(e - s for s, e in va)
        try:
            fdir = split_dir(hp.data_root, hp.test_split)
            nf = n_members(hp.data_root, hp.test_split)
            if hp.limit_members:
                nf = min(nf, hp.limit_members)
            self.test_ds = BlockDataset(fdir, make_blocks(nf, hp.batch_size), self.norm)
        except FileNotFoundError:
            self.test_ds = None

    def _loader(self, ds, shuffle: bool) -> DataLoader:
        w = self.hparams.num_workers
        return DataLoader(ds, batch_size=None, shuffle=shuffle, num_workers=w,
                          persistent_workers=w > 0, prefetch_factor=4 if w > 0 else None)

    def train_dataloader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_ds, shuffle=False) if self.test_ds is not None else None


class LitTCN(L.LightningModule):
    """Masked Huber regression on log1p(monthly allowed cost)."""

    def __init__(self, channels: int = 128, kernel: int = 3, dilations: tuple[int, ...] = DEFAULT_DILATIONS,
                 dropout: float = 0.1, static_dim: int = 64, head_hidden: int = 256,
                 lr: float = 1e-3, weight_decay: float = 1e-2, warmup_steps: int = 50, max_steps: int = 1000,
                 huber_delta: float = 1.0):
        super().__init__()
        self.save_hyperparameters()
        self.model = CMDTCNRegressor(channels=channels, kernel=kernel, dilations=tuple(dilations), dropout=dropout,
                                     static_dim=static_dim, head_hidden=head_hidden)
        self.loss_fn = nn.HuberLoss(reduction="none", delta=huber_delta)

    @property
    def n_params(self) -> int:
        return count_params(self.model)

    def forward(self, batch):
        return self.model(batch["x_seq"], batch["enroll"], batch["x_static"])

    def _step(self, batch, prefix: str):
        pred = self(batch)                                   # log1p dollars
        target = torch.log1p(batch["y"].clamp_min(0.0))
        mask = batch["mask"]
        denom = mask.sum().clamp_min(1.0)
        loss = (self.loss_fn(pred, target) * mask).sum() / denom
        with torch.no_grad():
            mae_log = ((pred - target).abs() * mask).sum() / denom
            mae_usd = ((torch.expm1(pred) - batch["y"]).abs() * mask).sum() / denom
        bs = batch["y"].shape[0]
        self.log(f"{prefix}/loss", loss, prog_bar=True, batch_size=bs, on_step=prefix == "train", on_epoch=True)
        self.log(f"{prefix}/mae_log1p", mae_log, batch_size=bs, on_step=False, on_epoch=True)
        self.log(f"{prefix}/mae_usd", mae_usd, batch_size=bs, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._step(batch, "test")

    def configure_optimizers(self):
        hp = self.hparams
        opt = torch.optim.AdamW(self.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
        warm, total = max(1, hp.warmup_steps), max(hp.max_steps, hp.warmup_steps + 1)

        def lr_lambda(step: int) -> float:
            if step < warm:
                return (step + 1) / warm
            p = (step - warm) / max(1, total - warm)
            return 0.1 + 0.9 * 0.5 * (1 + torch.cos(torch.tensor(p * 3.141592653589793)).item())

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}


class ThroughputCallback(L.Callback):
    """Steps/s, members/s and peak device memory, excluding the first ``skip`` steps (warm-up)."""

    def __init__(self, skip: int = 5):
        self.skip = skip
        self.t0 = None
        self.n_steps = 0
        self.n_members = 0
        self.peak_bytes = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step <= self.skip:
            self.t0 = time.perf_counter()
            self.n_steps, self.n_members = 0, 0
            return
        self.n_steps += 1
        self.n_members += batch["y"].shape[0]
        dev = pl_module.device.type
        if dev == "mps":
            self.peak_bytes = max(self.peak_bytes, torch.mps.driver_allocated_memory())
        elif dev == "cuda":
            self.peak_bytes = max(self.peak_bytes, torch.cuda.max_memory_allocated())
        if self.n_steps % trainer.log_every_n_steps == 0:
            s = self.summary()
            pl_module.log_dict({"perf/steps_per_sec": s["steps_per_sec"], "perf/members_per_sec": s["members_per_sec"]},
                               prog_bar=True, on_step=True, on_epoch=False)

    def summary(self) -> dict:
        el = (time.perf_counter() - self.t0) if self.t0 else 0.0
        return {
            "measured_steps": self.n_steps,
            "wall_s": el,
            "steps_per_sec": self.n_steps / el if el > 0 else 0.0,
            "members_per_sec": self.n_members / el if el > 0 else 0.0,
            "peak_device_bytes": self.peak_bytes,
        }
