"""Memory-mapped, block-streaming access to a synthetic CMD split.

The arrays are member-major, so a *contiguous* slice of members is one sequential read from disk.
``BlockDataset`` therefore treats each dataset item as a whole batch of ``block_size`` consecutive
members; the DataLoader is used with ``batch_size=None`` and shuffles the *order of blocks*. That
gives near-disk-speed streaming of the 107 GiB ``X_seq`` with no per-sample gather.

Memory maps are opened lazily inside each worker (macOS DataLoader workers are spawned, and a
pickled ``np.memmap`` would otherwise copy the array data).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT_FILES = ("X_seq", "X_static", "enrollment_mask", "y", "outcome_enrollment_mask")


@dataclass
class Normalizer:
    """Per-feature standardisation for X_seq and X_static, estimated from a sample of train members."""

    seq_mean: np.ndarray
    seq_std: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray
    clip: float = 10.0

    @classmethod
    def fit(cls, split_dir: Path, n_sample: int = 4096, clip: float = 10.0) -> "Normalizer":
        xs = np.load(split_dir / "X_seq.npy", mmap_mode="r")
        xst = np.load(split_dir / "X_static.npy", mmap_mode="r")
        n = min(n_sample, xs.shape[0])
        seq = np.nan_to_num(np.asarray(xs[:n], dtype=np.float32)).reshape(-1, xs.shape[-1])
        sta = np.nan_to_num(np.asarray(xst[:n], dtype=np.float32))
        return cls(
            seq_mean=seq.mean(0), seq_std=seq.std(0) + 1e-6,
            static_mean=sta.mean(0), static_std=sta.std(0) + 1e-6, clip=clip,
        )

    def to_dict(self) -> dict:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "Normalizer":
        return cls(**{k: (np.asarray(v, dtype=np.float32) if isinstance(v, list) else v) for k, v in d.items()})


def split_dir(data_root: str | Path, split: str) -> Path:
    p = Path(data_root).expanduser() / "arrays" / split
    if not (p / "X_seq.npy").exists():
        raise FileNotFoundError(f"no X_seq.npy under {p} (run scripts/generate_synthetic_cmd_arrays.py first)")
    return p


def n_members(data_root: str | Path, split: str) -> int:
    meta = json.loads((split_dir(data_root, split) / "metadata.json").read_text())
    return int(meta["X_seq_shape"][0])


def make_blocks(n: int, block_size: int) -> list[tuple[int, int]]:
    return [(s, min(s + block_size, n)) for s in range(0, n, block_size)]


def train_val_blocks(n: int, block_size: int, val_fraction: float, seed: int) -> tuple[list, list]:
    """Deterministic 80/20 (by default) split of contiguous member blocks."""
    blocks = make_blocks(n, block_size)
    perm = np.random.default_rng(seed).permutation(len(blocks))
    n_val = int(round(len(blocks) * val_fraction))
    val_ids = set(perm[:n_val].tolist())
    train = [b for i, b in enumerate(blocks) if i not in val_ids]
    val = [b for i, b in enumerate(blocks) if i in val_ids]
    return train, val


class BlockDataset(Dataset):
    """Each item is a batch: all tensors for members ``start:end`` of one split."""

    def __init__(self, split_path: Path, blocks: list[tuple[int, int]], normalizer: Normalizer | None):
        self.split_path = Path(split_path)
        self.blocks = blocks
        self.norm = normalizer
        self._arrays: dict[str, np.memmap] | None = None

    def _open(self) -> dict[str, np.memmap]:
        if self._arrays is None:
            self._arrays = {k: np.load(self.split_path / f"{k}.npy", mmap_mode="r") for k in ROOT_FILES}
        return self._arrays

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        a = self._open()
        s, e = self.blocks[i]
        x_seq = np.nan_to_num(np.asarray(a["X_seq"][s:e], dtype=np.float32))
        x_static = np.nan_to_num(np.asarray(a["X_static"][s:e], dtype=np.float32))
        if self.norm is not None:
            x_seq = np.clip((x_seq - self.norm.seq_mean) / self.norm.seq_std, -self.norm.clip, self.norm.clip)
            x_static = np.clip((x_static - self.norm.static_mean) / self.norm.static_std, -self.norm.clip, self.norm.clip)
        y = np.asarray(a["y"][s:e], dtype=np.float32)
        out_mask = np.asarray(a["outcome_enrollment_mask"][s:e], dtype=np.float32)
        out_mask = out_mask * np.isfinite(y)  # NaN targets are never scored
        return {
            "x_seq": torch.from_numpy(np.ascontiguousarray(x_seq)),            # [B, 105, 324]
            "enroll": torch.from_numpy(np.array(a["enrollment_mask"][s:e], dtype=np.float32)),  # [B, 105] (copy: memmap is read-only)
            "x_static": torch.from_numpy(np.ascontiguousarray(x_static)),      # [B, 255]
            "y": torch.from_numpy(np.nan_to_num(y)),                           # [B, 12] monthly allowed cost
            "mask": torch.from_numpy(np.ascontiguousarray(out_mask)),          # [B, 12] 1 = enrolled, scored
        }
