"""Memory-mapped, block-streaming access to a synthetic CMD split (numpy only, no torch).

Same scheme as ``cmd_tcn/data.py``: arrays are member-major, so one batch = one contiguous slice
of members = one sequential read. For data-parallel runs every rank owns a contiguous member range
(proportional to ``rank_weights``) and reads blocks of its own batch size from it, so all ranks do
the same number of steps per epoch. Blocks are prefetched by a small thread pool; normalisation is
done on the GPU inside the training step.
"""

from __future__ import annotations

import json
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

ROOT_FILES = ("X_seq", "X_static", "enrollment_mask", "y", "outcome_enrollment_mask")
SEQ_LEN, SEQ_FEATURES = 105, 324
X_SEQ_BYTES_PER_MEMBER = SEQ_LEN * SEQ_FEATURES * 4


@dataclass
class Normalizer:
    """Per-feature standardisation, estimated from the first ``n_sample`` train members (identical
    to the PyTorch version, so both frameworks see the same inputs)."""

    seq_mean: np.ndarray
    seq_std: np.ndarray
    static_mean: np.ndarray
    static_std: np.ndarray
    clip: float = 10.0

    @classmethod
    def fit(cls, split_path: Path, n_sample: int = 4096, clip: float = 10.0) -> "Normalizer":
        xs = np.load(split_path / "X_seq.npy", mmap_mode="r")
        xst = np.load(split_path / "X_static.npy", mmap_mode="r")
        n = min(n_sample, xs.shape[0])
        seq = np.nan_to_num(np.asarray(xs[:n], dtype=np.float32)).reshape(-1, xs.shape[-1])
        sta = np.nan_to_num(np.asarray(xst[:n], dtype=np.float32))
        return cls(seq_mean=seq.mean(0), seq_std=seq.std(0) + 1e-6,
                   static_mean=sta.mean(0), static_std=sta.std(0) + 1e-6, clip=clip)

    def to_dict(self) -> dict:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}


def split_dir(data_root: str | Path, split: str) -> Path:
    p = Path(data_root).expanduser() / "arrays" / split
    if not (p / "X_seq.npy").exists():
        raise FileNotFoundError(f"no X_seq.npy under {p} (run scripts/generate_synthetic_cmd_arrays.py first)")
    return p


def n_members(data_root: str | Path, split: str) -> int:
    meta = json.loads((split_dir(data_root, split) / "metadata.json").read_text())
    return int(meta["X_seq_shape"][0])


def rank_batch_sizes(global_batch: int, size: int, weights: list[float] | None) -> list[int]:
    w = [float(x) for x in (weights or [1.0] * size)]
    if len(w) != size:
        raise ValueError(f"rank_weights has {len(w)} entries but there are {size} ranks")
    exact = [global_batch * wi / sum(w) for wi in w]
    sizes = [int(e) for e in exact]
    for i in sorted(range(size), key=lambda i: -(exact[i] - sizes[i]))[: global_batch - sum(sizes)]:
        sizes[i] += 1
    if any(s <= 0 for s in sizes):
        raise ValueError(f"global batch {global_batch} too small for {size} ranks with weights {w}")
    return sizes


def rank_ranges(n: int, size: int, weights: list[float] | None) -> list[tuple[int, int]]:
    """Contiguous member range per rank, proportional to weights."""
    w = [float(x) for x in (weights or [1.0] * size)]
    bounds = [0]
    for i in range(size):
        bounds.append(int(round(n * sum(w[: i + 1]) / sum(w))))
    bounds[-1] = n
    return [(bounds[i], bounds[i + 1]) for i in range(size)]


def make_blocks(start: int, end: int, block_size: int) -> list[tuple[int, int]]:
    return [(s, min(s + block_size, end)) for s in range(start, end, block_size)]


def plan_blocks(n: int, global_batch: int, size: int, weights, val_fraction: float, seed: int):
    """Per-rank (train_blocks, val_blocks) with the same count on every rank. Returns a list indexed
    by rank, so each rank can compute everyone's plan deterministically."""
    sizes = rank_batch_sizes(global_batch, size, weights)
    ranges = rank_ranges(n, size, weights)
    per_rank = [make_blocks(s, e, b) for (s, e), b in zip(ranges, sizes)]
    n_blocks = min(len(b) for b in per_rank)
    per_rank = [b[:n_blocks] for b in per_rank]
    perm = np.random.default_rng(seed).permutation(n_blocks)
    n_val = int(round(n_blocks * val_fraction))
    val_ids = set(perm[:n_val].tolist())
    out = []
    for blocks in per_rank:
        out.append(([b for i, b in enumerate(blocks) if i not in val_ids],
                    [b for i, b in enumerate(blocks) if i in val_ids]))
    return out, sizes


class BlockReader:
    """Reads whole member blocks from one split's memory-mapped arrays, with thread prefetch."""

    def __init__(self, split_path: Path, prefetch: int = 4):
        self.path = Path(split_path)
        self.prefetch = max(1, prefetch)
        self._arrays: dict[str, np.memmap] | None = None

    def _open(self):
        if self._arrays is None:
            self._arrays = {k: np.load(self.path / f"{k}.npy", mmap_mode="r") for k in ROOT_FILES}
        return self._arrays

    def read(self, block: tuple[int, int]) -> dict[str, np.ndarray]:
        a = self._open()
        s, e = block
        return {k: np.ascontiguousarray(a[k][s:e], dtype=np.float32) for k in ROOT_FILES}

    def iter(self, blocks: list[tuple[int, int]]) -> Iterator[dict[str, np.ndarray]]:
        with ThreadPoolExecutor(max_workers=self.prefetch) as pool:
            pending: deque = deque()
            it = iter(blocks)
            for _ in range(self.prefetch):
                b = next(it, None)
                if b is None:
                    break
                pending.append(pool.submit(self.read, b))
            while pending:
                fut = pending.popleft()
                b = next(it, None)
                if b is not None:
                    pending.append(pool.submit(self.read, b))
                yield fut.result()
