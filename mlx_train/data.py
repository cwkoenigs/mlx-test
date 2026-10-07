"""Deterministic datasets with a sharding hook ``shard(rank, size)``.

* ``synthetic``: noisy bigram chains. One random next-token map (a permutation of the vocab)
  is drawn from the seed; each token follows the map with probability 0.85, otherwise it is
  random. Loss starts near ln(vocab) and falls towards ~1.3 nats within a few hundred steps,
  so loss curves are comparable across runs. Generated on the fly, no files.
* ``bytes``: a byte-level LM over any text file (vocab 256). Contiguous slice per rank.
* ``images``: synthetic NHWC images with class-dependent low-frequency patterns + noise.

Batches are reproducible per (seed, rank, step), so a 2-node run with global batch B
sees exactly the same *set* of samples per step as a 1-node run with batch B.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import mlx.core as mx
import numpy as np


class SyntheticLM:
    tokens_per_sample: int

    def __init__(self, n_samples: int, seq_len: int, vocab_size: int, seed: int, rank: int = 0, size: int = 1):
        self.n_samples, self.seq_len, self.vocab_size, self.seed = n_samples, seq_len, vocab_size, seed
        self.rank, self.size = rank, size
        per = n_samples // size
        self.start, self.end = rank * per, (rank + 1) * per
        self.tokens_per_sample = seq_len
        self.p_follow = 0.85
        self.next_map = np.random.default_rng([seed, 99]).permutation(vocab_size)

    def shard(self, rank: int, size: int) -> "SyntheticLM":
        return SyntheticLM(self.n_samples, self.seq_len, self.vocab_size, self.seed, rank, size)

    def sample(self, idx: int) -> np.ndarray:
        rng = np.random.default_rng([self.seed, int(idx)])
        n = self.seq_len + 1
        follow = rng.random(n) < self.p_follow
        rand = rng.integers(0, self.vocab_size, size=n)
        seq = np.empty(n, dtype=np.int64)
        seq[0] = rand[0]
        for i in range(1, n):
            seq[i] = self.next_map[seq[i - 1]] if follow[i] else rand[i]
        return seq.astype(np.int32)

    def batches(self, batch_size: int, seed: int = 0) -> Iterator[tuple[mx.array, mx.array]]:
        step = 0
        while True:
            rng = np.random.default_rng([self.seed, seed, self.rank, step])
            idx = rng.integers(self.start, self.end, size=batch_size)
            arr = np.stack([self.sample(i) for i in idx])
            yield mx.array(arr[:, :-1]), mx.array(arr[:, 1:])
            step += 1


class BytesLM:
    def __init__(self, path: str, seq_len: int, seed: int, rank: int = 0, size: int = 1, data: np.ndarray | None = None):
        self.path, self.seq_len, self.seed, self.rank, self.size = path, seq_len, seed, rank, size
        self.data = data if data is not None else np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
        per = len(self.data) // size
        self.start, self.end = rank * per, (rank + 1) * per
        if self.end - self.start <= seq_len + 1:
            raise ValueError(f"{path}: too small for seq_len={seq_len} and {size} shards")
        self.tokens_per_sample = seq_len
        self.vocab_size = 256

    def shard(self, rank: int, size: int) -> "BytesLM":
        return BytesLM(self.path, self.seq_len, self.seed, rank, size, data=self.data)

    def batches(self, batch_size: int, seed: int = 0) -> Iterator[tuple[mx.array, mx.array]]:
        step = 0
        while True:
            rng = np.random.default_rng([self.seed, seed, self.rank, step])
            starts = rng.integers(self.start, self.end - self.seq_len - 1, size=batch_size)
            arr = np.stack([self.data[s : s + self.seq_len + 1] for s in starts]).astype(np.int32)
            yield mx.array(arr[:, :-1]), mx.array(arr[:, 1:])
            step += 1


class SyntheticImages:
    def __init__(self, n_classes: int, image_size: int, channels: int, seed: int, rank: int = 0, size: int = 1):
        self.n_classes, self.image_size, self.channels, self.seed = n_classes, image_size, channels, seed
        self.rank, self.size = rank, size
        rng = np.random.default_rng([seed, 7])
        # one low-frequency pattern per class: 4x4 noise upsampled to image_size
        small = rng.standard_normal((n_classes, 4, 4, channels))
        rep = image_size // 4
        self.patterns = np.repeat(np.repeat(small, rep, axis=1), rep, axis=2).astype(np.float32)
        self.tokens_per_sample = 1  # throughput reported as samples/sec

    def shard(self, rank: int, size: int) -> "SyntheticImages":
        return SyntheticImages(self.n_classes, self.image_size, self.channels, self.seed, rank, size)

    def batches(self, batch_size: int, seed: int = 0) -> Iterator[tuple[mx.array, mx.array]]:
        step = 0
        while True:
            rng = np.random.default_rng([self.seed, seed, self.rank, step])
            labels = rng.integers(0, self.n_classes, size=batch_size)
            x = self.patterns[labels] + 0.5 * rng.standard_normal((batch_size, self.image_size, self.image_size, self.channels)).astype(np.float32)
            yield mx.array(x), mx.array(labels.astype(np.int32))
            step += 1


def build_dataset(data_cfg, model_cfg):
    """data_cfg: DataConfig, model_cfg: ModelConfig (from mlx_train.config)."""
    if model_cfg.type == "cnn" or data_cfg.type == "images":
        return SyntheticImages(model_cfg.n_classes, model_cfg.image_size, model_cfg.channels, data_cfg.seed)
    if data_cfg.type == "synthetic":
        return SyntheticLM(data_cfg.n_samples, model_cfg.seq_len, model_cfg.vocab_size, data_cfg.seed)
    if data_cfg.type == "bytes":
        if not data_cfg.path:
            raise ValueError("data.type=bytes needs data.path")
        if model_cfg.vocab_size < 256:
            raise ValueError("data.type=bytes needs model.vocab_size >= 256")
        return BytesLM(data_cfg.path, model_cfg.seq_len, data_cfg.seed)
    raise ValueError(f"unknown dataset type {data_cfg.type!r}")
