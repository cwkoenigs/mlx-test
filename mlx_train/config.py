"""YAML -> dataclass config with dotted overrides (``--set model.d_model=512``)."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, get_type_hints

import yaml


@dataclass
class ModelConfig:
    type: str = "transformer"  # transformer | cnn
    n_layers: int = 4
    d_model: int = 256
    n_heads: int = 4
    d_ff: Optional[int] = None  # default: SwiGLU 8/3 * d_model
    vocab_size: int = 256
    seq_len: int = 256
    # cnn
    n_classes: int = 10
    image_size: int = 32
    channels: int = 3
    width: int = 32


@dataclass
class DataConfig:
    type: str = "synthetic"  # synthetic | bytes | images
    path: Optional[str] = None
    n_samples: int = 1 << 16
    seed: int = 1234


@dataclass
class TrainConfig:
    name: str = "run"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    global_batch_size: int = 32
    # Optional per-rank weights for heterogeneous nodes, e.g. [2, 1] gives rank 0 twice the
    # samples of rank 1 while keeping the global batch (and the loss curve) identical.
    rank_batch_weights: Optional[List[float]] = None
    steps: int = 100
    warmup_steps: int = 10
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    compile: bool = True
    dtype: str = "float32"  # float32 | bfloat16
    seed: int = 0
    log_every: int = 10
    skip_first: int = 5  # steps excluded from throughput averages (warm-up / compile)
    log_dir: str = "results/logs"
    benchmarks_file: str = "results/benchmarks.jsonl"
    checkpoint: Optional[str] = None
    all_reduce_size: int = 32 * 1024 * 1024
    accumulate_steps: int = 1  # micro-batches per optimizer update; gradients are exchanged once per update


def _from_dict(cls, d: dict, path: str = ""):
    fields = {f.name: f for f in dataclasses.fields(cls)}
    hints = get_type_hints(cls)  # resolves string annotations to real types
    kwargs = {}
    for k, v in d.items():
        if k not in fields:
            raise ValueError(f"unknown config key {path + k!r}; valid: {sorted(fields)}")
        t = hints[k]
        if dataclasses.is_dataclass(t) and isinstance(v, dict):
            v = _from_dict(t, v, path + k + ".")
        kwargs[k] = v
    return cls(**kwargs)


def apply_override(cfg, dotted: str):
    """``model.d_model=512`` / ``steps=50`` / ``compile=false``. Values are parsed as YAML."""
    key, _, raw = dotted.partition("=")
    if not _:
        raise ValueError(f"override must look like key=value, got {dotted!r}")
    value = yaml.safe_load(raw)
    parts = key.split(".")
    obj = cfg
    for p in parts[:-1]:
        obj = getattr(obj, p)
    if not hasattr(obj, parts[-1]):
        raise ValueError(f"unknown config key {key!r}")
    setattr(obj, parts[-1], value)


def load_config(path: str | None, overrides: list[str] | None = None) -> TrainConfig:
    if path:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        cfg = _from_dict(TrainConfig, raw)
        if "name" not in raw:
            cfg.name = Path(path).stem
    else:
        cfg = TrainConfig()
    for o in overrides or []:
        apply_override(cfg, o)
    return cfg


def to_dict(cfg) -> dict:
    return dataclasses.asdict(cfg)
