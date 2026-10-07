"""Small helpers on top of mx.distributed used by every script in this repo.

Verified against mlx 0.32.3: ``mx.distributed.init / all_sum / all_gather / send / recv_like``.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

import mlx.core as mx
import numpy as np


def init_group(backend: str = "any") -> mx.distributed.Group:
    """Initialise the global group. Under ``mlx.launch`` the backend is picked by the
    launcher (ring/jaccl/mpi); in a single process this returns a size-1 group."""
    return mx.distributed.init(backend=backend)


def backend_name(group: Optional[mx.distributed.Group] = None) -> str:
    """Best-effort name of the backend ``mlx.launch`` selected (from its env vars)."""
    group = group or mx.distributed.init()
    if group.size() == 1:
        return "single"
    if os.environ.get("MLX_JACCL_COORDINATOR"):
        return "jaccl-ring" if os.environ.get("MLX_JACCL_RING") else "jaccl"
    if os.environ.get("MLX_HOSTFILE"):
        return "ring"
    if os.environ.get("MLX_MPI_LIBNAME") or os.environ.get("OMPI_COMM_WORLD_SIZE"):
        return "mpi"
    if os.environ.get("NCCL_HOST_IP"):
        return "nccl"
    return "unknown"


def barrier(group: Optional[mx.distributed.Group] = None) -> None:
    """Cheap barrier: a tiny all_sum on the CPU stream."""
    group = group or mx.distributed.init()
    if group.size() > 1:
        mx.eval(mx.distributed.all_sum(mx.array(1.0), group=group, stream=mx.cpu))


def gather_json(obj: Any, group: Optional[mx.distributed.Group] = None, max_bytes: int = 1 << 16) -> list:
    """all_gather an arbitrary JSON-serialisable object from every rank.

    Collectives only move arrays, so the object is packed into a fixed-size uint8
    buffer (4-byte length prefix) and gathered on the CPU stream. Returns a list
    indexed by rank."""
    group = group or mx.distributed.init()
    data = json.dumps(obj, default=str).encode()
    if len(data) > max_bytes:
        raise ValueError(f"gather_json payload too large: {len(data)} > {max_bytes} bytes")
    buf = np.zeros(max_bytes + 4, dtype=np.uint8)
    buf[:4] = np.frombuffer(len(data).to_bytes(4, "little"), dtype=np.uint8)
    buf[4 : 4 + len(data)] = np.frombuffer(data, dtype=np.uint8)
    arr = mx.array(buf)[None]
    gathered = mx.distributed.all_gather(arr, group=group, stream=mx.cpu)
    mx.eval(gathered)
    rows = np.array(gathered)
    out = []
    for row in rows:
        n = int.from_bytes(row[:4].tobytes(), "little")
        out.append(json.loads(row[4 : 4 + n].tobytes().decode()))
    return out


def all_sum_scalar(x: float, group: Optional[mx.distributed.Group] = None) -> float:
    group = group or mx.distributed.init()
    y = mx.distributed.all_sum(mx.array(float(x), dtype=mx.float64), group=group, stream=mx.cpu)
    mx.eval(y)
    return float(y.item())


def human_bytes(n: float) -> str:
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.2f} {unit}" if unit != "B" else f"{int(n)} B"
        n /= 1024
    return f"{n:.2f} TiB"


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    def lap(self) -> float:
        now = time.perf_counter()
        dt = now - self.t0
        self.t0 = now
        return dt
