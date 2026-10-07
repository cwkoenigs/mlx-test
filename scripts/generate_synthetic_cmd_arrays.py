#!/usr/bin/env python3
"""Generate entirely synthetic CMD member-level arrays: train__CMD and one_year_forward_test__CMD.

Implements docs/synthetic_cmd_generator_prompt.md. Needs numpy (and pyarrow for the parquet
labels file). The contract (shapes, families, column names) lives in synthetic_cmd_spec.py next
to this file; copy both files into the member-level repository's scripts/ directory to use it
there.

  generate (default)  python scripts/generate_synthetic_cmd_arrays.py --output-root OUT           # smoke: --scale 0.001
                      python scripts/generate_synthetic_cmd_arrays.py --output-root OUT --dry-run --full
                      python scripts/generate_synthetic_cmd_arrays.py --output-root OUT --full --resume
  validate            python scripts/generate_synthetic_cmd_arrays.py validate --output-root OUT
  reassemble          python scripts/generate_synthetic_cmd_arrays.py reassemble --output-root OUT --split train__CMD
  spec                python scripts/generate_synthetic_cmd_arrays.py spec

Design: every array is written sequentially in member blocks through a bounded-memory .npy
writer (header first, then raw rows), never materialised whole. Each block's values come from
numpy.random.default_rng([seed, split, file, block]), so --resume and chunked vs. assembled
layouts produce identical bytes. Member "profiles" (enrollment masks, risk, New/partial flags)
are derived from fixed 4096-member chunks with their own seeds, so every file sees the same
member.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import shutil
import sys
import time
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import synthetic_cmd_spec as spec  # noqa: E402

GENERATOR_VERSION = "1.0"
PROFILE_CHUNK = 4096
FULL_ROWS_PER_CHUNK = 39_452  # 5 GiB member-major X_seq chunks at full scale -> 22 chunks per split
TAG_PROFILE = 0x7F0F11E
TAG_SPLIT = 0x5B117


def crc(name: str) -> int:
    return zlib.crc32(name.encode()) & 0xFFFFFFFF


def rss_bytes() -> int:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r) if sys.platform == "darwin" else int(r) * 1024


def gib(n: float) -> float:
    return n / spec.GIB


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------------------------------
# Bounded-memory, resumable .npy writer
# ------------------------------------------------------------------------------------------
class NpyWriter:
    """Writes a C-order .npy file row by row. A ``.partial`` file plus a ``.progress`` JSON hold
    the state, so an interrupted run resumes at the last checkpoint; the rename to the final
    name is the completion marker."""

    def __init__(self, path: Path, shape: tuple, dtype, resume: bool):
        self.final = Path(path)
        self.partial = self.final.with_name(self.final.name + ".partial")
        self.progress = self.final.with_name(self.final.name + ".progress")
        self.shape = tuple(int(s) for s in shape)
        self.dtype = np.dtype(dtype)
        self.row_bytes = int(np.prod(self.shape[1:], dtype=np.int64)) * self.dtype.itemsize
        self.rows_done = 0
        self.state: dict = {}
        header = {"descr": np.lib.format.dtype_to_descr(self.dtype), "fortran_order": False, "shape": self.shape}
        if resume and self.partial.exists() and self.progress.exists():
            st = json.loads(self.progress.read_text())
            if st.get("shape") == list(self.shape) and st.get("dtype") == self.dtype.str:
                self.rows_done = int(st["rows_done"])
                self.state = st.get("state", {})
                self.header_len = int(st["header_len"])
                self.f = open(self.partial, "r+b")
                self.f.seek(self.header_len + self.rows_done * self.row_bytes)
                self.f.truncate()
                return
        self.f = open(self.partial, "wb")
        np.lib.format.write_array_header_1_0(self.f, header)
        self.header_len = self.f.tell()

    def write_rows(self, arr: np.ndarray) -> None:
        arr = np.ascontiguousarray(arr, dtype=self.dtype)
        assert arr.shape[1:] == self.shape[1:], (arr.shape, self.shape)
        self.f.write(memoryview(arr).cast("B"))
        self.rows_done += arr.shape[0]

    def checkpoint(self, **state) -> None:
        self.f.flush()
        self.state.update(state)
        self.progress.write_text(json.dumps({"rows_done": self.rows_done, "shape": list(self.shape),
                                             "dtype": self.dtype.str, "header_len": self.header_len,
                                             "state": self.state}))

    def finish(self) -> None:
        assert self.rows_done == self.shape[0], f"{self.final}: wrote {self.rows_done} of {self.shape[0]} rows"
        self.f.flush()
        os.fsync(self.f.fileno())
        self.f.close()
        os.replace(self.partial, self.final)
        if self.progress.exists():
            self.progress.unlink()

    def abort(self) -> None:
        self.f.close()


def block_rows_for(row_bytes: int, block_bytes: int) -> int:
    return int(min(1 << 16, max(64, block_bytes // max(1, row_bytes))))


# ------------------------------------------------------------------------------------------
# Member profiles: identical across files, independent of block sizes
# ------------------------------------------------------------------------------------------
class Profiles:
    def __init__(self, split: str, n: int, seed: int):
        self.split, self.n, self.seed = split, n, seed
        s = spec.SPLITS[split]
        self.partial_target = spec.scaled_count(s["partial_exposure_members"], split, n)
        self.new_target = spec.scaled_count(s["strat_new"], split, n)
        self._cache: dict[int, dict] = {}

    def _quota(self, target: int, start: int, end: int) -> int:
        return (target * end) // self.n - (target * start) // self.n

    def chunk(self, c: int) -> dict:
        if c in self._cache:
            return self._cache[c]
        start, end = c * PROFILE_CHUNK, min((c + 1) * PROFILE_CHUNK, self.n)
        m = end - start
        rng = np.random.default_rng([self.seed, spec.SPLITS[self.split]["index"], TAG_PROFILE, c])
        risk = np.exp(rng.standard_normal(m) * 0.9).astype(np.float32)
        high = rng.random(m) < 0.005  # a small number of high-cost members
        risk[high] *= 15.0
        is_new = np.zeros(m, dtype=bool)
        is_new[rng.choice(m, size=self._quota(self.new_target, start, end), replace=False)] = True
        partial = np.zeros(m, dtype=bool)
        partial[rng.choice(m, size=self._quota(self.partial_target, start, end), replace=False)] = True
        enrolled_months = np.where(partial, rng.integers(1, spec.OUT_LEN, size=m), spec.OUT_LEN)
        outcome = (np.arange(spec.OUT_LEN)[None, :] < enrolled_months[:, None]).astype(np.uint8)
        # 105-bin history: contiguous span; New members only have a recent short span
        start_bin = np.where(is_new, rng.integers(spec.SEQ_LEN - 16, spec.SEQ_LEN - 1, size=m),
                             np.where(rng.random(m) < 0.65, 0, rng.integers(0, spec.SEQ_LEN - 17, size=m)))
        end_bin = np.where(rng.random(m) < 0.92, spec.SEQ_LEN, rng.integers(1, spec.SEQ_LEN + 1, size=m))
        end_bin = np.maximum(end_bin, start_bin + 1)
        bins = np.arange(spec.SEQ_LEN)[None, :]
        enroll = ((bins >= start_bin[:, None]) & (bins < end_bin[:, None])).astype(np.uint8)
        gaps = (rng.random((m, spec.SEQ_LEN)) < 0.02) & ~is_new[:, None]
        enroll[gaps] = 0
        enroll[np.arange(m), start_bin] = 1  # every member keeps at least one enrolled bin
        archetype = rng.choice(6, size=m, p=[0.55, 0.20, 0.10, 0.08, 0.05, 0.02]).astype(np.int8)
        p_multi = np.array([0.06, 0.04, 0.03, 0.03, 0.02, 0.01], dtype=np.float32)
        multilabel = (rng.random((m, 6), dtype=np.float32) < p_multi[None, :] * np.minimum(risk, 3)[:, None])
        multilabel[np.arange(m), archetype] |= (archetype > 0) & (rng.random(m) < 0.5)
        avail = {k: (outcome.astype(bool) & (rng.random((m, spec.OUT_LEN)) < 0.95)).astype(np.uint8)
                 for k in ("cluster", "member", "crude")}
        prof = {"risk": risk, "is_new": is_new, "partial": partial, "outcome": outcome, "enroll": enroll,
                "archetype": archetype, "multilabel": multilabel.astype(np.float32),
                "prospective": (rng.random(m) < 0.95).astype(np.uint8), **{f"avail_{k}": v for k, v in avail.items()}}
        if len(self._cache) > 16:
            self._cache.pop(next(iter(self._cache)))
        self._cache[c] = prof
        return prof

    def rows(self, start: int, end: int) -> dict:
        c0, c1 = start // PROFILE_CHUNK, (end - 1) // PROFILE_CHUNK
        parts = [self.chunk(c) for c in range(c0, c1 + 1)]
        out = {}
        for k in parts[0]:
            arr = parts[0][k] if len(parts) == 1 else np.concatenate([p[k] for p in parts])
            off = start - c0 * PROFILE_CHUNK
            out[k] = arr[off: off + (end - start)]
        return out

    def scan(self, key: str) -> np.ndarray:
        """Concatenate one profile field over all members (bool/uint8 fields only; cheap)."""
        return np.concatenate([self.chunk(c)[key] for c in range(math.ceil(self.n / PROFILE_CHUNK))])


# ------------------------------------------------------------------------------------------
# Value generation per block
# ------------------------------------------------------------------------------------------
def _classify_seq(name: str) -> str:
    if name.startswith("CUM_"):
        return "cum"
    if "_DELTA_" in name or "_MA_" in name or "TREND" in name:
        return "signed"
    if name.startswith(("WEEKS_SINCE", "TIME_SINCE", "MONTHS_SINCE", "GAP_LENGTH", "CONTINUOUS_ENROLLMENT")):
        return "since"
    if "SPEND" in name or name == "DAYS_SUPPLY_AVG":
        return "spend"
    if (name.endswith(("_FLAG", "_FLAG_T", "_ANY", "_OPEN", "_CLOSED", "_START")) or "OBSERVED" in name
            or name.startswith(("DISCHARGE_", "RECENT_", "READMISSION", "EXPIRED", "ELIGIBILITY", "NEW_MEMBER",
                                "RETURNING_", "NEW_PRIOR_AUTH", "NEW_REFERRING", "NEW_CHRONIC_DX", "NEW_ACUTE_DX"))
            or name in ("NEW_DRUG_START", "COMPLEX_CARE_ENROLLMENT_THIS_PERIOD", "OBSERVATION_TO_IP_THIS_PERIOD",
                        "ER_TO_IP_THIS_PERIOD")):
        return "binary"
    if name.endswith(("_RATE", "_FRACTION", "_PCT", "_HHI", "MATURITY", "_RATE_4W", "_RATE_12W")) or "ACCUM" in name:
        return "rate"
    return "count"


def _classify_static(name: str) -> str:
    if name.startswith("AGE_AS_OF"):
        return "age"
    if name.endswith(("_FLAG", "_INDC", "_EVER", "_COMPOSITE")) or name.startswith("CM_ANY"):
        return "binary"
    if name.startswith("ZIP5_AREA_SUMMARY"):
        return "rate"
    if "TREND" in name:
        return "signed"
    if "PMPM" in name or "SPEND" in name or "PREMIUM" in name or name.endswith("_AMT"):
        return "spend"
    if "RATE" in name or "RATIO" in name or "FRACTION" in name or "DENSITY" in name:
        return "rate"
    if "MONTHS_SINCE" in name or "MONTHS_TO" in name or name.endswith("_MONTHS"):
        return "since"
    return "count"


SEQ_GROUPS: dict[str, np.ndarray] = {}
STATIC_GROUPS: dict[str, np.ndarray] = {}
for _i, _n in enumerate(spec.SEQ_COLUMNS):
    SEQ_GROUPS.setdefault(_classify_seq(_n), []).append(_i)
for _i, _n in enumerate(spec.STATIC_COLUMNS):
    STATIC_GROUPS.setdefault(_classify_static(_n), []).append(_i)
SEQ_GROUPS = {k: np.array(v) for k, v in SEQ_GROUPS.items()}
STATIC_GROUPS = {k: np.array(v) for k, v in STATIC_GROUPS.items()}


def _fill_groups(out: np.ndarray, groups: dict, rng, risk: np.ndarray, time_axis: bool) -> None:
    """Fill columns of ``out`` ([m, T, C] or [m, C]) by semantic group. risk is [m]."""
    m = out.shape[0]
    lead = (m, out.shape[1]) if time_axis else (m,)
    rk = np.minimum(risk, 4.0).astype(np.float32)
    rk = rk[:, None, None] if time_axis else rk[:, None]
    for group, cols in groups.items():
        shape = (*lead, len(cols))
        if group == "spend":
            nz = rng.random(shape, dtype=np.float32) < 0.12 * rk
            amt = np.exp(5.0 + 1.2 * rng.standard_normal(shape, dtype=np.float32)) * rk
            out[..., cols] = nz * amt
        elif group == "count":
            out[..., cols] = rng.poisson(0.35 * rk, size=shape).astype(np.float32)
        elif group == "binary":
            out[..., cols] = (rng.random(shape, dtype=np.float32) < 0.08 * rk).astype(np.float32)
        elif group == "rate":
            out[..., cols] = rng.random(shape, dtype=np.float32)
        elif group == "since":
            out[..., cols] = rng.integers(0, spec.SEQ_LEN, size=shape).astype(np.float32)
        elif group == "signed":
            out[..., cols] = rng.standard_normal(shape, dtype=np.float32) * 50.0
        elif group == "age":
            out[..., cols] = rng.integers(18, 95, size=shape).astype(np.float32)
        elif group == "cum":
            inc = rng.poisson(0.3 * rk, size=shape).astype(np.float32)
            out[..., cols] = np.cumsum(inc, axis=1) if time_axis else inc
        else:
            raise KeyError(group)


def gen_root_block(name: str, split: str, prof: dict, rng, pattern: str) -> np.ndarray:
    m = len(prof["risk"])
    risk = prof["risk"]
    enroll = prof["enroll"].astype(np.float32)
    outcome = prof["outcome"].astype(np.float32)
    dtype = np.dtype(spec.root_array_dtype(name))
    trailing = spec.root_array_shape(name, m)[1:]
    if name == "X_seq":
        out = np.empty((m, *trailing), dtype=np.float32)
        if pattern == "fast-io":
            out[...] = rng.random(out.shape, dtype=np.float32)
        else:
            _fill_groups(out, SEQ_GROUPS, rng, risk, time_axis=True)
        out *= enroll[:, :, None]
        return out
    if name == "X_static":
        out = np.empty((m, *trailing), dtype=np.float32)
        if pattern == "fast-io":
            out[...] = rng.random(out.shape, dtype=np.float32)
        else:
            _fill_groups(out, STATIC_GROUPS, rng, risk, time_axis=False)
        return out
    if name == "X_plan_cat":
        vocab = np.array(list(spec.SPLITS[split]["plan_cat_vocab_sizes"].values()), dtype=np.int64)
        u = rng.random((m, len(vocab)), dtype=np.float32) ** 1.5  # skewed towards low ids
        return np.minimum((u * vocab[None, :]).astype(np.int64), vocab[None, :] - 1)
    if name == "X_prospective_static":
        return rng.random((m, *trailing), dtype=np.float32) * prof["prospective"][:, None]
    if name == "prospective_static_mask":
        return prof["prospective"].astype(np.float32)
    if name in ("X_cluster_future", "X_member_future", "X_crude_future"):
        key = {"X_cluster_future": "avail_cluster", "X_member_future": "avail_member", "X_crude_future": "avail_crude"}[name]
        return rng.random((m, *trailing), dtype=np.float32) * prof[key][:, :, None]
    if name in ("cluster_profile_mask", "member_future_mask", "crude_future_mask"):
        key = {"cluster_profile_mask": "avail_cluster", "member_future_mask": "avail_member", "crude_future_mask": "avail_crude"}[name]
        return np.repeat(prof[key][:, :, None].astype(np.float32), trailing[-1], axis=2)
    if name == "enrollment_mask":
        return enroll
    if name == "outcome_enrollment_mask":
        return outcome
    if name == "y":
        rk = np.minimum(risk, 4.0)[:, None]
        nz = rng.random((m, spec.OUT_LEN), dtype=np.float32) < 0.55 * np.minimum(rk, 1.8)
        amt = np.exp(5.5 + 1.1 * rng.standard_normal((m, spec.OUT_LEN), dtype=np.float32)) * risk[:, None]
        y = (nz * amt).astype(np.float32)
        y[outcome == 0] = np.nan
        return y
    if name == "event_y":
        p = np.array([0.010, 0.020, 0.005, 0.015, 0.008], dtype=np.float32)
        ev = (rng.random((m, spec.OUT_LEN, 5), dtype=np.float32) < p[None, None, :] * np.minimum(risk, 4.0)[:, None, None])
        ev = ev.astype(np.float32)
        ev[outcome == 0] = np.nan
        return ev
    if name == "spend_archetype":
        return prof["archetype"]
    if name == "spend_archetype_multilabel":
        return prof["multilabel"]
    raise KeyError(name)


def block_rng(seed: int, split: str, file_tag: str, block: int):
    return np.random.default_rng([seed, spec.SPLITS[split]["index"], crc(file_tag), block])


# ------------------------------------------------------------------------------------------
# Row sinks: one assembled .npy, or X_seq chunk files
# ------------------------------------------------------------------------------------------
class ChunkSink:
    """Routes consecutive rows into X_seq_chunk_XXX.npy files and records a manifest."""

    def __init__(self, split_dir: Path, shape: tuple, dtype, rows_per_chunk: int, resume: bool, rows_done: int):
        self.dir, self.shape, self.dtype, self.rpc = split_dir, shape, np.dtype(dtype), rows_per_chunk
        self.n_chunks = math.ceil(shape[0] / rows_per_chunk)
        self.manifest_path = split_dir / "X_seq_manifest.json"
        self.rows_done = rows_done
        self.crcs: dict[str, list] = {}
        self.writer: NpyWriter | None = None
        if resume and self.manifest_path.with_suffix(".partial.json").exists():
            self.crcs = json.loads(self.manifest_path.with_suffix(".partial.json").read_text()).get("row_crc32", {})
        self._open_current(resume)

    def chunk_path(self, i: int) -> Path:
        return self.dir / f"X_seq_chunk_{i:03d}.npy"

    def chunk_rows(self, i: int) -> int:
        return min(self.rpc, self.shape[0] - i * self.rpc)

    def _open_current(self, resume: bool):
        i = self.rows_done // self.rpc
        if i >= self.n_chunks or self.rows_done >= self.shape[0]:
            self.writer = None
            return
        self.writer = NpyWriter(self.chunk_path(i), (self.chunk_rows(i), *self.shape[1:]), self.dtype, resume)
        expected = self.rows_done - i * self.rpc
        if self.writer.rows_done != expected:  # partial chunk out of step with the progress -> restart chunk
            self.writer.abort()
            self.writer = NpyWriter(self.chunk_path(i), (self.chunk_rows(i), *self.shape[1:]), self.dtype, False)
            self.rows_done = i * self.rpc

    def write_rows(self, arr: np.ndarray):
        while arr.shape[0]:
            i = self.rows_done // self.rpc
            room = (i + 1) * self.rpc - self.rows_done
            part, arr = arr[:room], arr[room:]
            key = str(i)
            if self.rows_done == i * self.rpc:
                self.crcs[key] = [zlib.crc32(np.ascontiguousarray(part[0]).tobytes()) & 0xFFFFFFFF, None]
            self.writer.write_rows(part)
            self.rows_done += part.shape[0]
            if self.rows_done == min((i + 1) * self.rpc, self.shape[0]):
                self.crcs[key][1] = zlib.crc32(np.ascontiguousarray(part[-1]).tobytes()) & 0xFFFFFFFF
                self.writer.finish()
                self._open_current(False)

    def checkpoint(self, **state):
        if self.writer is not None:
            self.writer.checkpoint(**state)
        self.manifest_path.with_suffix(".partial.json").write_text(json.dumps({"rows_done": self.rows_done, "row_crc32": self.crcs}))

    def finish(self):
        assert self.rows_done == self.shape[0]
        manifest = {"n_chunks": self.n_chunks, "shape": list(self.shape), "dtype": self.dtype.name,
                    "rows_per_chunk": self.rpc, "files": [self.chunk_path(i).name for i in range(self.n_chunks)],
                    "row_crc32": self.crcs}
        self.manifest_path.write_text(json.dumps(manifest, indent=2))
        p = self.manifest_path.with_suffix(".partial.json")
        if p.exists():
            p.unlink()


# ------------------------------------------------------------------------------------------
# Writers for one split
# ------------------------------------------------------------------------------------------
def write_root_array(name: str, split: str, split_dir: Path, n: int, profiles: Profiles, args, stats: dict) -> None:
    shape = spec.root_array_shape(name, n)
    dtype = spec.root_array_dtype(name)
    final = split_dir / f"{name}.npy"
    chunked = name == "X_seq" and args.stage_chunks
    row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * np.dtype(dtype).itemsize
    rows_per_block = block_rows_for(row_bytes, args.block_bytes)
    if chunked:
        if (split_dir / "X_seq_manifest.json").exists() and args.resume:
            log(f"  {name}: chunks complete, skipping")
            return
        progress = split_dir / "X_seq_chunks.progress"
        rows_done = json.loads(progress.read_text())["rows_done"] if (args.resume and progress.exists()) else 0
        sink = ChunkSink(split_dir, shape, dtype, args.rows_per_chunk, args.resume, rows_done)
        rows_done = sink.rows_done
    else:
        if final.exists() and args.resume:
            log(f"  {name}: complete, skipping")
            return
        sink = NpyWriter(final, shape, dtype, args.resume)
        rows_done = sink.rows_done
    if rows_done % rows_per_block:  # block size changed between runs: restart the file
        if chunked:
            sink.rows_done = 0
            sink = ChunkSink(split_dir, shape, dtype, args.rows_per_chunk, False, 0)
        else:
            sink.abort()
            sink = NpyWriter(final, shape, dtype, False)
        rows_done = 0
    t0 = time.perf_counter()
    n_blocks = math.ceil(n / rows_per_block)
    for b in range(rows_done // rows_per_block, n_blocks):
        if args.max_blocks is not None and b >= args.max_blocks:
            sink.checkpoint()
            if chunked:
                progress.write_text(json.dumps({"rows_done": sink.rows_done}))
            log(f"  {name}: stopped after {b} blocks (--max-blocks)")
            return
        s, e = b * rows_per_block, min((b + 1) * rows_per_block, n)
        block = gen_root_block(name, split, profiles.rows(s, e), block_rng(args.seed, split, name, b), args.pattern)
        sink.write_rows(block)
        if (b + 1) % 8 == 0 or b + 1 == n_blocks:
            sink.checkpoint()
            if chunked:
                progress.write_text(json.dumps({"rows_done": sink.rows_done}))
        if name == "X_seq" and (b + 1) % max(1, n_blocks // 20) == 0:
            done = min(e, n)
            log(f"    X_seq {done}/{n} rows ({100 * done / n:.0f}%)  {gib(done * row_bytes) / (time.perf_counter() - t0 + 1e-9):.2f} GiB/s  rss {gib(rss_bytes()):.2f} GiB")
    sink.finish()
    if chunked and progress.exists():
        progress.unlink()
    stats[name] = {"seconds": time.perf_counter() - t0, "bytes": n * row_bytes}


def write_code_family(fam: tuple, split: str, split_dir: Path, n: int, profiles: Profiles, args, stats: dict) -> None:
    name = fam[0]
    code_dir = split_dir / "code_ids"
    ids_path, off_path = code_dir / f"{name}_flat_ids.npy", code_dir / f"{name}_offsets.npy"
    if args.resume and ids_path.exists() and off_path.exists():
        log(f"  {name}: complete, skipping")
        return
    n_bags = n * spec.SEQ_LEN
    n_ids = spec.expected_n_ids(fam, split, n)
    vocab = spec.vocab_size(fam, split)
    extra_total = n_ids - n_bags
    assert n_ids < 2**31 - 1, f"{name}: {n_ids} ids exceed the int32 offset ceiling"
    rows_per_block = block_rows_for(spec.SEQ_LEN * 4 * 2, args.block_bytes)
    ids_w = NpyWriter(ids_path, (n_ids,), np.int32, args.resume)
    off_w = NpyWriter(off_path, (n_bags + 1,), np.int32, args.resume)
    start_block = int(off_w.state.get("block", 0))
    if start_block and (ids_w.rows_done != int(off_w.state.get("ids_written", -1)) or off_w.rows_done != 1 + start_block * rows_per_block * spec.SEQ_LEN):
        ids_w.abort(); off_w.abort()
        ids_w = NpyWriter(ids_path, (n_ids,), np.int32, False)
        off_w = NpyWriter(off_path, (n_bags + 1,), np.int32, False)
        start_block = 0
    running = ids_w.rows_done
    if start_block == 0:
        off_w.write_rows(np.zeros(1, dtype=np.int32))
    t0 = time.perf_counter()
    n_blocks = math.ceil(n / rows_per_block)
    for b in range(start_block, n_blocks):
        if args.max_blocks is not None and b >= args.max_blocks:
            ids_w.checkpoint(); off_w.checkpoint(block=b, ids_written=ids_w.rows_done)
            log(f"  {name}: stopped after {b} blocks (--max-blocks)")
            return
        s, e = b * rows_per_block, min((b + 1) * rows_per_block, n)
        bags = (e - s) * spec.SEQ_LEN
        quota = (extra_total * (e * spec.SEQ_LEN)) // n_bags - (extra_total * (s * spec.SEQ_LEN)) // n_bags
        enroll = profiles.rows(s, e)["enroll"].reshape(-1).astype(bool)
        rng = block_rng(args.seed, split, f"code_ids/{name}", b)
        sizes = np.ones(bags, dtype=np.int64)
        if quota > 0:
            active = np.flatnonzero(enroll)
            target = active if active.size else np.arange(bags)
            sizes += np.bincount(target[rng.integers(0, target.size, size=quota)], minlength=bags)
        total = int(sizes.sum())
        if args.pattern == "fast-io":
            ids = rng.integers(1, vocab, size=total, dtype=np.int32)
        else:
            ids = (1 + (rng.random(total, dtype=np.float32) ** 2) * (vocab - 1)).astype(np.int32)
        np.minimum(ids, vocab - 1, out=ids)
        local_start = np.cumsum(sizes) - sizes
        pad = (sizes == 1) & (~enroll | (rng.random(bags) < 0.35))
        ids[local_start[pad]] = 0
        off_w.write_rows((np.cumsum(sizes) + running).astype(np.int32))
        ids_w.write_rows(ids)
        running += total
        if (b + 1) % 8 == 0 or b + 1 == n_blocks:
            ids_w.checkpoint()
            off_w.checkpoint(block=b + 1, ids_written=ids_w.rows_done)
    assert running == n_ids, f"{name}: generated {running} ids, expected {n_ids}"
    ids_w.finish(); off_w.finish()
    stats[f"code_ids/{name}"] = {"seconds": time.perf_counter() - t0, "bytes": (n_ids + n_bags + 1) * 4}


# ------------------------------------------------------------------------------------------
# Planning, sidecar files, report
# ------------------------------------------------------------------------------------------
def plan_split(split: str, n: int) -> dict:
    files = []
    for name, trailing, dtype in spec.ROOT_ARRAYS:
        shape = (n, *trailing)
        files.append({"file": f"{name}.npy", "shape": list(shape), "dtype": dtype,
                      "bytes": int(np.prod(shape, dtype=np.int64)) * np.dtype(dtype).itemsize})
    for fam in spec.CODE_FAMILIES:
        n_ids = spec.expected_n_ids(fam, split, n)
        files.append({"file": f"code_ids/{fam[0]}_flat_ids.npy", "shape": [n_ids], "dtype": "int32", "bytes": n_ids * 4})
        files.append({"file": f"code_ids/{fam[0]}_offsets.npy", "shape": [n * spec.SEQ_LEN + 1], "dtype": "int32",
                      "bytes": (n * spec.SEQ_LEN + 1) * 4})
    return {"split": split, "n_members": n, "files": files, "n_npy_files": len(files),
            "npy_bytes": sum(f["bytes"] for f in files),
            "x_seq_bytes": n * spec.SEQ_LEN * 324 * 4}


def free_bytes(path: Path) -> int:
    p = path
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def file_sizes(path: Path) -> tuple[int, int]:
    st = path.stat()
    return st.st_size, getattr(st, "st_blocks", 0) * 512


def write_metadata(split: str, n: int, split_dir: Path, args) -> None:
    s = spec.SPLITS[split]
    meta = {
        "split": split, "synthetic": True,
        "generator": {"script": "generate_synthetic_cmd_arrays.py", "version": GENERATOR_VERSION, "seed": args.seed,
                      "scale": args.scale, "pattern": args.pattern, "n_members_full": s["n_members"],
                      "stage_chunks": bool(args.stage_chunks)},
        "n_members": n,
        "X_static_shape": list(spec.root_array_shape("X_static", n)),
        "X_seq_shape": list(spec.root_array_shape("X_seq", n)),
        "y_shape": list(spec.root_array_shape("y", n)),
        "sequence_length": spec.SEQ_LEN, "outcome_length": spec.OUT_LEN,
        "static_columns": spec.STATIC_COLUMNS, "seq_columns": spec.SEQ_COLUMNS,
        "code_families": {f[0]: {"n_ids": spec.expected_n_ids(f, split, n), "n_member_months": n * spec.SEQ_LEN,
                                 "vocab_size": spec.vocab_size(f, split)} for f in spec.CODE_FAMILIES},
        "plan_cat_columns": spec.PLAN_CAT_COLUMNS, "plan_cat_vocab_sizes": s["plan_cat_vocab_sizes"],
        "outcome_months": s["outcome_months"],
        "event_y_shape": list(spec.root_array_shape("event_y", n)),
        "spend_archetype_multilabel_shape": list(spec.root_array_shape("spend_archetype_multilabel", n)),
    }
    for arr, (key, cols) in spec.FUTURE_COLUMN_LISTS.items():
        meta[f"{arr}_shape"] = list(spec.root_array_shape(arr, n))
        meta[key] = cols
        meta[f"{arr}_mask"] = spec.MASK_FOR[arr]
    (split_dir / "metadata.json").write_text(json.dumps(meta, indent=1))


def write_index_files(split: str, n: int, split_dir: Path, profiles: Profiles, args) -> dict:
    partial = np.flatnonzero(profiles.scan("partial")).tolist()
    is_new = profiles.scan("is_new")
    new_idx, non_new_idx = np.flatnonzero(is_new).tolist(), np.flatnonzero(~is_new).tolist()
    (split_dir / "partial_exposure_indices.json").write_text(json.dumps(partial))
    (split_dir / "member_strat_indices.json").write_text(json.dumps({"New": new_idx, "Non-New": non_new_idx}))
    extras = {}
    if split == "train__CMD" and args.train_val_split:
        rng = np.random.default_rng([args.seed, spec.SPLITS[split]["index"], TAG_SPLIT])
        perm = rng.permutation(n)
        n_val = int(round(0.2 * n))
        (split_dir / "train_val_split.json").write_text(json.dumps({
            "seed": args.seed, "fraction": 0.2, "n_members": n,
            "val_indices": sorted(perm[:n_val].tolist()), "train_indices": sorted(perm[n_val:].tolist())}))
        extras["train_val_split.json"] = "optional extra: not part of the 121-file source layout"
    return {"partial_exposure": len(partial), "New": len(new_idx), "Non-New": len(non_new_idx), "extras": extras}


def write_labels_parquet(split: str, n: int, split_dir: Path) -> bool:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        log("  member_labels.parquet skipped: pyarrow is not installed (uv sync --extra cmd)")
        return False
    schema = pa.schema([("PERSON_ID", pa.large_string()), ("PRODUCT_LINE_OF_BUSINESS", pa.large_string())])
    prefix = spec.SPLITS[split]["id_prefix"]
    tmp = split_dir / "member_labels.parquet.partial"
    with pq.ParquetWriter(tmp, schema) as w:
        for s in range(0, n, 131_072):
            e = min(s + 131_072, n)
            ids = pa.array([f"{prefix}{i + 1:09d}" for i in range(s, e)], type=pa.large_string())
            lob = pa.array(["CMD"] * (e - s), type=pa.large_string())
            w.write_table(pa.table({"PERSON_ID": ids, "PRODUCT_LINE_OF_BUSINESS": lob}, schema=schema))
    os.replace(tmp, split_dir / "member_labels.parquet")
    return True


def size_report(split_dir: Path) -> dict:
    rows, logical, allocated = [], 0, 0
    for p in sorted(list(split_dir.glob("*")) + list((split_dir / "code_ids").glob("*"))):
        if p.is_dir():
            continue
        lo, al = file_sizes(p)
        logical += lo
        allocated += al
        rows.append({"file": str(p.relative_to(split_dir)), "logical_bytes": lo, "allocated_bytes": al})
    npy = [r for r in rows if r["file"].endswith(".npy")]
    flag = [r["file"] for r in npy if r["allocated_bytes"] and r["allocated_bytes"] < 0.9 * r["logical_bytes"]]
    return {"files": rows, "n_files": len(rows), "n_npy_files": len(npy), "logical_bytes": logical,
            "allocated_bytes": allocated, "sparse_flagged": flag}


def cmd_generate(args) -> int:
    if args.full:
        args.scale = 1.0
    if args.scale >= 1.0 and not args.full:
        log("error: --scale 1.0 writes the exact full populations; pass --full to confirm")
        return 2
    if not 0 < args.scale <= 1.0:
        log("error: --scale must be in (0, 1]")
        return 2
    args.rows_per_chunk = args.rows_per_chunk or max(1, int(round(FULL_ROWS_PER_CHUNK * args.scale)))
    out = Path(args.output_root)
    splits = list(spec.SPLITS) if args.splits == "both" else [s for s in spec.SPLITS if s.startswith(args.splits)]
    plans = {s: plan_split(s, spec.n_members_at(s, args.scale)) for s in splits}
    total = sum(p["npy_bytes"] for p in plans.values())
    duplicate = 0
    for s, p in plans.items():
        if args.stage_chunks and (out / "arrays" / s / "X_seq.npy").exists():
            duplicate += p["x_seq_bytes"]
    already = 0
    if args.resume:
        for s in splits:
            d = out / "arrays" / s
            if d.exists():
                already += sum(file_sizes(f)[0] for f in list(d.glob("*.npy")) + list((d / "code_ids").glob("*.npy")))
    need = int((total + duplicate - already) * 1.15)
    free = free_bytes(out)
    log(f"== synthetic CMD generator v{GENERATOR_VERSION}: scale={args.scale} seed={args.seed} pattern={args.pattern} "
        f"chunks={'on (rows_per_chunk=%d)' % args.rows_per_chunk if args.stage_chunks else 'off'} ==")
    for s, p in plans.items():
        log(f"  {s}: {p['n_members']:,} members, {p['n_npy_files']} .npy files, {gib(p['npy_bytes']):.3f} GiB "
            f"(X_seq {gib(p['x_seq_bytes']):.3f} GiB)")
    log(f"  output {out}: need {gib(need):.2f} GiB incl. 15% headroom"
        f"{' and %.1f GiB duplicate X_seq' % gib(duplicate) if duplicate else ''}, free {gib(free):.2f} GiB")
    if args.dry_run:
        for s, p in plans.items():
            log(f"\n  {s} per-file plan:")
            for f in p["files"]:
                if not f["file"].startswith("code_ids/") or f["file"].endswith("_flat_ids.npy") and p["n_members"] < 1000:
                    log(f"    {f['file']:48s} {str(f['shape']):24s} {f['dtype']:8s} {gib(f['bytes']):9.3f} GiB")
            log(f"    code_ids/: 100 files, {gib(sum(f['bytes'] for f in p['files'] if f['file'].startswith('code_ids/'))):.3f} GiB")
        log("\ndry run: nothing written")
        return 0
    if free < need:
        log(f"error: not enough free disk ({gib(free):.2f} GiB < {gib(need):.2f} GiB)")
        return 3
    if not args.force and not args.resume:
        for s in splits:
            if (out / "arrays" / s).exists() and any((out / "arrays" / s).iterdir()):
                log(f"error: {out / 'arrays' / s} exists; use --resume to continue or --force to overwrite")
                return 2
    t_start = time.perf_counter()
    report = {"generator_version": GENERATOR_VERSION, "seed": args.seed, "scale": args.scale, "pattern": args.pattern,
              "stage_chunks": bool(args.stage_chunks), "output_root": str(out), "splits": {}}
    for s in splits:
        n = plans[s]["n_members"]
        split_dir = out / "arrays" / s
        if args.force and split_dir.exists() and not args.resume:
            shutil.rmtree(split_dir)
        (split_dir / "code_ids").mkdir(parents=True, exist_ok=True)
        profiles = Profiles(s, n, args.seed)
        stats: dict = {}
        log(f"\n[{s}] {n:,} members")
        t0 = time.perf_counter()
        for name, _, _ in spec.ROOT_ARRAYS:
            write_root_array(name, s, split_dir, n, profiles, args, stats)
        for fam in spec.CODE_FAMILIES:
            write_code_family(fam, s, split_dir, n, profiles, args, stats)
        if args.max_blocks is not None:
            log(f"[{s}] stopped early by --max-blocks; rerun with --resume to finish")
            continue
        write_metadata(s, n, split_dir, args)
        idx = write_index_files(s, n, split_dir, profiles, args)
        parquet = write_labels_parquet(s, n, split_dir)
        sizes = size_report(split_dir)
        elapsed = time.perf_counter() - t0
        report["splits"][s] = {"n_members": n, "elapsed_seconds": elapsed, "planned_npy_bytes": plans[s]["npy_bytes"],
                               "index_counts": idx, "member_labels_parquet": parquet, **sizes,
                               "per_file_seconds": stats}
        log(f"[{s}] done in {elapsed:.0f}s: {sizes['n_files']} files, logical {gib(sizes['logical_bytes']):.3f} GiB, "
            f"allocated {gib(sizes['allocated_bytes']):.3f} GiB, partial-exposure {idx['partial_exposure']:,}, "
            f"New/Non-New {idx['New']:,}/{idx['Non-New']:,}"
            + (f"; SPARSE FILES FLAGGED: {sizes['sparse_flagged']}" if sizes["sparse_flagged"] else ""))
    report["elapsed_seconds"] = time.perf_counter() - t_start
    report["peak_rss_bytes"] = rss_bytes()
    out.mkdir(parents=True, exist_ok=True)
    (out / "synthetic_cmd_report.json").write_text(json.dumps(report, indent=2))
    written = sum(v.get("logical_bytes", 0) for v in report["splits"].values())
    rate = written / max(report["elapsed_seconds"], 1e-9)
    full_bytes = sum(plan_split(s, spec.SPLITS[s]["n_members"])["npy_bytes"] for s in spec.SPLITS)
    log(f"\ntotal {report['elapsed_seconds']:.0f}s, {gib(written):.3f} GiB written ({rate / 2**20:.0f} MiB/s), "
        f"peak RSS {gib(report['peak_rss_bytes']):.2f} GiB; report: {out / 'synthetic_cmd_report.json'}")
    if args.scale < 1.0 and rate > 0:
        log(f"at this rate the full {gib(full_bytes):.0f} GiB run would take about {full_bytes / rate / 3600:.1f} h")
        log(f"full-run command: python {Path(__file__).name} --output-root {out} --full --pattern {args.pattern} "
            f"--seed {args.seed}{' --stage-chunks' if args.stage_chunks else ''} --resume")
    return 0


# ------------------------------------------------------------------------------------------
# validate
# ------------------------------------------------------------------------------------------
class Checker:
    def __init__(self):
        self.failures: list[str] = []
        self.passed = 0

    def check(self, ok: bool, what: str):
        if ok:
            self.passed += 1
        else:
            self.failures.append(what)
            log(f"    FAIL: {what}")


def _slabs(n: int, rows: int):
    for s in range(0, n, rows):
        yield s, min(s + rows, n)


def validate_split(split: str, split_dir: Path, args, ck: Checker) -> dict:
    meta_path = split_dir / "metadata.json"
    ck.check(meta_path.exists(), f"{split}: metadata.json missing")
    if not meta_path.exists():
        return {}
    meta = json.loads(meta_path.read_text())
    n = int(meta["n_members"])
    scale = meta.get("generator", {}).get("scale", 1.0)
    log(f"  [{split}] n_members={n:,} scale={scale}")
    # ---- files, shapes, dtypes, bytes ----
    plan = plan_split(split, n)
    present = {}
    for f in plan["files"]:
        p = split_dir / f["file"]
        if not p.exists():
            if f["file"] == "X_seq.npy" and (split_dir / "X_seq_manifest.json").exists():
                ck.check(True, "")
                continue
            ck.check(False, f"{split}: missing {f['file']}")
            continue
        a = np.load(p, mmap_mode="r")
        ck.check(list(a.shape) == f["shape"] and a.dtype.name == f["dtype"],
                 f"{split}: {f['file']} is {a.shape} {a.dtype}, expected {f['shape']} {f['dtype']}")
        lo, al = file_sizes(p)
        ck.check(0 <= lo - f["bytes"] <= 4096, f"{split}: {f['file']} logical bytes {lo} vs expected payload {f['bytes']}")
        present[f["file"]] = a
    for extra in ("partial_exposure_indices.json", "member_strat_indices.json", "member_labels.parquet"):
        ck.check((split_dir / extra).exists(), f"{split}: missing {extra}")
    n_files = sum(1 for p in list(split_dir.glob("*")) + list((split_dir / "code_ids").glob("*")) if p.is_file()
                  and not p.name.endswith((".partial", ".progress", ".partial.json")))
    expected_files = 121 + (1 if (split_dir / "train_val_split.json").exists() else 0)
    if (split_dir / "X_seq_manifest.json").exists():
        expected_files += int(json.load(open(split_dir / "X_seq_manifest.json"))["n_chunks"])  # chunks + manifest - X_seq
    ck.check(n_files == expected_files, f"{split}: {n_files} files, expected {expected_files}")
    if n == spec.SPLITS[split]["n_members"]:
        tot = sum(file_sizes(split_dir / f["file"])[0] for f in plan["files"] if (split_dir / f["file"]).exists())
        ck.check(abs(gib(tot) - spec.TOTAL_GIB[split]) < 0.05, f"{split}: full-size logical total {gib(tot):.2f} GiB vs {spec.TOTAL_GIB[split]}")
    # ---- metadata consistency ----
    ck.check(meta["static_columns"] == spec.STATIC_COLUMNS and meta["seq_columns"] == spec.SEQ_COLUMNS, f"{split}: column lists differ from spec")
    ck.check(meta["X_seq_shape"] == [n, spec.SEQ_LEN, 324] and meta["X_static_shape"] == [n, 255] and meta["y_shape"] == [n, 12], f"{split}: metadata shapes")
    ck.check(meta["sequence_length"] == spec.SEQ_LEN and meta["outcome_length"] == spec.OUT_LEN, f"{split}: lengths")
    ck.check(meta["plan_cat_vocab_sizes"] == spec.SPLITS[split]["plan_cat_vocab_sizes"], f"{split}: plan_cat_vocab_sizes")
    ck.check(len(meta["outcome_months"]) == 12, f"{split}: outcome_months")
    for arr, (key, cols) in spec.FUTURE_COLUMN_LISTS.items():
        ck.check(meta.get(key) == cols and meta.get(f"{arr}_shape") == list(spec.root_array_shape(arr, n)), f"{split}: metadata {key}")
    # ---- masks, targets ----
    slab = max(256, args.slab_bytes // (spec.SEQ_LEN * 324 * 4))
    om = present.get("outcome_enrollment_mask.npy")
    em = present.get("enrollment_mask.npy")
    if om is not None and em is not None:
        om_all = np.asarray(om)
        ck.check(np.isin(om_all, [0.0, 1.0]).all(), f"{split}: outcome_enrollment_mask not binary")
        ck.check(np.isin(np.asarray(em), [0.0, 1.0]).all() and np.asarray(em).sum(axis=1).min() >= 1, f"{split}: enrollment_mask not binary/plausible")
        partial_rows = np.flatnonzero(om_all.sum(axis=1) < 12)
        pe = json.loads((split_dir / "partial_exposure_indices.json").read_text()) if (split_dir / "partial_exposure_indices.json").exists() else []
        pe = pe["indices"] if isinstance(pe, dict) else pe
        ck.check(np.array_equal(np.asarray(pe, dtype=np.int64), partial_rows), f"{split}: partial_exposure_indices.json != rows with <12 enrolled months")
        target = spec.scaled_count(spec.SPLITS[split]["partial_exposure_members"], split, n)
        ck.check(abs(len(partial_rows) - target) <= max(2, 0.01 * target), f"{split}: {len(partial_rows)} partial-exposure members vs target {target}")
        for tname, ch in (("y.npy", None), ("event_y.npy", 5)):
            t = present.get(tname)
            if t is None:
                continue
            for s, e in _slabs(n, 65536):
                blk = np.asarray(t[s:e]); m = om_all[s:e].astype(bool)
                m = m[:, :, None] if ch else m
                nan_ok = np.isnan(blk[~np.broadcast_to(m, blk.shape)]).all()
                val = blk[np.broadcast_to(m, blk.shape)]
                ok = nan_ok and np.isfinite(val).all() and (val >= 0).all()
                if not ok:
                    ck.check(False, f"{split}: {tname} NaN/negativity pattern wrong in rows {s}-{e}")
                    break
            else:
                ck.check(True, "")
            if tname == "event_y.npy" and t is not None:
                v = np.asarray(t[: min(n, 65536)])
                ck.check(np.isin(v[np.isfinite(v)], [0.0, 1.0]).all() and np.nansum(v) > 0, f"{split}: event_y not binary / all zero")
        sam = present.get("spend_archetype_multilabel.npy")
        if sam is not None:
            v = np.asarray(sam)
            ck.check(np.isin(v, [0.0, 1.0]).all() and v.sum() > 0, f"{split}: spend_archetype_multilabel")
        sa = present.get("spend_archetype.npy")
        if sa is not None:
            ck.check(0 <= int(np.asarray(sa).min()) and int(np.asarray(sa).max()) <= 5, f"{split}: spend_archetype range")
    for arr, mask_name in spec.MASK_FOR.items():
        a, mk = present.get(f"{arr}.npy"), present.get(f"{mask_name}.npy")
        if a is None or mk is None:
            continue
        mv = np.asarray(mk)
        ck.check(np.isin(mv, [0.0, 1.0]).all(), f"{split}: {mask_name} not binary")
        av = np.asarray(a)
        mb = mv if mv.shape == av.shape else mv[:, None]
        ck.check(np.isfinite(av).all() and (av[np.broadcast_to(mb, av.shape) == 0] == 0).all(), f"{split}: {arr} not aligned with {mask_name}")
    pc = present.get("X_plan_cat.npy")
    if pc is not None:
        vocab = np.array(list(spec.SPLITS[split]["plan_cat_vocab_sizes"].values()))
        v = np.asarray(pc)
        ck.check((v >= 0).all() and (v < vocab[None, :]).all(), f"{split}: X_plan_cat ids out of vocab range")
    xs = present.get("X_static.npy")
    if xs is not None:
        ck.check(np.isfinite(np.asarray(xs)).all(), f"{split}: X_static has non-finite values")
    # ---- X_seq (assembled or chunked), slab by slab ----
    xq = present.get("X_seq.npy")
    if xq is None and (split_dir / "X_seq_manifest.json").exists():
        man = json.load(open(split_dir / "X_seq_manifest.json"))
        ck.check(man["shape"] == [n, spec.SEQ_LEN, 324] and len(man["files"]) == man["n_chunks"], f"{split}: X_seq manifest")
        chunks = [np.load(split_dir / f, mmap_mode="r") for f in man["files"]]
        ck.check(sum(c.shape[0] for c in chunks) == n and all(c.shape[1:] == (spec.SEQ_LEN, 324) for c in chunks), f"{split}: X_seq chunk shapes")
        xq = None
        if em is not None:
            emv = np.asarray(em).astype(bool)
            row = 0
            ok = True
            for c in chunks:
                for s, e in _slabs(c.shape[0], slab):
                    blk = np.asarray(c[s:e])
                    ok &= bool(np.isfinite(blk).all() and (blk[~emv[row + s: row + e]] == 0).all())
                row += c.shape[0]
            ck.check(ok, f"{split}: X_seq chunk values non-finite or non-zero outside enrollment")
    elif xq is not None and em is not None:
        emv = np.asarray(em).astype(bool)
        ok, nonzero = True, 0
        for s, e in _slabs(n, slab):
            blk = np.asarray(xq[s:e])
            ok &= bool(np.isfinite(blk).all() and (blk[~emv[s:e]] == 0).all())
            nonzero += int(np.count_nonzero(blk[emv[s:e]]))
        ck.check(ok and nonzero > 0, f"{split}: X_seq values non-finite, non-zero outside enrollment, or all zero")
    # ---- code families ----
    for fam in spec.CODE_FAMILIES:
        ids, off = present.get(f"code_ids/{fam[0]}_flat_ids.npy"), present.get(f"code_ids/{fam[0]}_offsets.npy")
        if ids is None or off is None:
            continue
        o = np.asarray(off)
        vocab = spec.vocab_size(fam, split)
        ok = o[0] == 0 and o[-1] == ids.shape[0] and (np.diff(o) >= 1).all() and o.shape[0] == n * spec.SEQ_LEN + 1
        ok &= ids.shape[0] < 2**31 - 1
        mn, mx = int(np.asarray(ids).min()), int(np.asarray(ids).max())
        ok &= 0 <= mn and mx < vocab
        ck.check(bool(ok), f"{split}: code family {fam[0]} offsets/ids invalid (offsets {o[0]}..{o[-1]}, ids {mn}..{mx}, vocab {vocab})")
        ck.check(meta["code_families"][fam[0]]["n_ids"] == ids.shape[0], f"{split}: metadata n_ids for {fam[0]}")
    # ---- strat lists ----
    sp = split_dir / "member_strat_indices.json"
    if sp.exists():
        st = json.loads(sp.read_text())
        new, non = np.asarray(st.get("New", []), dtype=np.int64), np.asarray(st.get("Non-New", []), dtype=np.int64)
        allrows = np.sort(np.concatenate([new, non]))
        ck.check(np.array_equal(allrows, np.arange(n)), f"{split}: New/Non-New lists do not partition all rows")
        target = spec.scaled_count(spec.SPLITS[split]["strat_new"], split, n)
        ck.check(abs(len(new) - target) <= max(2, 0.01 * target), f"{split}: {len(new)} New members vs target {target}")
    # ---- parquet ----
    pq_path = split_dir / "member_labels.parquet"
    if pq_path.exists():
        try:
            import pyarrow.parquet as pq
            t = pq.read_table(pq_path)
            ck.check(t.num_rows == n and t.schema.names == ["PERSON_ID", "PRODUCT_LINE_OF_BUSINESS"]
                     and str(t.schema.field("PERSON_ID").type) == "large_string"
                     and set(t.column("PRODUCT_LINE_OF_BUSINESS").unique().to_pylist()) == {"CMD"}
                     and t.column("PERSON_ID")[0].as_py().startswith(spec.SPLITS[split]["id_prefix"]),
                     f"{split}: member_labels.parquet schema/content")
        except ImportError:
            log("    (pyarrow missing: parquet content not checked)")
    sizes = size_report(split_dir)
    ck.check(not sizes["sparse_flagged"], f"{split}: allocated bytes far below logical for {sizes['sparse_flagged']}")
    return {"n_members": n, "n_files": sizes["n_files"], "logical_gib": gib(sizes["logical_bytes"]),
            "allocated_gib": gib(sizes["allocated_bytes"])}


def try_member_dataset(args, out: Path, ck: Checker) -> None:
    if not args.member_dataset:
        log("  MemberDataset check skipped (pass --member-dataset module:Class from the member-level repo)")
        return
    prev = os.environ.get("DATA_ROOT")
    if prev and Path(prev).resolve() != out.resolve():
        log(f"  note: DATA_ROOT was {prev}; overriding with {out} for this check")
    os.environ["DATA_ROOT"] = str(out)
    mod, _, cls = args.member_dataset.partition(":")
    try:
        import importlib
        MemberDataset = getattr(importlib.import_module(mod), cls)
    except Exception as e:  # noqa: BLE001
        log(f"  MemberDataset import failed ({e}); skipped")
        return
    for split in spec.SPLITS:
        try:
            ds = MemberDataset(split)
            item = ds[0]
            dims = {k: tuple(getattr(v, "shape", ())) for k, v in (item.items() if isinstance(item, dict) else enumerate(item))}
            log(f"  {split}: MemberDataset len={len(ds)} item dims={dims}")
            if args.batch_size:
                import torch
                dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers)
                t0 = time.perf_counter()
                batch = next(iter(dl))
                dev = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
                if isinstance(batch, dict):
                    batch = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in batch.items()}
                log(f"    one batch of {args.batch_size} in {time.perf_counter() - t0:.2f}s on {dev} (workers={args.workers})")
            ck.check(True, "")
        except Exception as e:  # noqa: BLE001
            ck.check(False, f"{split}: MemberDataset failed: {e}")


def cmd_validate(args) -> int:
    out = Path(args.output_root)
    ck = Checker()
    summary = {}
    t0 = time.perf_counter()
    for split in spec.SPLITS:
        d = out / "arrays" / split
        if not d.exists():
            if args.splits == "both" or split.startswith(args.splits):
                ck.check(False, f"{split}: directory {d} missing")
            continue
        summary[split] = validate_split(split, d, args, ck)
    try_member_dataset(args, out, ck)
    log(f"\nvalidation: {ck.passed} checks passed, {len(ck.failures)} failed in {time.perf_counter() - t0:.0f}s, "
        f"peak RSS {gib(rss_bytes()):.2f} GiB")
    for s, v in summary.items():
        if v:
            log(f"  {s}: {v['n_members']:,} members, {v['n_files']} files, logical {v['logical_gib']:.3f} GiB, allocated {v['allocated_gib']:.3f} GiB")
    (out / "synthetic_cmd_validation.json").write_text(json.dumps({"passed": ck.passed, "failures": ck.failures, "summary": summary}, indent=2))
    return 1 if ck.failures else 0


# ------------------------------------------------------------------------------------------
# reassemble
# ------------------------------------------------------------------------------------------
def cmd_reassemble(args) -> int:
    split_dir = Path(args.output_root) / "arrays" / args.split
    man_path = split_dir / "X_seq_manifest.json"
    if not man_path.exists():
        log(f"error: {man_path} not found")
        return 2
    man = json.loads(man_path.read_text())
    shape, dtype = tuple(man["shape"]), np.dtype(man["dtype"])
    target = split_dir / "X_seq.npy"
    if target.exists() and not args.force:
        log(f"error: {target} exists (use --force to rebuild)")
        return 2
    need = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if free_bytes(split_dir) < need * 1.15:
        log(f"error: need {gib(need * 1.15):.1f} GiB free to reassemble (chunks and X_seq.npy coexist until validated)")
        return 3
    w = NpyWriter(target, shape, dtype, False)
    row_bytes = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
    slab = max(64, args.slab_bytes // row_bytes)
    row, t0, ok = 0, time.perf_counter(), True
    for i, fname in enumerate(man["files"]):
        c = np.load(split_dir / fname, mmap_mode="r")
        first, last = man["row_crc32"][str(i)]
        ok &= zlib.crc32(np.ascontiguousarray(c[0]).tobytes()) & 0xFFFFFFFF == first
        ok &= zlib.crc32(np.ascontiguousarray(c[-1]).tobytes()) & 0xFFFFFFFF == last
        for s, e in _slabs(c.shape[0], slab):
            w.write_rows(np.asarray(c[s:e]))
        row += c.shape[0]
        log(f"  chunk {i + 1}/{man['n_chunks']} ({fname}) -> rows {row - c.shape[0]}..{row}")
    if row != shape[0] or not ok:
        w.abort()
        log("error: row order / checksum validation failed; X_seq.npy not written")
        return 1
    w.finish()
    a = np.load(target, mmap_mode="r")
    for i, fname in enumerate(man["files"]):
        s = i * man["rows_per_chunk"]
        c = np.load(split_dir / fname, mmap_mode="r")
        ok &= np.array_equal(a[s], c[0]) and np.array_equal(a[s + c.shape[0] - 1], c[-1])
    log(f"reassembled {gib(need):.2f} GiB in {time.perf_counter() - t0:.0f}s; row-order check {'OK' if ok else 'FAILED'}")
    if not ok:
        return 1
    if not args.keep_chunks:
        for fname in man["files"]:
            (split_dir / fname).unlink()
        man_path.unlink()
        log("chunks removed (--keep-chunks keeps both layouts)")
    return 0


# ------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command")
    g = sub.add_parser("generate", help="write the arrays (default command)")
    g.add_argument("--output-root", required=True)
    g.add_argument("--scale", type=float, default=0.001, help="fraction of the full populations (default 0.001)")
    g.add_argument("--seed", type=int, default=20240101)
    g.add_argument("--dry-run", action="store_true", help="print the plan and size estimate only")
    g.add_argument("--resume", action="store_true", help="skip complete files, continue partial ones")
    g.add_argument("--full", action="store_true", help="write the exact full-scale populations (scale 1.0)")
    g.add_argument("--pattern", choices=["fast-io", "structured"], default="structured")
    g.add_argument("--stage-chunks", action="store_true", help="write X_seq as ~5 GiB member-major chunks + manifest")
    g.add_argument("--rows-per-chunk", type=int, default=None, help="default 39452 x scale")
    g.add_argument("--splits", choices=["both", "train", "one_year"], default="both")
    g.add_argument("--block-bytes", type=int, default=128 * 2**20, help="bytes per member block (peak RSS ~3-4x this)")
    g.add_argument("--train-val-split", action="store_true", help="also write the optional train_val_split.json")
    g.add_argument("--force", action="store_true", help="overwrite an existing split directory")
    g.add_argument("--max-blocks", type=int, default=None, help=argparse.SUPPRESS)  # testing: simulate interruption
    v = sub.add_parser("validate", help="check an output root against the contract")
    v.add_argument("--output-root", required=True)
    v.add_argument("--splits", choices=["both", "train", "one_year"], default="both")
    v.add_argument("--slab-bytes", type=int, default=256 * 2**20)
    v.add_argument("--member-dataset", default=None, help="module:Class of the project's MemberDataset")
    v.add_argument("--batch-size", type=int, default=None, help="also pull one DataLoader batch of this size")
    v.add_argument("--workers", type=int, default=0)
    v.add_argument("--device", default=None, help="mps|cpu (default: mps if available)")
    r = sub.add_parser("reassemble", help="stream X_seq chunks into X_seq.npy and validate row order")
    r.add_argument("--output-root", required=True)
    r.add_argument("--split", choices=list(spec.SPLITS), required=True)
    r.add_argument("--keep-chunks", action="store_true")
    r.add_argument("--force", action="store_true")
    r.add_argument("--slab-bytes", type=int, default=256 * 2**20)
    sub.add_parser("spec", help="print the contract as JSON")
    return p


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or (argv[0].startswith("-") and argv[0] not in ("-h", "--help")):
        argv = ["generate", *argv]
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        return cmd_generate(args)
    if args.command == "validate":
        return cmd_validate(args)
    if args.command == "reassemble":
        return cmd_reassemble(args)
    if args.command == "spec":
        print(json.dumps({"splits": spec.SPLITS, "root_arrays": [(n, list(s), d) for n, s, d in spec.ROOT_ARRAYS],
                          "code_families": spec.CODE_FAMILIES, **spec.metadata_lists()}, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
