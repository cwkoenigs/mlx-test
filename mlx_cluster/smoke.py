"""Distributed smoke test + link micro-benchmark (Phase 2/3 gate).

    mlx.launch --hostfile hosts.json -- python -m mlx_cluster.smoke
    mlx.launch -n 2 --backend ring -- python -m mlx_cluster.smoke     # 2 processes on one machine

Checks: ranks see each other, mlx versions match, all_sum is correct, then measures collective
latency, all_sum / all_gather bandwidth and send/recv ping-pong (2 ranks). Rank 0 writes
results/cluster/smoke-<timestamp>.json. Exit 1 on any failed check.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mlx.core as mx

from mlx_cluster import nodeinfo
from mlx_cluster.comm import backend_name, barrier, gather_json, human_bytes


def timeit(fn, reps: int, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes-mib", default="1,16,128", help="all_sum payload sizes in MiB")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out-dir", default="results/cluster")
    args = ap.parse_args()

    group = mx.distributed.init()
    rank, size = group.rank(), group.size()
    backend = backend_name(group)
    main_rank = rank == 0
    failures = []

    def say(*a):
        if main_rank:
            print(*a, flush=True)

    infos = gather_json(nodeinfo.collect(), group)
    say(f"== MLX distributed smoke test: {size} rank(s), backend={backend} ==")
    for i in infos:
        say(f"  rank {i.get('hostname')}: {i.get('chip')} | RAM {i['ram_bytes'] / 2**30:.0f} GiB | "
            f"mlx {i['mlx_version']} | python {i['python']} | {i.get('macos_version') or i.get('os_version')}")
    if size < 2:
        say("WARNING: only one rank. Launch with mlx.launch (-n 2 or --hostfile) to test the cluster.")
    if len({i["mlx_version"] for i in infos}) != 1:
        failures.append("mlx versions differ across ranks: " + str([i["mlx_version"] for i in infos]))
    if len({i["python"] for i in infos}) != 1:
        failures.append("python versions differ across ranks: " + str([i["python"] for i in infos]))

    # correctness
    x = mx.ones((1024,)) * (rank + 1)
    y = mx.distributed.all_sum(x, group=group)
    mx.eval(y)
    expected = size * (size + 1) / 2
    ok = abs(float(y[0].item()) - expected) < 1e-5 and float(y.sum().item()) == expected * 1024
    say(f"all_sum correctness: {'OK' if ok else 'FAILED'} (got {float(y[0].item())}, expected {expected})")
    if not ok:
        failures.append("all_sum returned the wrong value")

    results = {"ranks": size, "backend": backend, "nodes": infos, "all_sum_ok": ok}
    if size > 1:
        # latency of a tiny collective
        s = mx.array(1.0)
        lat_gpu = timeit(lambda: mx.eval(mx.distributed.all_sum(s, group=group)), 50)
        lat_cpu = timeit(lambda: mx.eval(mx.distributed.all_sum(s, group=group, stream=mx.cpu)), 50)
        results["all_sum_latency_us"] = {"default_stream": lat_gpu * 1e6, "cpu_stream": lat_cpu * 1e6}
        say(f"all_sum latency (1 float): {lat_gpu * 1e6:.0f} us default stream, {lat_cpu * 1e6:.0f} us cpu stream")

        # bandwidth
        bw = []
        for mib in [float(v) for v in args.sizes_mib.split(",") if v]:
            n = int(mib * 2**20 / 4)
            a = mx.ones((n,), dtype=mx.float32)
            mx.eval(a)
            t_sum = timeit(lambda: mx.eval(mx.distributed.all_sum(a, group=group)), args.reps)
            t_gather = timeit(lambda: mx.eval(mx.distributed.all_gather(a, group=group)), max(1, args.reps // 2))
            row = {"mib": mib, "all_sum_s": t_sum, "all_sum_gbps": a.nbytes / t_sum / 1e9,
                   "all_gather_s": t_gather, "all_gather_gbps": a.nbytes * (size - 1) / t_gather / 1e9}
            bw.append(row)
            say(f"{mib:7.1f} MiB  all_sum {t_sum * 1e3:8.2f} ms ({row['all_sum_gbps']:.2f} GB/s)   "
                f"all_gather {t_gather * 1e3:8.2f} ms ({row['all_gather_gbps']:.2f} GB/s)")
        results["bandwidth"] = bw

        # send/recv ping-pong between rank 0 and 1 (ring backend only supports neighbours)
        if size == 2:
            n = 2**18  # 1 MiB float32
            payload = mx.ones((n,), dtype=mx.float32)
            mx.eval(payload)
            barrier(group)
            t0 = time.perf_counter()
            iters = 20
            for _ in range(iters):
                if rank == 0:
                    mx.eval(mx.distributed.send(payload, 1, group=group))
                    r = mx.distributed.recv_like(payload, 1, group=group)
                    mx.eval(r)
                else:
                    r = mx.distributed.recv_like(payload, 0, group=group)
                    mx.eval(r)
                    mx.eval(mx.distributed.send(r, 0, group=group))
            rt = (time.perf_counter() - t0) / iters
            results["pingpong_1mib_roundtrip_ms"] = rt * 1e3
            say(f"send/recv ping-pong (1 MiB): {rt * 1e3:.2f} ms round trip -> {2 * payload.nbytes / rt / 1e9:.2f} GB/s")
    barrier(group)

    results["failures"] = failures
    if main_rank:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"smoke-{time.strftime('%Y%m%d-%H%M%S')}.json"
        path.write_text(json.dumps(results, indent=2, default=str) + "\n")
        print(f"wrote {path}")
        print("SMOKE:", "FAILED - " + "; ".join(failures) if failures else "OK")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
