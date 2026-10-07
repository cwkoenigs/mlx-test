"""Collect hardware / software facts about this node (Phase 0 deliverable).

    python -m mlx_cluster.nodeinfo            # pretty JSON
    python -m mlx_cluster.nodeinfo --check    # sanity check (import, matmul, versions); exit 1 on failure
    python -m mlx_cluster.nodeinfo --write    # results/hardware/<hostname>.{json,md}
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import socket
import subprocess
import sys
import time
from pathlib import Path


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:
        return None


def _sysctl(key: str) -> str | None:
    return _run(["sysctl", "-n", key])


def physical_ram_bytes() -> int:
    if sys.platform == "darwin":
        v = _sysctl("hw.memsize")
        if v:
            return int(v)
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except FileNotFoundError:
        pass
    return 0


def rss_bytes() -> int:
    """Max resident set size of this process in bytes (ru_maxrss is bytes on macOS, KiB on Linux)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r) if sys.platform == "darwin" else int(r) * 1024


def gpu_cores_macos() -> int | None:
    out = _run(["system_profiler", "SPDisplaysDataType"])
    if not out:
        return None
    for line in out.splitlines():
        if "Total Number of Cores" in line:
            try:
                return int(line.split(":")[1].strip())
            except ValueError:
                return None
    return None


def thunderbolt_ips() -> dict:
    """IPs on bridge0 / Thunderbolt interfaces, if any (macOS)."""
    if sys.platform != "darwin":
        return {}
    ips = {}
    for iface in ["bridge0", "en1", "en2", "en3", "en4", "en5", "en6"]:
        ip = _run(["ipconfig", "getifaddr", iface])
        if ip:
            ips[iface] = ip
    return ips


def collect(with_mlx: bool = True) -> dict:
    info = {
        "hostname": socket.gethostname(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "ram_bytes": physical_ram_bytes(),
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if sys.platform == "darwin":
        info["chip"] = _sysctl("machdep.cpu.brand_string")
        info["macos_version"] = _run(["sw_vers", "-productVersion"])
        info["macos_build"] = _run(["sw_vers", "-buildVersion"])
        info["cpu_cores"] = int(_sysctl("hw.ncpu") or 0)
        info["gpu_cores"] = gpu_cores_macos()
        info["thunderbolt_ips"] = thunderbolt_ips()
        info["rdma_devices"] = (_run(["ibv_devices"]) or "").strip() or None
    else:
        info["chip"] = platform.processor() or None
        info["os_version"] = platform.platform()
        info["cpu_cores"] = int(_run(["nproc"]) or 0)
    if with_mlx:
        import mlx.core as mx

        info["mlx_version"] = mx.__version__
        try:
            import mlx_lm

            info["mlx_lm_version"] = mlx_lm.__version__
        except Exception:
            info["mlx_lm_version"] = None
        info["mlx_default_device"] = str(mx.default_device())
        info["mlx_memory_limit_bytes"] = int(mx.get_memory_limit())
        info["metal_available"] = bool(mx.metal.is_available())
        if info["metal_available"]:
            info["metal_device_info"] = {k: v for k, v in (mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()).items()}
            ws = info["metal_device_info"].get("max_recommended_working_set_size")
            info["gpu_working_set_bytes"] = int(ws) if ws else None
        else:
            info["metal_device_info"] = None
            info["gpu_working_set_bytes"] = None
    return info


def usable_bytes(info: dict, criterion: str) -> int:
    """How much memory this node 'has' for the purpose of the bigger-than-one proof."""
    if criterion == "ram":
        return int(info["ram_bytes"])
    if criterion == "working-set":
        ws = info.get("gpu_working_set_bytes")
        return int(ws) if ws else int(info["ram_bytes"])
    raise ValueError(criterion)


def sanity_check() -> dict:
    import mlx.core as mx

    t0 = time.perf_counter()
    a = mx.ones((2048, 2048))
    s = (a @ a).sum()
    mx.eval(s)
    dt = time.perf_counter() - t0
    ok = abs(float(s.item()) - 2048**3) < 1.0
    return {"matmul_ok": ok, "matmul_seconds": round(dt, 4), "matmul_result": float(s.item())}


def to_markdown(info: dict) -> str:
    lines = [f"# Node: {info['hostname']}", ""]
    keys = [
        ("chip", "Chip"),
        ("cpu_cores", "CPU cores"),
        ("gpu_cores", "GPU cores"),
        ("ram_bytes", "RAM"),
        ("gpu_working_set_bytes", "GPU max recommended working set"),
        ("mlx_memory_limit_bytes", "MLX default memory limit"),
        ("macos_version", "macOS"),
        ("os_version", "OS"),
        ("python", "Python"),
        ("mlx_version", "mlx"),
        ("mlx_lm_version", "mlx-lm"),
        ("mlx_default_device", "MLX default device"),
        ("thunderbolt_ips", "Thunderbolt / bridge IPs"),
        ("rdma_devices", "RDMA devices (ibv_devices)"),
        ("collected_at", "Collected"),
    ]
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    for k, label in keys:
        if k not in info or info[k] in (None, {}, ""):
            continue
        v = info[k]
        if k.endswith("_bytes"):
            v = f"{int(v) / 2**30:.1f} GiB ({int(v)} bytes)"
        lines.append(f"| {label} | {v} |")
    if info.get("metal_device_info"):
        lines += ["", "## mx.device_info()", "", "```json", json.dumps(info["metal_device_info"], indent=2), "```"]
    if info.get("sanity"):
        lines += ["", "## Sanity check", "", "```json", json.dumps(info["sanity"], indent=2), "```"]
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="run a matmul sanity check; non-zero exit on failure")
    p.add_argument("--write", action="store_true", help="write results/hardware/<hostname>.json and .md")
    p.add_argument("--out-dir", default="results/hardware")
    p.add_argument("--markdown", action="store_true", help="print markdown instead of JSON")
    args = p.parse_args()

    info = collect()
    if args.check or args.write:
        info["sanity"] = sanity_check()
    if args.write:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        host = info["hostname"].split(".")[0]
        (out / f"{host}.json").write_text(json.dumps(info, indent=2, default=str) + "\n")
        (out / f"{host}.md").write_text(to_markdown(info))
        print(f"wrote {out / (host + '.json')} and {out / (host + '.md')}")
    if args.markdown:
        print(to_markdown(info))
    else:
        print(json.dumps(info, indent=2, default=str))
    if args.check:
        ok = info["sanity"]["matmul_ok"]
        print(f"\nSANITY: {'OK' if ok else 'FAILED'}  mlx={info['mlx_version']} mlx-lm={info.get('mlx_lm_version')} "
              f"device={info['mlx_default_device']} ram={info['ram_bytes'] / 2**30:.1f} GiB")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
