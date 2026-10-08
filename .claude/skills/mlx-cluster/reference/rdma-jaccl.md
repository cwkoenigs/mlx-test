# RDMA over Thunderbolt: the `jaccl` backend

Requirements (mlx 0.32.3): **macOS 26.2 or newer on both Macs**, **Thunderbolt 5 on both**, a direct
cable (two nodes = a complete mesh). This pair qualifies: MacBook Pro M5 Pro + Mac Studio M5 Max,
both macOS 27.0.1, both with RDMA enabled (`ibv_devices` lists `rdma_enX`). The link negotiates
at 80 Gb/s.

## Enable RDMA once per machine (Recovery mode)
1. Shut down, boot into Recovery (hold the power button until "Loading startup options").
2. Utilities -> Terminal.
3. `rdma_ctl enable`
4. Reboot. Verify in a normal shell: `ibv_devices` lists `rdma_enX` devices.

## Step by step (what worked on 2026-10-08)
1. Passwordless ssh in **all four** directions: each Mac to the other and each Mac to itself
   (`mlx.distributed_config` and `mlx.launch` ssh into every host in the file, including the one
   they run on). Remote Login must be on on both; `ssh-copy-id` to the other Mac *and* to yourself.
2. Discover the link and RDMA devices with MLX's helper (non-interactive wrapper):
   ```bash
   ./scripts/make_hostfile.sh Clarks-Mac-Studio.local Clarks-MacBook-Pro.local --backend jaccl --output hosts.jaccl.json
   ```
   First host = rank 0. It prints per-node `sudo ifconfig ...` setup commands and writes the file
   with `"rdma": [null, "rdma_en2"]` / `["rdma_en2", null]` (the interface that reaches the other
   host; `null` for self).
3. Give the RDMA interfaces their own IPv4 addresses, on **both** Macs (sudo; not persistent):
   ```bash
   sudo ifconfig bridge0 down
   sudo ifconfig en2 inet 192.168.0.1 netmask 255.255.255.252   # rank 0; 192.168.0.2 on rank 1
   ```
   Without this jaccl fails with `No IPv4-mapped GID for this device`. Do **not** add
   `route add -host ... -interface en2` entries (they break ARP: "No route to host").
4. Put those addresses in the hostfile `ips` (the helper fills in en0/en1, usually the LAN), and
   write a ring twin of the file for comparisons:
   ```bash
   # hosts.jaccl.json: ips 192.168.0.1 / 192.168.0.2 ; hosts.json: same hosts+ips, "backend": "ring"
   ./scripts/check_cluster.sh hosts.jaccl.json && ./scripts/run_smoke.sh --hostfile hosts.jaccl.json
   ```
   The launcher sets `MLX_JACCL_COORDINATOR=<rank0 ip>:<port>` and `MLX_IBV_DEVICES` for the processes.

## Measured on this pair (`results/cluster/smoke-20261008-0810*.json`)

| | ring (TCP over the same cable) | jaccl (RDMA) |
|---|---|---|
| all_sum latency, 1 float | 139 us | 21 us |
| all_sum 128 MiB | 6.6 GB/s | 9.3 GB/s |
| all_gather 128 MiB | 6.9 GB/s | 8.1 GB/s |
| 1 MiB send/recv round trip | 0.37 ms | 0.30 ms |

## Why it matters
Ring over TCP does a socket round trip per collective; the data-parallel step does one bucketed
all-reduce per 32 MiB of gradients, so latency matters most for small models. jaccl is what makes
tensor parallelism viable in MLX; for this experiment it shows up as a higher 2-node speedup on the
10M/100M rows of `results/benchmarks.md`.

## Debugging
`.venv/bin/mlx.distributed_config --verbose --hosts studio.local,macbook.local --over thunderbolt --dot`
prints the discovered Thunderbolt topology (pipe into `dot -Tpng` if graphviz is installed).
"does not seem to have RDMA enabled" -> redo the Recovery-mode step on that Mac.
More in `troubleshooting.md`.
