# RDMA over Thunderbolt: the `jaccl` backend (optional Phase 5)

Requirements (mlx 0.32.3 docs): **macOS 26.2 or newer on both Macs**, **Thunderbolt 5 on both**
(MacBook Pro M4 Pro: yes; Mac Studio: M4 Max / M3 Ultra yes, M2 Ultra no), a direct cable
(two nodes = a complete mesh). Check the Studio's chip in `results/hardware/` first.

## Enable RDMA once per machine (Recovery mode)
1. Shut down, boot into Recovery (hold the power button until "Loading startup options").
2. Utilities -> Terminal.
3. `rdma_ctl enable`
4. Reboot. Verify in a normal shell: `ibv_devices` lists `rdma_enX` devices.

## Hostfile with RDMA devices
```bash
./scripts/make_hostfile.sh studio.local macbook.local --backend jaccl --output hosts.jaccl.json
```
Produces entries like `{"ssh": "...", "ips": [...], "rdma": [null, "rdma_en5"]}` (the device that
reaches each other host; `null` for self). Launch with `--backend jaccl --hostfile hosts.jaccl.json`
(our wrappers: `--hostfile hosts.jaccl.json`; the file carries the backend). The launcher sets
`MLX_JACCL_COORDINATOR=<rank0 ip>:<port>` and `MLX_IBV_DEVICES` for the processes.

## Why it matters
Ring over TCP does a socket round trip per collective; the data-parallel step does one bucketed
all-reduce per 32 MiB of gradients, so latency matters most for small models. jaccl is what makes
tensor parallelism viable in MLX; for this experiment it should show up as a higher 2-node
speedup on the 10M/100M rows of `results/benchmarks.md`.

## Debugging
`.venv/bin/mlx.distributed_config --verbose --hosts studio.local,macbook.local --over thunderbolt --dot`
prints the discovered Thunderbolt topology (pipe into `dot -Tpng` if graphviz is installed).
"does not seem to have RDMA enabled" -> redo the Recovery-mode step on that Mac.
