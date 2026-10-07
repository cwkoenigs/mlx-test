# Networking the two Macs (Phase 2)

## 1. Cable
Thunderbolt 5 cable between the MacBook Pro (TB5) and the Mac Studio (TB5 on M4 Max / M3 Ultra,
TB4 on M2 Ultra; TB4 works for the `ring` backend, RDMA/`jaccl` needs TB5 on both ends).

## 2. Passwordless SSH both ways (and to yourself)
On both Macs: System Settings -> General -> Sharing -> **Remote Login: on** (allow your user).
```bash
ssh-keygen -t ed25519 -N "" -f ~/.ssh/id_ed25519      # if you have no key yet
ssh-copy-id user@<other-mac>.local
ssh-copy-id user@$(hostname)                          # mlx.distributed_config ssh's into every host, including itself
ssh -o BatchMode=yes <other-mac>.local hostname       # must print without a prompt
```
Use the same username on both Macs if you can; otherwise put `Host studio.local\n  User <name>`
entries in `~/.ssh/config` on each side. `hosts.json` `ssh` fields are whatever `ssh <that>` works with.

## 3. IPs on the Thunderbolt link
Preferred (MLX's own helper, run from one node):
```bash
./scripts/make_hostfile.sh studio.local macbook.local              # prints the ifconfig commands to run
./scripts/make_hostfile.sh studio.local macbook.local --auto-setup # runs them (needs passwordless sudo)
```
`mlx.distributed_config --over thunderbolt` discovers which Thunderbolt ports are cabled
(`system_profiler SPThunderboltDataType`), takes `bridge0` down and assigns point-to-point IPs
to the Thunderbolt interfaces with `sudo ifconfig enX inet <ip> netmask ...` + `sudo route change`.
Those settings are **not persistent**: rerun after a reboot (or after unplugging the cable).

Manual alternative (System Settings -> Network -> Thunderbolt Bridge -> Details -> TCP/IP ->
Configure IPv4: Manually): `10.10.10.1/255.255.255.0` on the Studio, `10.10.10.2` on the MacBook,
then write `hosts.json` by hand from `hosts.example.json`. The first entry is rank 0.

Verify: `ping 10.10.10.2`, then `./scripts/check_cluster.sh`.

## 4. Measure the link
```bash
./scripts/iperf.sh server            # on the Studio       (brew install iperf3 on both)
./scripts/iperf.sh client 10.10.10.1 # on the MacBook -> results/cluster/iperf-<ts>.txt
./scripts/run_smoke.sh               # MLX collectives: latency, all_sum/all_gather GB/s, ping-pong
```
Thunderbolt 4/5 IP bridging typically lands in the 10-20 Gbit/s range for TCP; RDMA (jaccl) is
much lower latency. Record both numbers in `results/SUMMARY.md`.

## 5. hostfile format (mlx 0.32.3, `mlx/_distributed_utils/common.py`)
```json
[
  {"ssh": "studio.local",  "ips": ["10.10.10.1"]},
  {"ssh": "macbook.local", "ips": ["10.10.10.2"]}
]
```
Optional top-level form `{"backend": "ring", "envs": ["MLX_METAL_FAST_SYNCH=1"], "hosts": [...]}`
lets the file pick the backend. `rdma` arrays are added for jaccl (see rdma-jaccl.md).
Launcher flags that matter: `--hosts` (IPs only, ring), `--hostfile`, `--backend`,
`--starting-port` (ring listens on it +1 per rank/IP), `--connections-per-ip` (more TCP streams
between neighbours; try 2-4 if bandwidth looks low), `--cwd`, `--python`, `--env K=V`.
