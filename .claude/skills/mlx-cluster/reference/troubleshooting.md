# Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `check_cluster.sh`: `FAIL ssh/repo` | `ssh -o BatchMode=yes host` fails: Remote Login off, key not copied, different username (use `~/.ssh/config`). |
| `FAIL: repo not found at <path> on host` | The repo must live at the same absolute path on both Macs (or symlink both to e.g. `/opt/mlx-cluster` and always run from there). |
| `FAIL: no .venv` / versions differ | Run `./scripts/setup_node.sh` on that Mac; both must use the committed `uv.lock` (`uv sync --frozen`). |
| `mlx.launch` prints `Failed to change directory` | Same path problem as above (`--cwd` is the local repo path). |
| `exec: python: not found` or import errors on the remote | Always launch through `scripts/launch.sh` (it passes `--python <venv>`); ssh shells do not activate venvs. |
| Ring: `Address already in use` / connect refused | A previous run left processes behind: `pkill -f mlx_train` / `pkill -f mlx_cluster` on both Macs, or pass `--starting-port 40000` to `mlx.launch`. Hosts in `--hosts` must be IPs. |
| Ring hangs at start | The IPs in `hosts.json` are not reachable on the Thunderbolt link (`ping`), firewall blocks the port (System Settings -> Network -> Firewall), or `ifconfig` settings were lost after reboot -> rerun `make_hostfile.sh`. Add `--env MLX_RING_VERBOSE=1` for logs. |
| 2-node run much slower than 1 node | Expected for small models over TCP: all-reduce every step. Compare `results/cluster/smoke-*.json` bandwidth with the gradient bytes per step (`param_bytes` in the summary). Try `--connections-per-ip 4`, bigger models/batches, or jaccl. |
| `init check` not identical after broadcast | Should never happen; it means the collectives are corrupting data. Rerun the smoke test; check versions. |
| Proof: `FATAL: this plan does not fit` | Lower `--total-gb`, raise `--fill` (close apps), or give `--split`. Use `--plan-only` to iterate. |
| Proof: model would NOT exceed the largest node | The pair's RAM ratio makes the physical-RAM claim impossible at that fill. Use `--criterion working-set` and document it, or raise `--fill`. |
| Proof: a node swaps / becomes unresponsive | Fill too high for what macOS and other apps need. Lower `--fill` (0.7) or increase `--headroom-gb`. Peak memory per node is in the report. |
| Proof: loss does not decrease | Lower `--lr` (0.05 default for float32 SGD on the fixed batch), or use `--dtype float32` if you tried bfloat16. 1-2 steps are not enough; default is 4. |
| `mx.compile` errors in train.py | Run with `-- --no-compile` to confirm, then report the config. Compile is a speed option only. |
| Metal: `[metal] Attempting to allocate ... which is greater than the maximum allowed buffer size` | One array exceeds Metal's max buffer; reduce `d_model` or vocab in the proof (per-layer matrices are d x 2.7d). |
| `mlx.distributed_config` dies with `EOFError: EOF when reading a line` | It waits for Enter after printing each node's setup commands; `scripts/make_hostfile.sh` now pipes `yes ''` into it. |
| jaccl: `No IPv4-mapped GID for this device ... assign a link-local address` | The RDMA interface (e.g. `en2`) has no IPv4 address of its own (being a `bridge0` member is not enough). Run the helper's `sudo ifconfig bridge0 down; sudo ifconfig en2 inet 192.168.0.X netmask 255.255.255.252` on both Macs and put those IPs in the hostfile. Not persistent across reboots. |
| `No route to host` between the 192.168.0.x addresses although both `en2` are up | A `route add -host <peer> -interface en2` entry pins the peer to your *own* MAC. `sudo route delete -host <peer>` on each Mac; the `/30` route from `ifconfig` is all that is needed. |
| `check_cluster.sh` prints a Python `TypeError` and then "OK" | Fixed: it now reads both the helper's `{"backend","envs","hosts":[...]}` form and the plain list form. |
