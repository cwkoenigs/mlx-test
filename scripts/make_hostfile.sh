#!/usr/bin/env bash
# Phase 2: generate hosts.json for the Thunderbolt link with MLX's own helper (mlx 0.32.3:
# mlx.distributed_config). Run from ONE node; it ssh's into both hosts (passwordless SSH required,
# including to the host you run it on, by name).
#
#   ./scripts/make_hostfile.sh studio.local macbook.local                 # ring over Thunderbolt IP
#   ./scripts/make_hostfile.sh studio.local macbook.local --backend jaccl # RDMA (macOS 26.2+, TB5)
#   ./scripts/make_hostfile.sh studio.local macbook.local --auto-setup    # also run the sudo ifconfig steps
#
# The first host becomes rank 0. Without --auto-setup the helper PRINTS the ifconfig/route commands
# to run on each node (they are not persistent across reboots). Output: hosts.json (or $HOSTFILE).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"; need_venv; cd "$ROOT"
[[ $# -ge 2 ]] || die "usage: $0 <host-rank0> <host-rank1> [--backend ring|jaccl] [--auto-setup] [--output FILE]"
HOSTS="$1,$2"; shift 2
BACKEND=ring; EXTRA=(); OUT="$HOSTFILE"
while (($#)); do
  case "$1" in
    --backend) BACKEND="$2"; shift 2 ;;
    --auto-setup) EXTRA+=(--auto-setup); shift ;;
    --output) OUT="$2"; shift 2 ;;
    *) die "unknown argument $1" ;;
  esac
done
log "mlx.distributed_config --hosts $HOSTS --over thunderbolt --backend $BACKEND ${EXTRA[*]:-} -> $OUT"
# The helper pauses with "Enter to continue" after printing each node's setup commands; feed it
# newlines so it also works from scripts / Claude Code (mlx 0.32.3 raises EOFError otherwise).
yes '' | "$ROOT/.venv/bin/mlx.distributed_config" --verbose --hosts "$HOSTS" --over thunderbolt --backend "$BACKEND" ${EXTRA[@]+"${EXTRA[@]}"} --output-hostfile "$OUT"
echo; log "hostfile $OUT:"; cat "$OUT"
cat <<'MSG'

NOTE: the "ips" above are whatever en0/en1 report (often the office LAN). For jaccl the RDMA
interfaces MUST carry their own IPv4 address (the printed "sudo ifconfig <en> inet ..." lines), and
the hostfile "ips" should be those addresses. Do NOT run "route add -host ... -interface enX": it
pins the peer to your own MAC and gives "No route to host"; the /30 from ifconfig is enough.
If the helper could not find a Thunderbolt link, write the file by hand (see hosts.example.json).
Then: ./scripts/check_cluster.sh && ./scripts/run_smoke.sh
MSG
