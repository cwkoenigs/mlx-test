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
"$ROOT/.venv/bin/mlx.distributed_config" --verbose --hosts "$HOSTS" --over thunderbolt --backend "$BACKEND" ${EXTRA[@]+"${EXTRA[@]}"} --output-hostfile "$OUT"
echo; log "hostfile $OUT:"; cat "$OUT"
cat <<'MSG'

If the helper could not find a Thunderbolt link, write the file by hand (see hosts.example.json):
  [{"ssh": "studio.local", "ips": ["10.10.10.1"]}, {"ssh": "macbook.local", "ips": ["10.10.10.2"]}]
Then: ./scripts/check_cluster.sh && ./scripts/run_smoke.sh
MSG
