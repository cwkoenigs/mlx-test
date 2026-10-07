#!/usr/bin/env bash
# Train with a config, on one node or on the cluster. Extra args after -- go to mlx_train.train.
#   ./scripts/run_train.sh configs/10m.yaml                      # this node only
#   ./scripts/run_train.sh configs/10m.yaml --cluster            # both Macs (hosts.json)
#   ./scripts/run_train.sh configs/10m.yaml --cluster --backend jaccl
#   ./scripts/run_train.sh configs/10m.yaml --local 2            # 2 processes on this machine
#   ./scripts/run_train.sh configs/10m.yaml -- --no-compile --tag nocompile
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"; need_venv; cd "$ROOT"
[[ -n "${1:-}" ]] || die "usage: $0 <config.yaml> [--cluster | --local N] [--backend B] [-- train args]"
CFG="$1"; shift
MODE=single; N=2; LAUNCH=()
while (($#)); do
  case "$1" in
    --cluster) MODE=cluster; shift ;;
    --local) MODE=local; N="$2"; shift 2 ;;
    --backend) LAUNCH+=(--backend "$2"); shift 2 ;;
    --hostfile) LAUNCH+=(--hostfile "$2"); shift 2 ;;
    --) shift; break ;;
    *) die "unknown argument $1" ;;
  esac
done
case "$MODE" in
  single)  exec "$PY" -m mlx_train.train --config "$CFG" "$@" ;;
  local)   exec "$ROOT/scripts/launch.sh" -n "$N" ${LAUNCH[@]+"${LAUNCH[@]}"} -m mlx_train.train --config "$CFG" "$@" ;;
  cluster) exec "$ROOT/scripts/launch.sh" ${LAUNCH[@]+"${LAUNCH[@]}"} -m mlx_train.train --config "$CFG" "$@" ;;
esac
