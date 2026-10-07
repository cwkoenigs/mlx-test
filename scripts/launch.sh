#!/usr/bin/env bash
# Thin wrapper around mlx.launch that pins the working directory and the venv interpreter on
# every node (so the same command works over ssh on both Macs).
#   ./scripts/launch.sh [--hostfile F] [-n N] [--backend ring|jaccl] [--env K=V]... -m <module> [args...]
# Default target: $HOSTFILE (hosts.json). -n N runs N processes on this machine instead.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"; need_venv; cd "$ROOT"
N=""; BACKEND=""; ENVS=(); MOD=()
while (($#)); do
  case "$1" in
    --hostfile) HOSTFILE="$2"; shift 2 ;;
    -n) N="$2"; shift 2 ;;
    --backend) BACKEND="$2"; shift 2 ;;
    --env) ENVS+=(--env "$2"); shift 2 ;;
    -m) shift; MOD=("$@"); break ;;
    *) die "unknown argument $1 (module and its args must come after -m)" ;;
  esac
done
[[ ${#MOD[@]} -gt 0 ]] || die "usage: $0 [--hostfile F | -n N] [--backend B] -m module [args]"
if [[ -n "$N" ]]; then
  TARGET=(-n "$N" --backend "${BACKEND:-ring}")
else
  [[ -f "$HOSTFILE" ]] || die "no hostfile at $HOSTFILE (scripts/make_hostfile.sh) - or use -n 2 for a local run"
  TARGET=(--hostfile "$HOSTFILE"); [[ -n "$BACKEND" ]] && TARGET+=(--backend "$BACKEND")
fi
log "mlx.launch ${TARGET[*]} --cwd $ROOT --python $PY -- -m ${MOD[*]}"
exec "$MLX_LAUNCH" "${TARGET[@]}" ${ENVS[@]+"${ENVS[@]}"} --cwd "$ROOT" --python "$PY" -- -m "${MOD[@]}"
