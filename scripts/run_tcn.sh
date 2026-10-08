#!/usr/bin/env bash
# Train the CMD TCN in MLX on one Mac or on both (data-parallel over jaccl/ring).
#   ./scripts/run_tcn.sh --data-root ~/data/cmd --epochs 1 --test-forward            # this Mac
#   ./scripts/run_tcn.sh --cluster --data-root ~/data/cmd --epochs 1 --test-forward  # both Macs (hosts.jaccl.json)
#   ./scripts/run_tcn.sh --cluster --hostfile hosts.json ...                          # both Macs over ring
#   ./scripts/run_tcn.sh --local 2 --data-root <smoke> --max-steps 5                 # 2 processes on this Mac
# Both Macs need the data at the same path. Any other args go to cmd_tcn_mlx.train (--help).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"; need_venv; cd "$ROOT"
MODE=single; N=2; HF="$ROOT/hosts.jaccl.json"; ARGS=()
while (($#)); do
  case "$1" in
    --cluster) MODE=cluster; shift ;;
    --local) MODE=local; N="$2"; shift 2 ;;
    --hostfile) HF="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
case "$MODE" in
  single)  exec "$PY" -m cmd_tcn_mlx.train ${ARGS[@]+"${ARGS[@]}"} ;;
  local)   exec "$ROOT/scripts/launch.sh" -n "$N" -m cmd_tcn_mlx.train ${ARGS[@]+"${ARGS[@]}"} ;;
  cluster) [[ -f "$HF" ]] || die "no hostfile at $HF"; exec "$ROOT/scripts/launch.sh" --hostfile "$HF" -m cmd_tcn_mlx.train ${ARGS[@]+"${ARGS[@]}"} ;;
esac
