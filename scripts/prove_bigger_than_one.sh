#!/usr/bin/env bash
# THE PROOF: build + run + train a transformer whose parameters exceed either Mac's RAM, split
# across both Macs (pipeline parallel). Writes results/bigger_than_one/proof-<ts>.{md,json}.
#   ./scripts/prove_bigger_than_one.sh --plan-only       # see the plan (layers per node, GiB) first
#   ./scripts/prove_bigger_than_one.sh                   # fill both nodes to 80% and prove it
#   ./scripts/prove_bigger_than_one.sh --total-gb 100    # a specific size
#   ./scripts/prove_bigger_than_one.sh --criterion working-set   # if RAM sizes make 'ram' infeasible
#   ./scripts/prove_bigger_than_one.sh --local           # tiny 2-process functional check on this machine
# Close memory-hungry apps on both Macs first. Any other args go to mlx_cluster.prove_bigger_than_one.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"; need_venv; cd "$ROOT"
if [[ "${1:-}" == "--local" ]]; then
  shift
  exec "$ROOT/scripts/launch.sh" -n 2 -m mlx_cluster.prove_bigger_than_one \
    --d-model 256 --n-heads 4 --seq-len 64 --batch 4 --split 3,2 --criterion none --steps 4 "$@"
fi
LAUNCH=()
while (($#)); do
  case "$1" in
    --backend|--hostfile) LAUNCH+=("$1" "$2"); shift 2 ;;
    *) break ;;
  esac
done
exec "$ROOT/scripts/launch.sh" "${LAUNCH[@]}" -m mlx_cluster.prove_bigger_than_one "$@"
