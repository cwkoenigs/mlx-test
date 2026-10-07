#!/usr/bin/env bash
# End-to-end functional check on ONE machine (Mac or Linux CPU), ~1-3 minutes:
#   single-node training, 2-process data-parallel training (ring), heterogeneous batch split,
#   CNN, distributed smoke test, the pipeline-parallel proof mechanics (small model), report.
# Run it after setup on each Mac, and before any cluster run:  ./tests/selfcheck.sh
source "$(dirname "${BASH_SOURCE[0]}")/../scripts/common.sh"; need_venv; cd "$ROOT"
export MLX_CLUSTER_RESULTS_DIR=""   # (reserved)
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
LOG="$TMP/log"; BENCH="$TMP/bench.jsonl"
pass=0; fail=0
run() {
  local name="$1"; shift
  printf '%-48s' "$name"
  if "$@" >"$TMP/out" 2>&1; then echo "PASS"; pass=$((pass+1)); else echo "FAIL"; fail=$((fail+1)); tail -25 "$TMP/out" | sed 's/^/    /'; fi
}
T="--set log_dir=$LOG --set benchmarks_file=$BENCH"
# shellcheck disable=SC2086
run "node sanity (nodeinfo --check)"              "$PY" -m mlx_cluster.nodeinfo --check
# shellcheck disable=SC2086
run "train tiny, single process"                  "$PY" -m mlx_train.train --config configs/tiny.yaml $T --tag single
# shellcheck disable=SC2086
run "train tiny, single, no compile"              "$PY" -m mlx_train.train --config configs/tiny.yaml $T --tag nocompile --no-compile
# shellcheck disable=SC2086
run "train tiny, 2 ranks ring (data parallel)"    "$ROOT/scripts/launch.sh" -n 2 -m mlx_train.train --config configs/tiny.yaml $T --tag ring2
# shellcheck disable=SC2086
run "train tiny, 2 ranks, uneven batch [2,1]"     "$ROOT/scripts/launch.sh" -n 2 -m mlx_train.train --config configs/tiny.yaml $T --set global_batch_size=9 --set "rank_batch_weights=[2,1]" --tag hetero
# shellcheck disable=SC2086
run "train cnn, single process"                   "$PY" -m mlx_train.train --config configs/cnn.yaml $T --steps 8 --set global_batch_size=16 --set skip_first=2
run "smoke test, 2 ranks ring"                    "$ROOT/scripts/launch.sh" -n 2 -m mlx_cluster.smoke --sizes-mib 1,8 --reps 2 --out-dir "$TMP/cluster"
run "proof mechanics, 2 ranks (small model)"      "$ROOT/scripts/launch.sh" -n 2 -m mlx_cluster.prove_bigger_than_one --d-model 256 --n-heads 4 --seq-len 64 --batch 4 --split 3,2 --criterion none --steps 4 --out-dir "$TMP/proof"
run "report generation"                           "$PY" -m mlx_train.report --benchmarks "$BENCH" --out-dir "$TMP"
echo; echo "selfcheck: $pass passed, $fail failed"
[[ $fail -eq 0 ]]
