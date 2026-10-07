#!/usr/bin/env bash
# Phase 4: run the benchmark matrix and regenerate results/{single_node,benchmarks}.md.
#   ./scripts/benchmark_matrix.sh                       # configs 10m/100m/500m
#   CONFIGS="configs/10m.yaml" ./scripts/benchmark_matrix.sh
# Per config: single node (compile on + off), 2 nodes ring (hosts.json), 2 nodes jaccl
# (hosts.jaccl.json if present). Run it on the Mac Studio (rank 0 / the baseline node).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"; need_venv; cd "$ROOT"
CONFIGS="${CONFIGS:-configs/10m.yaml configs/100m.yaml configs/500m.yaml}"
for c in $CONFIGS; do
  log "$c: single node (compile)";      "$ROOT/scripts/run_train.sh" "$c" -- --tag single
  log "$c: single node (no compile)";   "$ROOT/scripts/run_train.sh" "$c" -- --tag nocompile --no-compile
  if [[ -f "$HOSTFILE" ]]; then
    log "$c: 2 nodes, ring";            "$ROOT/scripts/run_train.sh" "$c" --cluster -- --tag ring
  else
    log "no $HOSTFILE - skipping 2-node runs"
  fi
  if [[ -f "$ROOT/hosts.jaccl.json" ]]; then
    log "$c: 2 nodes, jaccl";           "$ROOT/scripts/run_train.sh" "$c" --cluster --hostfile "$ROOT/hosts.jaccl.json" -- --tag jaccl
  fi
done
"$PY" -m mlx_train.report
