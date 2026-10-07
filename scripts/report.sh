#!/usr/bin/env bash
# Regenerate results/single_node.md and results/benchmarks.md from results/benchmarks.jsonl.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"; need_venv; cd "$ROOT"
exec "$PY" -m mlx_train.report "$@"
