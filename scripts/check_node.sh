#!/usr/bin/env bash
# Quick per-node sanity check: versions, device, matmul. Exit 1 on failure.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"; need_venv; cd "$ROOT"
exec "$PY" -m mlx_cluster.nodeinfo --check "$@"
