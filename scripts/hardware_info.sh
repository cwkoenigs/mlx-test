#!/usr/bin/env bash
# Record this node's hardware/software facts in results/hardware/<hostname>.{json,md}.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"; need_venv; cd "$ROOT"
"$PY" -m mlx_cluster.nodeinfo --write --markdown "$@"
