#!/usr/bin/env bash
# Phase 0: one-shot per-node setup (idempotent). Run on BOTH Macs.
#   ./scripts/setup_node.sh
# Installs uv if missing, Python 3.12, the pinned environment from uv.lock, runs the MLX sanity
# check and records the hardware in results/hardware/<hostname>.{json,md}.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
cd "$ROOT"

if ! command -v uv >/dev/null 2>&1; then
  log "installing uv (https://astral.sh/uv)"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
log "uv $(uv --version)"
uv python install 3.12 >/dev/null 2>&1 || true

EXTRA=()
[[ "$(uname -s)" == "Linux" ]] && EXTRA=(--extra cpu)   # CPU-only MLX off-Mac
log "syncing the pinned environment (uv sync --frozen ${EXTRA[*]:-})"
uv sync --frozen "${EXTRA[@]}"

log "MLX sanity check + hardware record"
"$PY" -m mlx_cluster.nodeinfo --check --write | tail -1

if [[ "$(uname -s)" == "Darwin" ]] && ! command -v iperf3 >/dev/null 2>&1; then
  log "optional: 'brew install iperf3' enables scripts/iperf.sh (raw link bandwidth)"
fi
log "node ready: $(hostname) at $ROOT"
