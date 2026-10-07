# shellcheck shell=bash
# Shared helpers; sourced by scripts/*.sh. Not meant to be run directly.
set -euo pipefail
# Logical path on purpose: if both Macs symlink the repo to the same path (e.g. /opt/mlx-cluster),
# mlx.launch's --cwd works on both even when the real checkout paths differ.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -L)"
PY="$ROOT/.venv/bin/python"
MLX_LAUNCH="$ROOT/.venv/bin/mlx.launch"
HOSTFILE="${HOSTFILE:-$ROOT/hosts.json}"

log() { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
die() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
need_venv() { [[ -x "$PY" ]] || die "no environment at $PY - run ./scripts/setup_node.sh first"; }
