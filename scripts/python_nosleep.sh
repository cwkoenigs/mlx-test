#!/usr/bin/env bash
# The venv's python wrapped in caffeinate so a Mac does not go to idle-sleep mid-run (the Studio's
# default is 1 minute). Used as mlx.launch --python on every node and by the single-node wrappers.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -L)"
if [[ "$(uname -s)" == "Darwin" ]] && command -v caffeinate >/dev/null; then
  exec /usr/bin/caffeinate -i "$ROOT/.venv/bin/python" "$@"
fi
exec "$ROOT/.venv/bin/python" "$@"
