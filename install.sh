#!/usr/bin/env bash
# One-shot install on a Mac (run on BOTH Macs after cloning to the SAME path):
#   ./install.sh               # set up the node (uv, Python 3.12, pinned MLX env, sanity check,
#                              #   hardware record) AND install the Claude Code skill user-wide
#   ./install.sh --skill-only  # only (re)install the skill
#   ./install.sh --copy        # copy the skill instead of symlinking it
# The skill is installed to ~/.claude/skills/mlx-cluster so Claude Code can use it from any
# directory on this Mac ("/mlx-cluster" or just by asking about the cluster).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
SKILL_ONLY=0; COPY=0
for a in "$@"; do case "$a" in --skill-only) SKILL_ONLY=1 ;; --copy) COPY=1 ;; *) echo "unknown option $a" >&2; exit 1 ;; esac; done

if (( ! SKILL_ONLY )); then
  "$ROOT/scripts/setup_node.sh"
fi

SRC="$ROOT/.claude/skills/mlx-cluster"
DEST="$HOME/.claude/skills/mlx-cluster"
mkdir -p "$HOME/.claude/skills"
if [[ -L "$DEST" ]]; then rm "$DEST"
elif [[ -e "$DEST" ]]; then mv "$DEST" "$DEST.bak-$(date +%Y%m%d-%H%M%S)"; echo "backed up existing $DEST"; fi
if (( COPY )); then cp -R "$SRC" "$DEST"; else ln -s "$SRC" "$DEST"; fi
printf '%s\n' "$ROOT" > "$HOME/.claude/skills/mlx-cluster-repo-path"
echo "skill installed: $DEST -> $SRC"
echo "repo path recorded in ~/.claude/skills/mlx-cluster-repo-path"
echo
echo "Next: ./tests/selfcheck.sh   (then see .claude/skills/mlx-cluster/SKILL.md or ask Claude Code to 'set up the mlx cluster')"
