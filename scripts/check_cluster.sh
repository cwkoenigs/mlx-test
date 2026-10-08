#!/usr/bin/env bash
# Phase 2/3 gate: can mlx.launch reach every host in hosts.json, and are the nodes identical?
#   ./scripts/check_cluster.sh [hosts.json]
# Checks per host: passwordless ssh, the repo at the SAME path, the venv, matching mlx/python
# versions, and that each listed IP answers ping. Exit 1 on any failure.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"; need_venv; cd "$ROOT"
HF="${1:-$HOSTFILE}"
[[ -f "$HF" ]] || die "no hostfile at $HF (run scripts/make_hostfile.sh or copy hosts.example.json)"
LOCAL_VER="$("$PY" -c 'import mlx.core as mx, platform; print(mx.__version__, platform.python_version())')"
log "local: $(hostname) mlx/python = $LOCAL_VER, repo = $ROOT"
FAIL=0
while IFS=$'\t' read -r host ips; do
  printf '%-28s ' "$host"
  if ! out="$(ssh -n -o BatchMode=yes -o ConnectTimeout=8 -o LogLevel=ERROR "$host" \
        "cd '$ROOT' 2>/dev/null || { echo NO_REPO; exit 1; }; '$PY' -c 'import mlx.core as mx, platform; print(mx.__version__, platform.python_version())' 2>/dev/null || echo NO_VENV" 2>&1)"; then
    echo "FAIL ssh/repo: $out"; FAIL=1; continue
  fi
  case "$out" in
    *NO_REPO*) echo "FAIL: repo not found at $ROOT on $host"; FAIL=1; continue ;;
    *NO_VENV*) echo "FAIL: no .venv on $host (run scripts/setup_node.sh there)"; FAIL=1; continue ;;
  esac
  if [[ "$out" != "$LOCAL_VER" ]]; then echo "FAIL: versions differ ($out vs local $LOCAL_VER)"; FAIL=1; continue; fi
  pingres=""
  for ip in $ips; do
    if ping -c 1 -W 1000 "$ip" >/dev/null 2>&1 || ping -c 1 -w 1 "$ip" >/dev/null 2>&1; then pingres+=" $ip:ok"; else pingres+=" $ip:NO-PING"; FAIL=1; fi
  done
  echo "ok  mlx/python $out  ips:${pingres:- (none listed)}"
done < <("$PY" -c '
import json, sys
d = json.load(open(sys.argv[1]))
hosts = d["hosts"] if isinstance(d, dict) else d   # helper writes {"backend","envs","hosts"}; hand-written files are a list
for h in hosts:
    print(h["ssh"] + "\t" + " ".join(h.get("ips", [])))' "$HF")
if (( FAIL )); then die "cluster check FAILED"; fi
log "cluster check OK - next: ./scripts/run_smoke.sh"
