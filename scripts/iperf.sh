#!/usr/bin/env bash
# Phase 2: raw TCP bandwidth over the Thunderbolt link (needs: brew install iperf3 on both Macs).
#   on node A:  ./scripts/iperf.sh server
#   on node B:  ./scripts/iperf.sh client 10.10.10.1      # A's Thunderbolt IP
# The client writes results/cluster/iperf-<timestamp>.txt.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"; cd "$ROOT"
command -v iperf3 >/dev/null 2>&1 || die "iperf3 not installed (brew install iperf3)"
case "${1:-}" in
  server) exec iperf3 -s ;;
  client) [[ -n "${2:-}" ]] || die "usage: $0 client <server-ip>"
          mkdir -p results/cluster; out="results/cluster/iperf-$(date +%Y%m%d-%H%M%S).txt"
          { echo "# iperf3 -c $2 -P 4 -t 10  ($(hostname) -> $2)"; iperf3 -c "$2" -P 4 -t 10; } | tee "$out"
          log "wrote $out" ;;
  *) die "usage: $0 server | client <server-ip>" ;;
esac
