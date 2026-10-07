#!/usr/bin/env bash
# Phase 2/3 gate: distributed smoke test + link micro-benchmark on the cluster (hosts.json).
#   ./scripts/run_smoke.sh                 # both Macs
#   ./scripts/run_smoke.sh --backend jaccl # RDMA hostfile
#   ./scripts/run_smoke.sh -n 2            # 2 processes on this machine
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
exec "$ROOT/scripts/launch.sh" "$@" -m mlx_cluster.smoke
