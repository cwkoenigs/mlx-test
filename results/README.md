# Results

Everything in here is produced by the scripts and is meant to be committed (small JSONL logs).

| Path | Produced by |
|---|---|
| `hardware/<host>.{json,md}` | `scripts/setup_node.sh` / `scripts/hardware_info.sh` (Phase 0) |
| `logs/<run>.jsonl`, `logs/<run>.summary.json` | every `mlx_train.train` / `torch_baseline` run |
| `benchmarks.jsonl` | one line per training run (append-only) |
| `single_node.md`, `benchmarks.md` | `scripts/report.sh` (from `benchmarks.jsonl`) |
| `cluster/smoke-<ts>.json`, `cluster/iperf-<ts>.txt` | `scripts/run_smoke.sh`, `scripts/iperf.sh` (Phase 2) |
| `bigger_than_one/proof-<ts>.{md,json}` | `scripts/prove_bigger_than_one.sh` |
| `tcn/<run>.summary.json`, `tcn/comparison.md` | `cmd_tcn.train`, `scripts/run_tcn.sh` (`cmd_tcn_mlx.train`), `cmd_tcn_mlx.report` |
| `SUMMARY.md` | written by hand from the files above; the report in `docs/` is built on it |
| `SUMMARY-2026-10-07-previous-pair.md` | the same summary for the previous pair of Macs (kept for comparison) |
