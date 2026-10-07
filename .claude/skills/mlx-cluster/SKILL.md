---
name: mlx-cluster
description: Set up, verify, benchmark and train with MLX on the two-Mac Thunderbolt cluster in this repo (MacBook Pro M4 Pro + Mac Studio). Use when the user mentions MLX, mlx.launch, Thunderbolt bridge, hosts.json, ring/jaccl/RDMA, distributed or data-parallel training on Macs, the benchmark matrix, results/SUMMARY.md, or the proof that a model bigger than one Mac loads and trains across both. Covers Phases 0-4 of docs/MLX_DEEP_LEARNING.md end to end.
---

# MLX two-Mac cluster

Everything here is deep learning in MLX on models **we build and train ourselves** (synthetic,
reproducible data; no model downloads, no Hugging Face). Two machines: **Node A** MacBook Pro
M2 Pro (16 GB, Thunderbolt 4, no RDMA), **Node B** Mac Studio M5 Max (36 GB). Link negotiates at 40 Gb/s; `ring` only.

## Locate the repo first

- Working inside the repo: the root is the directory containing `pyproject.toml` and `scripts/`.
- Installed user-wide (`~/.claude/skills/mlx-cluster` is a symlink): the root is
  `cat ~/.claude/skills/mlx-cluster-repo-path` (or `readlink -f ~/.claude/skills/mlx-cluster`, three
  levels up). `cd` there before running anything.
- All Python runs through the pinned venv: `.venv/bin/python` (uv-managed; Python 3.12, mlx 0.32.3,
  see `uv.lock`). Never `pip install` into it. To change a dependency: edit `pyproject.toml`,
  `uv lock`, commit, then `uv sync --frozen` on **both** Macs.
- Both Macs need the repo at the **same absolute path** (or a symlink such as `/opt/mlx-cluster` on
  both, and always run from that path). `mlx.launch` cd's to the same `--cwd` over ssh.

## Workflow (in order; each step is a gate)

| Phase | Do | Pass when |
|---|---|---|
| 0 Node setup (each Mac) | `./install.sh` then `./tests/selfcheck.sh` | selfcheck prints `0 failed`; `results/hardware/<host>.md` exists |
| 1 Single-node training | `./scripts/run_train.sh configs/10m.yaml` (then 100m, 500m, cnn; `-- --no-compile` variant) | loss falls, JSONL + summary in `results/logs/`, row in `results/benchmarks.jsonl` |
| 2 Network | Thunderbolt cable, passwordless ssh both ways, `./scripts/make_hostfile.sh <studio> <macbook>`, `./scripts/check_cluster.sh`, `./scripts/run_smoke.sh`, optional `scripts/iperf.sh` | `check_cluster` OK, `SMOKE: OK`, bandwidth numbers in `results/cluster/` |
| 3 Distributed training | `./scripts/run_train.sh configs/10m.yaml --cluster` | `init check` says identical params on both ranks, loss curve matches the 1-node run, summary shows 2 hosts |
| **Proof** | `./scripts/prove_bigger_than_one.sh --plan-only`, then without `--plan-only` | `PROOF: PASS` and `results/bigger_than_one/proof-*.md` |
| 4 Benchmark matrix | `./scripts/benchmark_matrix.sh` (on the Studio), then fill `results/SUMMARY.md` | `results/single_node.md`, `results/benchmarks.md` regenerated; SUMMARY answers the 3 questions |
| 5 (optional) RDMA | `reference/rdma-jaccl.md`; `make_hostfile.sh ... --backend jaccl --output hosts.jaccl.json`; rerun smoke/train/matrix with `--hostfile hosts.jaccl.json` | jaccl rows in the matrix |

Run single-node things first; only add the second Mac once Phase 2 passes. Commit results as
you go (`results/` is meant to live in the repo; `hosts.json` is git-ignored).

## The proof (what "bigger than one Mac" means here)

`mlx_cluster/prove_bigger_than_one.py` builds a decoder-only transformer from the same `Block`
as the training harness, sized from the nodes' **measured** memory so that total parameter
bytes exceed the largest single node's RAM (`--criterion ram`, default) and splits the layers
across the two Macs in proportion to their memory (pipeline parallel: rank 0 = embedding +
first layers, last rank = last layers + head + loss). It then runs a forward pass across the
link and `--steps` (default 4) pipeline-parallel backward + SGD steps, layer-by-layer with
`mx.vjp` so only one layer's gradients exist at a time. PASS requires: parameters > largest
node, every node's peak within its own RAM, finite loss, loss decreasing. Report:
`results/bigger_than_one/proof-<ts>.md` + `.json`.

Practicalities:
- Always `--plan-only` first; it prints layers/GiB per node and says `BIGGER` or `NOT bigger`.
- Close other apps on both Macs. Default `--fill 0.80` of RAM per node, 2 GiB headroom.
- If the plan says the model cannot exceed the biggest node even at full fill (e.g. 48 GB +
  256 GB), the claim against physical RAM is impossible for that pair; use
  `--criterion working-set` (Metal's max recommended working set) and say so in the write-up.
- Expect minutes per step for a 100+ GiB float32 model; that's the compute, not a hang.
  Watch `built layer k/n` progress lines. `--dtype bfloat16` halves bytes per parameter (more
  parameters for the same GiB, faster matmuls); the claim is about bytes, either is fine.
- Nodes are listed in `hosts.json` order; put the Mac Studio first (rank 0) so it gets the
  bigger share and the embedding.

## Commands cheat sheet

```bash
./install.sh                                   # node setup + skill (each Mac)
./tests/selfcheck.sh                           # 1-machine functional check (~2 min)
./scripts/run_train.sh configs/10m.yaml        # single node
./scripts/run_train.sh configs/10m.yaml --cluster [--backend jaccl] [-- --no-compile --steps 50]
./scripts/run_train.sh configs/10m.yaml --local 2       # 2 processes on one Mac (debugging)
./scripts/run_smoke.sh [--backend jaccl] [-n 2]
./scripts/prove_bigger_than_one.sh [--plan-only] [--total-gb N] [--criterion working-set] [--split 30,10]
./scripts/benchmark_matrix.sh && cat results/benchmarks.md
./scripts/launch.sh --hostfile hosts.json -m <module> [args]   # anything else under mlx.launch
.venv/bin/python -m mlx_train.train --config configs/10m.yaml --set model.d_model=512 --set steps=50
```

`run_train.sh` logs to `results/logs/<name>-<tag>-<ranks>n-<backend>-<ts>.jsonl` (+ `.summary.json`)
and appends a row to `results/benchmarks.jsonl`; `scripts/report.sh` rebuilds the markdown tables.

## Interpreting results

- Tokens/sec in the logs are **global** (both ranks). Speedup = steps/sec vs the 1-node run with
  the same config and compile setting (`report.py` computes it).
- Keep the **global batch** fixed between 1-node and 2-node runs so loss curves are comparable.
  `configs/10m-hetero.yaml` shows `rank_batch_weights` for giving the Studio a bigger share.
- Gradient all-reduce happens every step (`nn.average_gradients`, 32 MiB buckets). Over plain
  TCP ring on Thunderbolt expect little or no speedup for ~10M models and better amortisation for
  100M-500M. The slower node (M2 Pro) sets the pace. `jaccl` (RDMA) would cut latency a lot, but the M2 Pro is Thunderbolt 4 with no RDMA, so it is not available for this pair.
- `init check` in the 2-node output records whether seeding alone gave identical parameters on
  both chips and confirms they are identical after the rank-0 broadcast (the run aborts if not).
- Peak memory per rank is `mx.get_peak_memory()`; compare with the Metal working set in
  `results/hardware/`.

## When something fails

Read `reference/troubleshooting.md` (ssh/cwd/venv mismatches, ring port/IP errors, hangs,
memory). Networking details: `reference/networking.md`. RDMA: `reference/rdma-jaccl.md`.
Exact mlx 0.32.3 API facts this repo relies on: `reference/api-notes.md`. The original plan with
the three questions to answer: `docs/MLX_DEEP_LEARNING.md`.

## Deliverables checklist (keep in the repo)

- `results/hardware/<host>.md` for both Macs (Phase 0)
- `results/logs/*.jsonl` + `.summary.json`, `results/single_node.md` (Phase 1)
- `results/cluster/smoke-*.json`, `iperf-*.txt`, `hosts.example.json` updated to the real names (Phase 2)
- `results/benchmarks.md` (Phase 4), `results/bigger_than_one/proof-*.md` (the proof)
- `results/SUMMARY.md`: the three questions + "is the second Mac worth it, for which workloads"
