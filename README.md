# mlx-test: can Apple Silicon stand in for cloud GPUs?

Cloud GPU capacity has been hard to get, so this repo tests whether a Mac Studio can be the team's
deep learning training machine. It trains and benchmarks models in [MLX](https://github.com/ml-explore/mlx)
and PyTorch (MPS) on a Mac Studio (M5 Max, 40 GPU cores, 48 GB) and a MacBook Pro (M5 Pro, 20 GPU cores,
48 GB) joined by a Thunderbolt 5 cable with RDMA. It covers a transformer ladder from 10M to 2B parameters,
the member-level claims model (a temporal convolutional network) at production data shape, and a
16B-parameter model trained across both Macs. Everything is built and trained from scratch on synthetic,
reproducible data (no pretrained models, no downloads, no real claims data).

## Results in brief (2026-10-08)

- **One Studio runs our stack.** MLX is 1.9x to 5.1x faster than PyTorch on transformers; PyTorch is
  faster on the claims TCN and on recurrent models. A 2B-parameter transformer trains in 18.9 GiB.
- **The claims model fits at production shape.** 863k members x 105 bins x 324 features (a 109 GiB
  sequence array) stream from disk; one epoch over the 690k-member training split takes 3.7 minutes in PyTorch.
- **Two Macs hold what one cannot.** A 16.1B-parameter, 60 GiB model trains split across both Macs
  (pipeline parallel), at 11 to 14 s per step.
- **Two Macs add speed only for some models:** 1.6x on the claims TCN, 1.16x on 1B to 2B transformers with
  gradient accumulation, and a slowdown for transformers that exchange gradients every step.

The full report, with method, tables and which models gain from a second Mac, is in [`docs/`](docs/index.md)
and builds into a local site:

```bash
uvx --with mkdocs-material mkdocs serve      # or: pip install -r requirements-docs.txt && mkdocs serve
# then open http://127.0.0.1:8000
```

The raw numbers: [`results/SUMMARY.md`](results/SUMMARY.md), [`results/benchmarks.md`](results/benchmarks.md),
[`results/tcn/comparison.md`](results/tcn/comparison.md), [`results/bigger_than_one/`](results/bigger_than_one/).
How to rerun every measurement: [`docs/reproduce.md`](docs/reproduce.md).
The original plan: [`docs/MLX_DEEP_LEARNING.md`](docs/MLX_DEEP_LEARNING.md).
The driver: the Claude Code skill [`.claude/skills/mlx-cluster/SKILL.md`](.claude/skills/mlx-cluster/SKILL.md).

## Quick start (on **both** Macs, same path)

```bash
git clone https://github.com/cwkoenigs/mlx-test.git ~/Developer/mlx-test && cd ~/Developer/mlx-test   # same path on both Macs
./install.sh          # uv + Python 3.12 + pinned mlx 0.32.3, sanity check, hardware record,
                      # and the "mlx-cluster" Claude Code skill in ~/.claude/skills
./tests/selfcheck.sh  # ~2 min: training, 2-process distributed, smoke test, proof mechanics
```

Then, from the Mac Studio (rank 0), once the Thunderbolt cable and passwordless ssh are in place:

```bash
./scripts/make_hostfile.sh studio.local macbook.local   # -> hosts.json (MLX's own helper)
./scripts/check_cluster.sh                              # ssh, same path, same versions, ping
./scripts/run_smoke.sh                                  # collectives: correctness, latency, GB/s
./scripts/run_train.sh configs/10m.yaml                 # single node
./scripts/run_train.sh configs/10m.yaml --cluster       # both Macs, data parallel
./scripts/prove_bigger_than_one.sh --plan-only          # how big a model the pair can hold
./scripts/prove_bigger_than_one.sh                      # the proof -> results/bigger_than_one/
./scripts/benchmark_matrix.sh                           # Phase 4 -> results/benchmarks.md
```

Or just ask Claude Code on either Mac: *"set up the mlx cluster and run the bigger-than-one proof"*.

## What is in here

| Path | What |
|---|---|
| `mlx_train/` | Training harness: decoder-only transformer + CNN (`models.py`), deterministic sharded datasets (`data.py`), single-node and data-parallel loop with JSONL logging, `mx.compile`, AdamW, gradient accumulation (`accumulate_steps`) and uneven per-rank batches (`rank_batch_weights`) (`train.py`), report generator (`report.py`), optional PyTorch/MPS baseline (`torch_baseline.py`). |
| `mlx_cluster/` | Cluster tooling: node facts (`nodeinfo.py`), collective helpers (`comm.py`), smoke test + link benchmark (`smoke.py`), **the proof** (`prove_bigger_than_one.py`). |
| `configs/` | `tiny`, `10m`, `100m`, `500m`, `1b` (float32), `2b` (bfloat16), `cnn`, `10m-hetero` / `10m-hetero3` (uneven batch split for a faster node). Override any key with `--set`, e.g. `--set accumulate_steps=16`. |
| `scripts/` | Setup, hostfile, cluster check, iperf, `mlx.launch` wrapper, train/smoke/proof/matrix runners. |
| `tests/selfcheck.sh` | One-machine functional check (runs on Linux CPU too). |
| `.claude/skills/mlx-cluster/` | The skill + references (networking, RDMA/jaccl, troubleshooting, verified mlx 0.32.3 API notes). |
| `cmd_tcn/`, `cmd_tcn_mlx/` | The claims TCN in PyTorch Lightning and its layer-for-layer MLX port (one Mac or both), with a parity check and a comparison report. |
| `docs/` | The report (`index`, `method`, `results`, `scaling`, `reproduce`, `references`), the original plan and the synthetic claims data spec. `mkdocs.yml` builds it into a local site. |
| `results/` | Hardware records, logs, benchmark tables, cluster measurements, proof reports, TCN runs, `SUMMARY.md`. |
| `pyproject.toml`, `uv.lock` | Pinned environment (Python 3.12, mlx 0.32.3). Extras: `cpu` (Linux), `llm` (mlx-lm, optional Phase 5), `torch`. |

## The proof, in one paragraph

`prove_bigger_than_one.py` reads both nodes' memory, plans a transformer whose parameter bytes
exceed the largest node's RAM, and splits its layers across the nodes in proportion to their
memory (pipeline parallel). Each node builds only its layers; the forward pass streams
activations over the Thunderbolt link with `mx.distributed.send/recv`, and four backward + SGD
steps run layer by layer with `mx.vjp` so only one layer's gradients exist at a time. It passes
when the measured parameter bytes exceed the largest node, every node stays within its own RAM,
and the loss decreases. The report lands in `results/bigger_than_one/proof-<ts>.md`.

## Synthetic CMD arrays (separate workload)

`scripts/generate_synthetic_cmd_arrays.py` implements the prompt in
[`docs/synthetic_cmd_generator_prompt.md`](docs/synthetic_cmd_generator_prompt.md): entirely synthetic
member-level CMD arrays (`train__CMD` 863,154 members, `one_year_forward_test__CMD` 847,568 members,
~305 GiB at full scale) for model-architecture, loader, memory-pressure and disk-I/O experiments.
It is standalone (numpy, plus pyarrow for the parquet labels: `uv sync --extra cmd`); copy it and
`scripts/synthetic_cmd_spec.py` into the member-level repo's `scripts/` to use it there.

```bash
P=.venv/bin/python
$P scripts/generate_synthetic_cmd_arrays.py --output-root /Volumes/data/cmd --dry-run --full   # plan + disk check
$P scripts/generate_synthetic_cmd_arrays.py --output-root /Volumes/data/cmd                    # smoke run (--scale 0.001)
$P scripts/generate_synthetic_cmd_arrays.py validate --output-root /Volumes/data/cmd            # contract checks
$P scripts/generate_synthetic_cmd_arrays.py --output-root /Volumes/data/cmd --full --resume     # the real thing (hours)
$P scripts/generate_synthetic_cmd_arrays.py --output-root /Volumes/data/cmd --full --stage-chunks --resume
$P scripts/generate_synthetic_cmd_arrays.py reassemble --output-root /Volumes/data/cmd --split train__CMD
```

Bounded-memory sequential `.npy` writer, deterministic per-(split, file, block) seeds (an interrupted
run resumed with `--resume` is byte-identical to a clean run, as is chunked-then-reassembled `X_seq`),
`structured` or `fast-io` value patterns, size/disk estimate with 15% headroom, and a validator that
checks every rule in the prompt with `mmap_mode='r'`.

## CMD temporal convolutional network (PyTorch Lightning)

`cmd_tcn/` trains a TCN on the synthetic CMD arrays: five residual blocks with dilations 1, 2, 4, 8, 16,
two causal dilated `Conv1d` layers per block (kernel 3, weight norm, dropout), receptive field 125 bins.
Input is `X_seq` (105 x 324) plus the historical `enrollment_mask` as a 325th channel and `X_static`
through a small projection; output is the 12 monthly allowed-cost targets `y`, trained with a masked
Huber loss in `log1p` dollars (months with `outcome_enrollment_mask == 0` are not scored). The loader
memory-maps the `.npy` files and reads **contiguous member blocks** (one block = one batch), so the
full 305 GiB dataset streams from disk on a 48 GB Mac (it did on a 36 GB one too). Needs `uv sync --extra torch --extra cmd`.

```bash
P=.venv/bin/python
$P -m cmd_tcn.train --data-root ~/data/cmd --epochs 1 --test-forward        # defaults: 512 ch, batch 1024, bf16, 8 workers
$P -m cmd_tcn.train --data-root ~/data/cmd --max-steps 300                    # throughput probe
$P -m cmd_tcn.train --data-root ~/data/cmd --epochs 1 --channels 128 --precision 32-true   # small fp32 variant
$P -m cmd_tcn.train --help
```

Each run writes `results/tcn/<run>.summary.json` (parameters, steps/s, members/s, GiB/s of `X_seq`
read, peak MPS memory, val and forward-test metrics). `tests/selfcheck.sh` runs five CPU steps on the
0.001-scale smoke data when Lightning is installed.

### The same TCN in MLX, on one Mac or both

`cmd_tcn_mlx/` is the PyTorch TCN ported layer for layer to MLX (`cmd_tcn_mlx.parity` copies the
torch weights across and checks the outputs agree), with the same block-streaming loader in plain
numpy and the data-parallel machinery from `mlx_train` (gradient averaging every step over jaccl or
ring, uneven batch split for the faster Mac). Both Macs need the data at the same path.

```bash
./scripts/run_tcn.sh --data-root ~/data/cmd --epochs 1 --test-forward                        # this Mac
./scripts/run_tcn.sh --cluster --data-root ~/data/cmd --epochs 1 --test-forward --rank-weights 2,1   # both Macs over jaccl
./scripts/run_tcn.sh --cluster --hostfile hosts.json --data-root ~/data/cmd --epochs 1       # both Macs over ring
.venv/bin/python -m cmd_tcn_mlx.parity --data-root ~/data/cmd                                 # torch == mlx check
.venv/bin/python -m cmd_tcn_mlx.report                                                        # results/tcn/comparison.md
```

Every run prints each `--log-every` step (loss, lr, steps/s, members/s, GiB/s of `X_seq`, peak
memory) and writes `results/tcn/<run>.summary.json` in the same schema as the PyTorch run, so
`cmd_tcn_mlx.report` puts PyTorch-on-one-Mac, MLX-on-one-Mac and MLX-on-two-Macs side by side.

## Status

Run end to end on the Macs on 2026-10-07 (previous pair: M2 Pro laptop and a 36 GB Studio, kept in
`results/SUMMARY-2026-10-07-previous-pair.md`) and 2026-10-08 (the current M5 Pro and 48 GB M5 Max pair):
link measurements, the transformer ladder in MLX and PyTorch, the two-Mac matrix over ring and jaccl, gradient
accumulation, the claims TCN in both frameworks on one and two Macs, and the 60 GiB bigger-than-one proof.
`tests/selfcheck.sh` also passes in a Linux CPU container with two local ring processes.

Open items:

- The claims arrays are synthetic, so the runs measure throughput, memory and I/O, not predictive value.
  The next step is a de-identified real extract on the Studio.
- The MLX TCN has only been run in float32 against PyTorch's bf16-mixed; `--dtype bfloat16` is unmeasured.
- The TCN runs of 2026-10-08 averaged members/s over training plus the forward test, so those figures
  understate training speed (see the note in [`docs/results.md`](docs/results.md)). Both TCN trainers now
  stop the throughput clock before the forward test.
- Two Studios have not been measured together; the 1.8x to 1.9x data-parallel figure for a Studio pair is an estimate.
