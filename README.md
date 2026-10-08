# mlx-test: MLX deep learning on a two-Mac Thunderbolt cluster

Train and benchmark deep learning models in [MLX](https://github.com/ml-explore/mlx) on a
MacBook Pro (M2 Pro, 16 GB) and a Mac Studio (M5 Max, 36 GB) connected by Thunderbolt, and **prove that the two Macs
together hold and train a model larger than either Mac's memory**. Everything is built and
trained from scratch on synthetic, reproducible data (no pretrained models, no downloads).

The plan: [`docs/MLX_DEEP_LEARNING.md`](docs/MLX_DEEP_LEARNING.md).
The driver: the Claude Code skill [`.claude/skills/mlx-cluster/SKILL.md`](.claude/skills/mlx-cluster/SKILL.md).

## Quick start (on **both** Macs, same path)

```bash
git clone <this repo> ~/mlx-test && cd ~/mlx-test   # same absolute path on both Macs
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
| `mlx_train/` | Training harness: decoder-only transformer + CNN (`models.py`), deterministic sharded datasets (`data.py`), single-node and data-parallel loop with JSONL logging, `mx.compile`, AdamW (`train.py`), report generator (`report.py`), optional PyTorch/MPS baseline (`torch_baseline.py`). |
| `mlx_cluster/` | Cluster tooling: node facts (`nodeinfo.py`), collective helpers (`comm.py`), smoke test + link benchmark (`smoke.py`), **the proof** (`prove_bigger_than_one.py`). |
| `configs/` | `tiny`, `10m`, `100m`, `500m`, `cnn`, `10m-hetero` (uneven batch split for a faster node). |
| `scripts/` | Setup, hostfile, cluster check, iperf, `mlx.launch` wrapper, train/smoke/proof/matrix runners. |
| `tests/selfcheck.sh` | One-machine functional check (runs on Linux CPU too). |
| `.claude/skills/mlx-cluster/` | The skill + references (networking, RDMA/jaccl, troubleshooting, verified mlx 0.32.3 API notes). |
| `results/` | Hardware records, logs, benchmark tables, cluster measurements, proof reports, `SUMMARY.md`. |
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
full 305 GiB dataset streams from disk on a 36 GB Mac. Needs `uv sync --extra torch --extra cmd`.

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

Validated in a Linux CPU container with two local ring processes (`tests/selfcheck.sh`: training
single/distributed/uneven/CNN, smoke test, proof mechanics, report). Not yet run on the Macs:
the Thunderbolt/RDMA steps, Metal memory numbers, the real-size proof and the benchmark matrix
are what the skill walks you through. `torch_baseline.py` is unexecuted (no PyTorch here).
