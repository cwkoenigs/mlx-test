# MLX Deep Learning on a Two-Mac Cluster

> **Status (2026-10-07):** this plan is implemented in this repo. Every CLI flag, hostfile
> field and API named below was verified against the pinned `mlx==0.32.3` (see
> `.claude/skills/mlx-cluster/reference/api-notes.md`). Phase-by-phase "Implemented as" notes
> point at the scripts. The Claude Code skill `.claude/skills/mlx-cluster/SKILL.md` drives the
> whole workflow on either Mac. Nothing here downloads pretrained models: the experiment builds
> and trains its own models in MLX on synthetic, reproducible data.

## Goal

Evaluate **MLX** as a deep learning framework (training, not just inference) on Apple Silicon, using:

- **Node A:** MacBook Pro, M4 Pro (Thunderbolt 5)
- **Node B:** Mac Studio (check the chip; M4 Max and M3 Ultra have Thunderbolt 5, M2 Ultra has Thunderbolt 4)

The experiment answers three questions:

1. How productive is MLX for writing and training models, compared with PyTorch on MPS?
2. What throughput does a single Mac Studio deliver on representative workloads?
3. Does data-parallel training across both Macs give a worthwhile speedup, and at what model and batch sizes?

Plus the headline test added for this repo: **prove the two Macs together hold, run and train a
model that is larger than either Mac's memory** (`scripts/prove_bigger_than_one.sh`).

## Agent instructions

- **Verify before you assume.** MLX moves fast. Before writing setup code, check the current versions and docs for `mlx` and `mlx.distributed` (GitHub: `ml-explore/mlx`). Treat the CLI flags, hostfile format, and backend names below as starting points to confirm. *(Done for 0.32.3; redo when bumping the pin.)*
- Pin exact versions in `pyproject.toml` / lockfile. **Both nodes must run identical Python and MLX versions.** *(`uv.lock`, `scripts/check_cluster.sh` enforces it.)*
- Keep everything reproducible: scripts, configs, and results go in the repo, and no manual one-off steps should go undocumented.
- Make single-node runs work first. Add distributed runs only after that.

## Background: what MLX provides

- `mlx.core`: NumPy-like arrays with **lazy evaluation**. Arrays live in unified memory, so there are no host/device copies.
- Function transforms: `mx.grad`, `mx.value_and_grad`, `mx.vjp`, `mx.vmap`, `mx.compile`.
- `mlx.nn`: layers, losses, `nn.value_and_grad(model, fn)`, `nn.average_gradients` (for distributed).
- `mlx.optimizers`: SGD, Adam, AdamW, schedules.
- `mlx.core.distributed`: collectives (`all_sum`, `all_gather`, `send`, `recv`, ...) with multiple backends:
  - `ring`: TCP over any network. Use it over the Thunderbolt link. Always available.
  - `jaccl`: RDMA over Thunderbolt 5 (macOS 26.2+). Much lower latency. RDMA must be enabled once per machine from Recovery mode (`rdma_ctl enable`). See `reference/rdma-jaccl.md`.
  - `mpi`, `nccl`: also supported; not used here.
- `mx.eval(...)` forces computation. Training loops must call it every step on parameters and optimizer state, or the lazy graph grows without bound.

## Phase 0: Environment setup (both nodes)

1. Install the same Python (via `uv`) and create the project environment on both Macs from the same lockfile.
2. The pinned `mlx` comes from `uv.lock`. (`mlx-lm` is an optional extra, only for the optional Phase 5.)
3. Sanity check on each node.
4. Record hardware info in `results/hardware/<hostname>.md`: chip, GPU core count, RAM, macOS version.

**Implemented as:** `./install.sh` (= `scripts/setup_node.sh` + skill install), `scripts/check_node.sh`,
`scripts/hardware_info.sh` (`mlx_cluster/nodeinfo.py`), `./tests/selfcheck.sh`.

## Phase 1: Single-node training (Mac Studio)

Training harness in `mlx_train/`:

- `models.py`: small decoder-only transformer (configurable layers, d_model, heads, seq len; RoPE, SwiGLU, RMSNorm) plus a CNN baseline.
- `data.py`: deterministic datasets with a sharding hook (`shard(rank, size)`): synthetic noisy-bigram token chains (default), byte-level LM over any text file, synthetic images for the CNN.
- `train.py`: the standard loop below, JSONL logging, `mx.compile` switch, AdamW with warmup + cosine.

Reference training step (as implemented):

```python
loss_and_grad = nn.value_and_grad(model, loss_fn)
for step, (x, y) in enumerate(batches):
    loss, grads = loss_and_grad(model, x, y)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss)  # required every step
```

Tasks:

- Log steps/sec, tokens/sec (or samples/sec), loss, and peak memory (`mx.get_peak_memory()`) to JSONL. *(`results/logs/*.jsonl` + `.summary.json`.)*
- Try `mx.compile` on the step function and record the speedup. *(`-- --no-compile` vs default; `results/single_node.md` has the table.)*
- Sweep model size (~10M, ~100M, ~500M params: `configs/10m.yaml`, `100m.yaml`, `500m.yaml`) and batch size (`--set global_batch_size=N`). Record results in `results/single_node.md` (`scripts/report.sh`).
- **Optional comparison:** `mlx_train/torch_baseline.py` ports the same model to PyTorch `mps` (`uv sync --frozen --extra torch`).

## Phase 2: Networking the two Macs

1. Connect the Macs directly with a **Thunderbolt 5 cable** (Thunderbolt 4 if the Studio lacks TB5).
2. Give the link static IPs (e.g. `10.10.10.1` / `10.10.10.2`), either with `mlx.distributed_config` or manually via Thunderbolt Bridge.
3. Set up **passwordless SSH** in both directions and confirm `ssh <other-node> hostname` works without a prompt.
4. Measure the link with `iperf3` and record the bandwidth.
5. **Optional RDMA:** if both nodes have TB5 and run macOS 26.2+, enable RDMA so you can test the `jaccl` backend.
6. `mlx.distributed_config` exists in 0.32.3 and is the preferred way to produce the hostfile.

**Implemented as:** `scripts/make_hostfile.sh`, `scripts/check_cluster.sh`, `scripts/iperf.sh`,
`scripts/run_smoke.sh` (`mlx_cluster/smoke.py`: correctness, latency, all_sum/all_gather GB/s,
send/recv ping-pong). Details: `.claude/skills/mlx-cluster/reference/networking.md`.

Hostfile (`hosts.json`, verified schema):

```json
[
  {"ssh": "studio.local",  "ips": ["10.10.10.1"]},
  {"ssh": "macbook.local", "ips": ["10.10.10.2"]}
]
```

## Phase 3: Distributed data-parallel training

As implemented in `mlx_train/train.py`:

```python
group = mx.distributed.init()          # backend chosen by mlx.launch
rank, size = group.rank(), group.size()
mx.random.seed(cfg.seed)               # identical init on every rank (+ rank-0 broadcast, checksum-verified)
dataset = dataset.shard(rank, size)    # each rank sees a disjoint slice

for step, (x, y) in enumerate(batches):
    loss, grads = loss_and_grad(model, x, y)
    grads = nn.average_gradients(grads, group)   # one bucketed all-reduce per 32 MiB
    optimizer.update(model, grads)
    loss = mx.distributed.all_sum(loss) / size
    mx.eval(model.parameters(), optimizer.state, loss)
```

Launch (verified flags; `scripts/launch.sh` adds `--cwd` and `--python` so it works over ssh):

```bash
mlx.launch --hostfile hosts.json --backend ring  --cwd $PWD --python $PWD/.venv/bin/python -- -m mlx_train.train --config configs/10m.yaml
mlx.launch --hostfile hosts.jaccl.json           --cwd $PWD --python $PWD/.venv/bin/python -- -m mlx_train.train --config configs/10m.yaml
```

Requirements (all implemented):

- Both ranks start with identical parameters: checksummed at step 0 on each rank, broadcast from rank 0, re-checked; the run aborts on mismatch.
- Only rank 0 writes logs and checkpoints.
- The **global batch size** stays constant between 1-node and 2-node runs. `rank_batch_weights` (see `configs/10m-hetero.yaml`) lets the faster node take a bigger share while the gradient is still the exact global-batch mean.

## The proof: a model larger than either Mac

`mlx_cluster/prove_bigger_than_one.py` (run via `scripts/prove_bigger_than_one.sh`):

- Reads each node's physical RAM and Metal working set, plans a decoder-only transformer whose
  total parameter bytes exceed the largest node (default: fill each node to 80% of RAM), and
  splits the layers across the nodes in proportion to their memory.
- Builds the model pipeline-parallel (rank 0: embedding + first layers; last rank: last layers +
  head + loss), measures resident memory with `mx.get_active_memory()`.
- Runs the forward pass across the link (`mx.distributed.send` / `recv_like`) and 4 pipeline-parallel
  backward + SGD steps, layer by layer with `mx.vjp` so only one layer's gradients exist at a time.
- PASS = total parameters > largest node's RAM, each node's peak within its own RAM, finite and
  decreasing loss. Report in `results/bigger_than_one/proof-<ts>.md`.

## Phase 4: Benchmark matrix

`scripts/benchmark_matrix.sh` runs each config single-node (compile on/off), 2-node ring and,
if `hosts.jaccl.json` exists, 2-node jaccl, then `scripts/report.sh` fills `results/benchmarks.md`:

| Model size | Global batch | Setup | Backend | Steps/sec | Tokens/sec | Speedup vs Studio | Peak mem/node |
|---|---|---|---|---|---|---|---|
| ~10M | 32 | Studio only | – | | | 1.0x | |
| ~10M | 32 | Studio + MacBook | ring | | | | |
| ~10M | 32 | Studio + MacBook | jaccl | | | | |
| ~100M | 16 | … | … | | | | |
| ~500M | 8 | … | … | | | | |

What to expect, and what to check:

- Gradient all-reduce runs **every step**, so the link speed dominates for small models. Plain TCP may show **no speedup or a slowdown**.
- Training runs in lockstep, so the slower node (M4 Pro) sets the pace. Expect well under 2x even in the best case. `rank_batch_weights` gives each node a different per-rank batch size; document which split you used.
- Larger models and larger per-step compute should amortize communication better.

## Phase 5: Fine-tuning check (optional)

`uv sync --frozen --extra llm` installs `mlx-lm 0.32.0`; LoRA-fine-tune a small open model on a
toy dataset, single node first (`mlx_lm.lora`), then distributed (`mlx_lm.lora` is distributed-aware).
Record tokens/sec, memory and wall-clock per epoch. Out of scope for the core deliverables.

## Deliverables

- `mlx_train/`: model, data, and training code (single-node and distributed). ✓
- `configs/`: one YAML file per benchmark config. ✓
- `scripts/`: setup checks, hostfile/iperf helpers, launch wrappers, benchmark matrix. ✓
- `mlx_cluster/`: node info, smoke test, the bigger-than-one proof. ✓
- `results/`: `hardware/`, `single_node.md`, `benchmarks.md`, `cluster/`, `bigger_than_one/`, raw JSONL logs. *(Filled by the runs on the Macs.)*
- `results/SUMMARY.md`: a short write-up answering the three goal questions, with a recommendation on whether the second Mac is worth using for training, and for which workloads. *(Template provided; fill after Phase 4.)*
