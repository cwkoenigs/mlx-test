# Summary

Run on 2026-10-07. Everything below comes from files in this directory; the commands that
produced them are in `.claude/skills/mlx-cluster/SKILL.md`.

**Hardware** (`hardware/`): Node A = MacBook Pro, Apple M2 Pro, 10 CPU / 16 GPU cores, 16 GiB,
Metal working set 11.8 GiB, macOS 27.2. Node B = Mac Studio, Apple M5 Max, 18 CPU / 32 GPU cores,
36 GiB, Metal working set 28.1 GiB, macOS 27.0.1. Both: Python 3.12.15, mlx 0.32.3.

**Link** (`cluster/smoke-20261007-082135.json`): direct Thunderbolt cable, negotiated at 40 Gb/s
(the M2 Pro is Thunderbolt 4, so TB5 and RDMA/`jaccl` are unavailable for this pair; `ring` over
TCP on `bridge0` link-local addresses only). MLX collectives: all_sum latency 143 µs (1 float),
all_sum 3.7 GB/s and all_gather 3.9 GB/s at 128 MiB, 1 MiB ping-pong 0.62 ms round trip.
iperf3 was not run (not installed on either Mac); the MLX numbers are what training sees.

## 1. How productive is MLX for writing and training models, compared with PyTorch on MPS?

Same transformer, same synthetic data, same optimizer and schedule, same loss curves
(both reach 1.246 on the 10M model):

| Model | MLX compiled | MLX eager | PyTorch MPS | MLX/Torch | Peak mem MLX / Torch |
|---|---|---|---|---|---|
| tf-10m (batch 32) | 6.87 steps/s, 112.5k tok/s | 5.85 steps/s | 3.71 steps/s, 60.8k tok/s | 1.85x | 6.08 / 6.13 GiB |
| tf-100m (batch 16) | 2.66 steps/s, 21.8k tok/s | 2.25 steps/s | 1.09 steps/s, 8.9k tok/s | 2.44x | 12.47 / 11.82 GiB |

- **Speed:** MLX is 1.6x faster than PyTorch on MPS before `mx.compile` and 1.9-2.4x after.
  Memory use is the same.
- **Code:** `mlx_train/train.py` (228 lines) covers single-node *and* multi-node training; the
  PyTorch port `torch_baseline.py` (167 lines) is single-device only. Model definitions are
  near line-for-line equivalent (`nn.Module`, `nn.Linear`, `nn.MultiHeadAttention`); the PyTorch
  file written without being executed ran correctly on the first try, which says how close the
  APIs are.
- **Ergonomics that differ:** MLX is lazy, so every step needs `mx.eval(params, optimizer state)`
  or the graph grows unbounded; `mx.compile` requires the step function to be pure in its
  captured state (we pass the model and optimizer as `inputs`/`outputs`). Going distributed was
  one line (`nn.average_gradients`) plus `mlx.launch`, with no process-group boilerplate.
- **Gotchas hit:** `mx.metal.device_info` deprecation (use `mx.device_info`); a Metal
  "GPU Timeout" error rather than an out-of-memory error when a node is under memory pressure
  (see Q3); `mlx.launch` reports ssh failures only as "exited with code 255".

## 2. What throughput does a single Mac Studio deliver on representative workloads?

From `single_node.md` (M5 Max, 36 GiB, float32, compiled unless noted):

| Workload | Global batch | Steps/s | Throughput | Peak memory | compile speedup |
|---|---|---|---|---|---|
| Transformer 9.8M, seq 512 | 32 | 6.87 | 112,518 tokens/s | 6.1 GiB | 1.17x |
| Transformer 99.5M, seq 512 | 16 | 2.66 | 21,793 tokens/s | 12.5 GiB | 1.16x |
| Transformer 510M, seq 512 | 8 | 1.66 | 6,805 tokens/s | 18.3 GiB | 1.28x |
| CNN 1.0M, 32x32x3 | 128 | 110.8 | 14,180 samples/s | 0.6 GiB | - |

The 510M float32 model with Adam fits in 18.3 GiB, so the Studio alone trains up to roughly
800M-1B parameters in float32 before the Metal working set (28 GiB) becomes the limit.

## 3. Does data-parallel training across both Macs give a worthwhile speedup, and at what sizes?

**Barely, and only with the batch split to match the two chips' speeds.** From `benchmarks.md`
(global batch and compile held fixed; speedup is steps/s vs the Studio alone):

| Model | Split (Studio:MacBook) | 1 node | 2 nodes ring | Speedup | Peak mem / node |
|---|---|---|---|---|---|
| tf-10m, batch 32 | 16:16 (even) | 6.59-6.87 | 3.80-4.03 | **0.58x** | 3.5 / 3.5 GiB |
| tf-10m, batch 33 | 22:11 (`rank_batch_weights: [2,1]`) | 6.62 | 5.62 | **0.85x** | 4.6 / 2.9 GiB |
| tf-10m, batch 32 | 24:8 (`rank_batch_weights: [3,1]`) | 6.56 | 7.03 | **1.07x** | 4.8 / 2.4 GiB |
| tf-100m, batch 16 | 8:8 (even) | 2.60 | 1.04 | **0.40x** | 7.4 / 7.4 GiB |
| tf-500m, batch 8 | 4:4 (even) | 1.66 | failed | - | 8.8 GiB at step 0, then Metal GPU timeout |

Why:

- **The slow node sets the pace.** Per sample the M2 Pro is roughly 3.3x slower than the M5 Max
  (from the 16:16 run: the laptop needs 0.25 s for 16 sequences, the Studio 0.15 s for 32).
  With an even split the Studio idles 60% of every step. Weighting the batch toward the Studio
  recovers it: 0.58x at 1:1, 0.85x at 2:1, 1.07x at 3:1, where the laptop handles a quarter of
  the batch and neither node waits for the other. The 3:1 run's loss curve still matches the
  1-node run (1.256 vs 1.246), as expected since the gradient is the batch-weighted mean.
  The ceiling for this pair is about 1.3x (the laptop adds ~30% more samples per unit time);
  1.07x is what remains after the all_sum and the slightly uneven step times.
- **Communication is secondary but not free.** Gradient all_sum per step is 37.5 MiB for the 10M
  model (~10 ms at 3.7 GB/s, 4% of a step) and 380 MiB for the 100M model (~100 ms, ~25% of a
  1 s step). RDMA (`jaccl`) would remove most of that, but needs TB5 on both ends.
- **Memory on the 16 GiB node fails before compute does.** The 500M run needs ~8 GiB of
  parameters + Adam state per rank plus activations; the laptop went to 7+ GB of swap and the
  Studio's GPU timed out waiting on it in the first all_sum after step 0 (0.22 steps/s on that
  step). Data-parallel replicates the whole model on every node, so the smallest node bounds the
  model size.
- **Correctness is fine.** The init check confirms identical parameters on both ranks after the
  rank-0 broadcast, and the 2-node loss curves match the 1-node curves (1.2385 vs 1.2464 at
  step 199 for tf-10m).

The 100M model was only run with an even split; a 3:1 split would lift it from 0.40x toward
~1.0x, but its 380 MiB all_sum (~25% of a step) caps it there. Meaningful speedups need a
near-equal second machine, and for 100M+ models a faster link (`jaccl`).

## The bigger-than-one-Mac proof

`bigger_than_one/proof-20261007-083932.md`: **PASS**, against physical RAM (the strict criterion).

- Decoder-only transformer, d_model 2048, 201 layers split 144 (Studio) / 57 (MacBook),
  **10.09 billion parameters = 37.6 GiB float32**, more than the Studio's 36 GiB.
- Resident after build: 26.9 GiB on the Studio, 10.7 GiB on the laptop. Peak during training:
  28.6 GiB and 11.6 GiB, both within their node's RAM.
- Forward pass across the link: finite loss 5.688. Four pipeline-parallel SGD steps: 5.688 ->
  5.309 -> 5.158 -> 5.046.
- Wall time per step 98-152 s (batch 4 x 256 tokens), dominated by both machines paging: the
  proof fills each node to 85% of RAM, which leaves macOS itself swapping (5.8 GB on the Studio,
  8 GB on the laptop during the run). The default 80% fill with 3 GiB layers could *not* exceed
  36 GiB for this pair; narrower 0.19 GiB layers shrink the per-node update reserve and made the
  physical-RAM claim reachable with 1.6 GiB to spare.

## Recommendation

- **For training speed: not worth it.** With an M2 Pro as the second node, data-parallel
  training is slower than the Studio alone at any even split, reaches 1.07x only after tuning
  `rank_batch_weights` to 3:1 on the 10M model, and fails outright at 500M parameters because
  the 16 GiB node runs out of memory. A 7% gain does not justify the cable, the second machine
  and the sleep/ssh fragility. Use the Studio by itself for anything that fits in its memory.
- **For model size: yes, with caveats.** Pipeline-parallel over the Thunderbolt link lets the
  pair hold and train a model that does not fit in either Mac (proved at 37.6 GiB / 10B
  parameters), at minutes per step. That is a capability the Studio alone does not have, useful
  for correctness checks or very small-batch fine-tuning of large models, not for throughput.
- **What would change the answer:** a second machine with 32+ GiB and Thunderbolt 5 (enables
  `jaccl` RDMA and a balanced split). Expect ~1.6-1.8x at 100M-500M parameters in that setup,
  based on the communication fractions above.

## Open items

- `tf-500m` single-node final loss differs between two otherwise identical runs (1.658 vs
  1.319 at step 99); the 10M and 100M runs are bit-for-bit repeatable. Worth a look at seeding
  or compiled-kernel nondeterminism at that size.
- The 500M 2-node row could be retried with `rank_batch_weights` skewed hard toward the Studio
  (e.g. `[3,1]`) and `--no-compile` to lower the laptop's peak, but it will still replicate
  8 GiB of state there.
- iperf3 raw TCP bandwidth was not measured (`brew install iperf3` on both; the laptop has no
  Homebrew).
- The laptop went to sleep mid-matrix once and the link dropped; keep `caffeinate -dis` running
  on it during cluster runs.
