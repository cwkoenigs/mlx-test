# Summary (2026-10-08, current pair of Macs)

Everything below comes from files in this directory, measured on 2026-10-08. The previous pair's
write-up (M2 Pro laptop + 36 GB Studio, 2026-10-07) is `SUMMARY-2026-10-07-previous-pair.md`.

**Hardware** (`hardware/`): Mac Studio, Apple M5 Max, 40 GPU cores, 48 GB (37.4 GB usable by the
GPU). MacBook Pro, Apple M5 Pro, 20 GPU cores, 48 GB. Direct Thunderbolt 5 cable at 80 Gb/s with RDMA.
Both macOS 27, Python 3.12.15, mlx 0.32.3, torch 2.14.1.

**Link** (`cluster/smoke-20261008-*.json`): MLX collectives over RDMA (`jaccl`) 21 µs all_sum latency,
9.3 GB/s at 128 MiB; over TCP (`ring`) 139 µs, 6.6 GB/s.

## 1. How productive is MLX compared with PyTorch on MPS?

Same decoder-only transformer, same synthetic tokens, same optimizer and schedule, same loss curves
(both reach 1.25 at 10M and 100M). Studio alone (`benchmarks.md`, `logs/`):

| Model | Batch | MLX (compiled) | PyTorch MPS | MLX / PyTorch |
|---|---|---|---|---|
| 10M | 32 | 9.00 steps/s, 147.5k tok/s | 4.74 steps/s, 77.6k tok/s | 1.9x |
| 100M | 16 | 3.51 steps/s, 28.7k tok/s | 1.38 steps/s, 11.3k tok/s | 2.5x |
| 500M | 8 | 2.17 steps/s, 8.9k tok/s | 0.57 steps/s, 2.3k tok/s | 3.8x |
| 1B | 4 | 2.08 steps/s, 4.3k tok/s | 0.41 steps/s, 0.8k tok/s | 5.1x |

MLX's lead grows with model size. Caveat from a separate experiment the same day (a recurrent
LSTM language model, repo since removed): for recurrent models PyTorch's fused MPS LSTM kernel was
1.5x to 12x faster than an MLX LSTM written as a Python loop over time steps, and MLX's unrolled graph
needed far more memory. MLX wins on transformers and convolutions; recurrence needs a fused kernel.

**The claims workload** (`tcn/comparison.md`): the CMD TCN, 512 channels, 7.9M parameters, batch
1,024 members, one epoch over 690k training members plus the 847k-member forward test.

| Setup | Members/s | X_seq from disk | Peak GPU mem | Epoch (fit) | Forward-test loss |
|---|---|---|---|---|---|
| PyTorch, Studio, bf16-mixed | 2,091 | 0.26 GiB/s | 6.1 GiB | 222 s | 1.1725 |
| MLX, Studio, float32 | 1,114 | 0.14 GiB/s | 10.3 GiB | 396 s | 1.1735 |
| MLX, both Macs over jaccl, float32, batch split 2:1 | 1,800 | 0.23 GiB/s | 8.8 GiB | 239 s | 1.1735 |

At 1,024 channels (30M parameters, 100 steps): PyTorch 1,026 members/s at 11.1 GiB, MLX 468 members/s
at 20.6 GiB. PyTorch's rows ran its default bf16-mixed precision, MLX's float32; the two frameworks
reach the same loss. An MLX bfloat16 TCN run (`--dtype bfloat16`) was not measured.

## 2. What does one Mac Studio deliver?

Transformers in MLX, Studio alone (`single_node.md`):

| Model | Parameters | Batch x 512 tokens | Steps/s | Tokens/s | Peak memory |
|---|---|---|---|---|---|
| 10M | 9.8M | 32 | 9.00 | 147,500 | 6.1 GiB |
| 100M | 99.5M | 16 | 3.51 | 28,700 | 12.5 GiB |
| 500M | 510M | 8 | 2.17 | 8,900 | 18.3 GiB |
| 1B (float32) | 1.06B | 4 | 2.08 | 4,300 | 22.4 GiB |
| 2B (bfloat16) | 2.05B | 4 | 1.76 | 3,600 | 18.9 GiB |

A 2B-parameter transformer trains on one Studio with half of the GPU's memory to spare; the 1B and
2B rows are the first ever run in this repo. The Studio is 1.3x the previous 32-core M5 Max at every
size from 10M to 500M. For the claims workload, PyTorch bf16 on the Studio processes the full 863k
training members in about 7 minutes per epoch at 512 channels and about 14 minutes at 1,024.

## 3. Is the second Mac worth it?

**For data-parallel training of these transformers with one exchange per batch: no.** Two Macs over
RDMA, global batch kept equal to the single-Studio run and split evenly (`benchmarks.md`):

| Model | Studio alone | Both Macs, jaccl | Both Macs, ring | Speedup (jaccl) |
|---|---|---|---|---|
| 10M | 9.00 steps/s | 8.27 | 7.88 | 0.92x |
| 100M | 3.51 | 2.90 | 2.50 | 0.83x |
| 500M | 2.17 | 1.27 | - | 0.59x |
| 1B | 2.08 | 0.87 | - | 0.42x |
| 2B | 1.76 | 0.78 | - | 0.45x |

Two effects: the MacBook has half the GPU cores, so with an even split it sets the pace; and every
step averages the full gradient (4 bytes per parameter: 4 GiB at 1B), which at 9.3 GB/s costs about
as much as the Studio's own compute at these batch sizes. Giving the Studio two thirds of the batch
(the 510M run with `rank_batch_weights`, batch 12) reaches 7,757 tok/s, still below the Studio's own
8,892. Data parallelism would need much larger per-step compute than these configs carry (bigger
batches or sequences, or gradient accumulation between exchanges) before the second Mac helps.

**With gradient accumulation and a Studio-heavy split: yes, by about 16%.** The exchange cost above
is per optimizer update, so `accumulate_steps` (added 2026-10-08 evening) sums k micro-batches locally
and exchanges once. Same effective batch for every row (global batch x k), tokens/s over 96 micro-steps
(`benchmarks.md`, tags `single-acc8`, `jaccl-acc8`, `jaccl-acc8-w21`, `jaccl-acc16-w21`):

| Model | Studio alone, k=8 | Both, even split, k=8 | Both, split 2:1, k=8 | Both, split 2:1, k=16 |
|---|---|---|---|---|
| 500M | 9,240 tok/s | 8,296 (0.90x) | 10,718 (1.16x) | - |
| 1B | 4,657 | 3,468 (0.74x) | 4,830 (1.04x) | 5,404 (1.16x) |
| 2B (bf16) | 3,887 | 2,914 (0.75x) | 4,119 (1.06x) | 4,558 (1.17x) |

Both levers are needed: accumulation alone (even split) still loses because the MacBook sets the pace;
the 2:1 split alone (the 510M row above) still loses to the exchange. Together the second Mac adds
16-17% at 1B-2B. The ceiling is the MacBook's half-size GPU: ideal would be 1.5x, and the rest is lost
to the remaining exchange and the lockstep. (With k=16 only 6 optimizer updates happen in 96 micro-steps,
so those rows' losses barely move; they are throughput measurements.)

**For the claims TCN: yes, modestly.** Small gradients (32 MB) and a large batch mean the exchange is
cheap: 1,800 members/s on both Macs against 1,114 on the Studio alone in MLX, a 1.6x speedup. PyTorch
cannot span the two Macs at all.

**For models that do not fit in one Mac: yes, and that is the headline.** `bigger_than_one/proof-20261008-193659.md`:
a 16.1-billion-parameter transformer, 60.0 GiB of float32 weights, split 10 layers per Mac, peak
34.9 GiB on each 48 GB Mac, forward loss finite, loss falling from 5.71 to 5.12 over 8 pipeline-parallel
SGD steps at 11 to 14 s per step. A 4-step attempt the same evening (`proof-20261008-191243.md`) passed
every memory check and failed only the "loss decreases" criterion through SGD noise; both reports are kept.

## Recommendation

Train on the Studio alone for anything that fits, which is at least 2B parameters in bfloat16 and the
claims TCN at 1,024 channels; use MLX for transformers and PyTorch for the TCN (its bf16 path) and for
recurrent models. Use both Macs for models larger than one Mac's memory (pipeline parallel over RDMA,
demonstrated at 60 GiB), for data-parallel workloads with small gradients and large batches such as the
TCN (1.6x), and for large transformers only with gradient accumulation and a 2:1 batch split (about 1.16x
at 1B-2B; not worth it below that). The link is not the problem: ring and jaccl are within 10% of each other here; the
all-reduce volume and the slower second GPU are.
