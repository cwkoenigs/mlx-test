# Summary (fill in after the runs)

Hardware: see `hardware/`. Node A = MacBook Pro M4 Pro (… GB). Node B = Mac Studio (chip …, … GB).
Link: Thunderbolt …, iperf3 … Gbit/s, MLX all_sum … GB/s / … µs latency (`cluster/`).

## 1. How productive is MLX for writing and training models, compared with PyTorch on MPS?

_(code size / ergonomics of `mlx_train` vs `mlx_train/torch_baseline.py`; compile; gotchas like
`mx.eval` every step; what was hard.)_

## 2. What throughput does a single Mac Studio deliver on representative workloads?

_(table from `single_node.md`: 10M / 100M / 500M tokens/sec, CNN samples/sec, compile speedup, peak memory.)_

## 3. Does data-parallel training across both Macs give a worthwhile speedup, and at what sizes?

_(table from `benchmarks.md`; ring vs jaccl; where the speedup crosses 1.0x; effect of
`rank_batch_weights`; the slower node's share.)_

## The bigger-than-one-Mac proof

_(from `bigger_than_one/proof-*.md`: total parameters/GiB vs each node's RAM, layers per node, loss
curve over the pipeline-parallel SGD steps, seconds per step.)_

## Recommendation

_Is the second Mac worth using for training, and for which workloads?_
