# Setup and method

|  | Mac Studio (machine under test) | MacBook Pro (second node) |
| --- | --- | --- |
| Chip | Apple M5 Max, 40 GPU cores | Apple M5 Pro, 20 GPU cores |
| Memory | 48 GB unified, 37.4 GB addressable by the GPU | 48 GB unified |
| Link | Thunderbolt 5, 80 Gb/s, RDMA: 21 µs latency, 9.3 GB/s measured | same cable |
| Software | macOS 27, Python 3.12, MLX 0.32.3, PyTorch 2.14.1 | identical |

Three workloads, each implemented in both frameworks and checked to give the same outputs for the same weights:

1. **Transformers** (decoder-only, 10M to 2B parameters, synthetic tokens): the capacity ladder for one machine and for two.
2. **The claims model**: a temporal convolutional network ([Bai et al. 2018](https://arxiv.org/abs/1803.01271)), 5 residual blocks of causal dilated convolutions, receptive field 125 bins, a dense branch for the static features, masked Huber loss on log1p monthly cost; 7.9M parameters at 512 channels, 30M at 1,024. Trained on synthetic arrays with the production schema, written by a deterministic generator so both Macs hold byte-identical copies.
3. **A model larger than one machine**: a 16B-parameter transformer split layer-wise across the two Macs (pipeline parallel over RDMA).

Every run logs every step and writes a summary; throughput excludes warm-up steps; one-Mac and two-Mac runs keep the same global batch so loss curves stay comparable. The claims arrays are synthetic and carry one hidden risk factor, so they measure throughput, memory and I/O at production shape, not predictive accuracy. Code, logs and tables: [cwkoenigs/mlx-test](https://github.com/cwkoenigs/mlx-test), `results/SUMMARY.md`.
