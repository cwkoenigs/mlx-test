# Results

**One Studio trains a 2B-parameter transformer** (MLX, batch x 512 tokens per step; PyTorch MPS on the same model and data):

| Model | Parameters | Batch | MLX tokens/s | Peak memory | PyTorch tokens/s | MLX advantage |
| --- | --- | --- | --- | --- | --- | --- |
| 10M | 9.8M | 32 | 147,500 | 6.1 GiB | 77,600 | 1.9x |
| 100M | 99.5M | 16 | 28,700 | 12.5 GiB | 11,300 | 2.5x |
| 500M | 510M | 8 | 8,900 | 18.3 GiB | 2,300 | 3.8x |
| 1B (float32) | 1.06B | 4 | 4,300 | 22.4 GiB | 840 | 5.1x |
| 2B (bfloat16) | 2.05B | 4 | 3,600 | 18.9 GiB | not run | - |

**The claims model runs at production shape** (512 channels, batch 1,024 members, one epoch over 690k training members, then the 847k-member forward test):

| Setup | Members/s | Sequence array from disk | Peak GPU memory | Epoch | Forward-test loss |
| --- | --- | --- | --- | --- | --- |
| PyTorch, one Studio, bf16-mixed | 2,091 | 0.26 GiB/s | 6.1 GiB | 222 s | 1.1725 |
| MLX, one Studio, float32 | 1,114 | 0.14 GiB/s | 10.3 GiB | 396 s | 1.1735 |
| MLX, both Macs over RDMA, float32 | 1,800 | 0.23 GiB/s | 8.8 GiB | 239 s | 1.1735 |

At 1,024 channels (30M parameters): PyTorch 1,026 members/s in 11.1 GiB, MLX 468 members/s in 20.6 GiB.

!!! note "How members/s was measured in these runs"
    In the 2026-10-08 runs, the members/s and GiB/s figures for the claims model were averaged over the
    whole run: training, the end-of-epoch validation and, for the 512-channel rows, the 847k-member
    forward test. They understate pure training speed. Dividing the 690,176 trained members by each
    run's fit time (`fit_wall_s`, which includes warm-up and validation but not the forward test) gives
    PyTorch about 3,100 members/s, MLX on one Studio about 1,740 and MLX on both Macs about 2,890.
    The comparisons hold: two Macs are 1.66x one Studio in MLX (1.6x above), and PyTorch is 1.78x MLX
    (1.9x above). The epoch times in the table are `fit_wall_s`. `cmd_tcn/train.py` and
    `cmd_tcn_mlx/train.py` now stop the throughput clock before the forward test, so future runs
    report training throughput only.

**Two Macs hold what one cannot.** A 16.1B-parameter transformer, 60.0 GiB of float32 weights, trained split 10 layers per Mac: peak 34.9 GiB on each 48 GB machine, loss falling from 5.71 to 5.12 over 8 pipeline-parallel steps at 11 to 14 s per step ([report](https://github.com/cwkoenigs/mlx-test/blob/main/results/bigger_than_one/proof-20261008-193659.md)).

**Two Macs on a model that fits in one** help only with gradient accumulation and a Studio-heavy batch split: 1.16x at 1B and 1.17x at 2B against the Studio alone (16 micro-batches per gradient exchange, batch split 2:1). With one exchange per step the pair is slower at every size, 0.92x at 10M down to 0.45x at 2B.
