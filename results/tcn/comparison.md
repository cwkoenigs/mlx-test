# CMD TCN: PyTorch vs MLX, one Mac vs two

| Run | Framework | Macs | Backend | Channels | Batch | Precision | Params | Steps/s | Members/s | X_seq GiB/s | Peak mem | Val loss | Val MAE $ | Fwd-test loss | Fwd-test MAE $ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tcn-mlx-studio-1n-single-20261008-183450 | mlx | 1 (Clarks-Mac-Studio.local) | single | 512 | 1024 | float32 | 7,921,420 | 1.09 | 1,114 | 0.14 | 10.3 GiB | 1.1882 | 485 | 1.1735 | 479 |
| tcn-mlx-studio-1024ch-1n-single-20261008-185044 | mlx | 1 (Clarks-Mac-Studio.local) | single | 1024 | 1024 | float32 | 29,962,508 | 0.46 | 468 | 0.06 | 20.6 GiB | 1.2082 | 492 | - | - |
| tcn-mlx-2n-jaccl-20261008-085606 | mlx | 2 (Clarks-Mac-Studio.local, Clarks-MacBook-Pro.local) | jaccl | 512 | 1024 | float32 | 7,921,420 | 1.76 | 1,800 | 0.23 | 8.8 GiB | 1.1778 | 478 | 1.1735 | 479 |
| tcn-torch-studio-20261008-142525 | torch-lightning | 1 (Clarks-Mac-Studio.local) | - | 512 | 1024 | bf16-mixed | 7,921,420 | 0.92 | 943 | 0.12 | 6.1 GiB | - | - | - | - |
| tcn-torch-studio-20261008-184510 | torch-lightning | 1 (Clarks-Mac-Studio.local) | - | 512 | 1024 | bf16-mixed | 7,921,420 | 2.04 | 2,091 | 0.26 | 6.1 GiB | 1.1868 | 485 | 1.1725 | 479 |
| tcn-torch-studio-1024ch-20261008-185424 | torch-lightning | 1 (Clarks-Mac-Studio.local) | - | 1024 | 1024 | bf16-mixed | 29,962,508 | 1.00 | 1,026 | 0.13 | 11.1 GiB | - | - | - | - |
