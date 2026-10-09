# References

1. Run logs, summaries and tables for every figure above: [cwkoenigs/mlx-test](https://github.com/cwkoenigs/mlx-test) (`results/SUMMARY.md`, `results/benchmarks.md`, `results/tcn/comparison.md`, `results/bigger_than_one/`).
2. Apple MLX: [framework](https://github.com/ml-explore/mlx), [distributed communication](https://ml-explore.github.io/mlx/build/html/usage/distributed.html) and [launching distributed programs](https://ml-explore.github.io/mlx/build/html/usage/launching_distributed.html), including the `jaccl` RDMA backend used here.
3. PyTorch on Apple GPUs: [MPS backend](https://pytorch.org/docs/stable/notes/mps.html).
4. Bai, Kolter, Koltun (2018), *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*, the temporal convolutional network used for the claims model: [arXiv:1803.01271](https://arxiv.org/abs/1803.01271).
5. Huang et al. (2019), *GPipe*, pipeline parallelism across devices: [arXiv:1811.06965](https://arxiv.org/abs/1811.06965).
6. Goyal et al. (2017), *Accurate, Large Minibatch SGD*, the data-parallel and gradient-accumulation arithmetic: [arXiv:1706.02677](https://arxiv.org/abs/1706.02677).
7. Apple, [Mac Studio](https://www.apple.com/mac-studio/) technical specifications.
