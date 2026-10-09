# Reproduce the measurements

Every number in this report comes from a script in this repository and a log in `results/`.
The commands below are the ones that produced them. Run them on the Mac Studio (rank 0) unless noted.

## Set up both Macs

On **both** Macs, at the same absolute path:

```bash
git clone https://github.com/cwkoenigs/mlx-test.git ~/Developer/mlx-test && cd ~/Developer/mlx-test
./install.sh          # uv, Python 3.12, pinned mlx 0.32.3 + torch, hardware record in results/hardware/
./tests/selfcheck.sh  # ~2 min functional check; must pass before any cluster run
```

Then, from the Studio, once the Thunderbolt cable and passwordless ssh are in place:

```bash
./scripts/make_hostfile.sh studio.local macbook.local   # -> hosts.json (ring over TCP)
./scripts/check_cluster.sh                              # ssh, same path, same versions
./scripts/run_smoke.sh                                  # collectives: correctness, latency, GB/s -> results/cluster/
```

For RDMA, put the `jaccl` hostfile in `hosts.jaccl.json` (see `.claude/skills/mlx-cluster/reference/rdma-jaccl.md`).
Both hostfiles are machine-specific and git-ignored.

## Transformers: one Studio, MLX and PyTorch

```bash
for c in 10m 100m 500m 1b 2b; do ./scripts/run_train.sh configs/$c.yaml -- --tag single; done
for c in 10m 100m 500m 1b; do .venv/bin/python -m mlx_train.torch_baseline --config configs/$c.yaml; done
.venv/bin/python -m mlx_train.report        # -> results/single_node.md, results/benchmarks.md
```

## Transformers: two Macs

```bash
# One gradient exchange per step, even split (the 0.92x to 0.45x rows)
./scripts/run_train.sh configs/1b.yaml --cluster --hostfile hosts.jaccl.json -- --tag jaccl

# Gradient accumulation (16 micro-batches per exchange) and a 2:1 batch split (the 1.16x rows)
./scripts/run_train.sh configs/1b.yaml --cluster --hostfile hosts.jaccl.json -- \
  --set accumulate_steps=16 --set "rank_batch_weights=[2,1]" --tag jaccl-acc16-w21

# Studio-alone baseline with the same effective batch
./scripts/run_train.sh configs/1b.yaml -- --set accumulate_steps=8 --tag single-acc8
```

`./scripts/benchmark_matrix.sh` runs the single-node, ring and jaccl rows for a list of configs in one go.

## The claims model (TCN)

Generate the synthetic arrays once on each Mac (about 305 GiB at full scale; same path on both):

```bash
.venv/bin/python scripts/generate_synthetic_cmd_arrays.py --output-root ~/data/cmd --dry-run --full   # plan + disk check
.venv/bin/python scripts/generate_synthetic_cmd_arrays.py --output-root ~/data/cmd --full --resume
.venv/bin/python scripts/generate_synthetic_cmd_arrays.py validate --output-root ~/data/cmd
```

Then:

```bash
.venv/bin/python -m cmd_tcn.train --data-root ~/data/cmd --epochs 1 --test-forward               # PyTorch, Studio
./scripts/run_tcn.sh --data-root ~/data/cmd --epochs 1 --test-forward                            # MLX, Studio
./scripts/run_tcn.sh --cluster --data-root ~/data/cmd --epochs 1 --test-forward --rank-weights 2,1   # MLX, both Macs
.venv/bin/python -m cmd_tcn_mlx.parity --data-root ~/data/cmd    # PyTorch and MLX give the same outputs
.venv/bin/python -m cmd_tcn_mlx.report                           # -> results/tcn/comparison.md
```

Add `--channels 1024` for the 30M-parameter rows.

## A model larger than one Mac

```bash
./scripts/prove_bigger_than_one.sh --plan-only    # layers per Mac and GiB, without allocating
./scripts/prove_bigger_than_one.sh --steps 8      # -> results/bigger_than_one/proof-<ts>.{md,json}
```

Close memory-hungry apps on both Macs first. The 2026-10-08 proof used the defaults (d_model 8192,
float32, 80% fill), which planned 20 layers split 10 and 10.
