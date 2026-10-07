# mlx 0.32.3 API facts this repo relies on (verified from the installed package)

The attached plan said "verify before you assume". These were checked against `mlx==0.32.3`
(`mlx/_distributed_utils/{launch,config,common}.py`, `mlx.core.distributed`, `mlx.nn`):

- `mx.distributed.init(strict=False, backend='any')` -> `Group` with `.rank() .size() .split()`.
  Backends: `ring`, `jaccl`, `mpi`, `nccl`, `any`. `is_available()` true on Mac and Linux-CPU.
- Collectives: `all_sum`, `all_gather`, `all_max`, `all_min`, `sum_scatter`, `send(x, dst)`,
  `recv(shape, dtype, src)`, `recv_like(x, src)`; all take `group=` and `stream=` (use
  `stream=mx.cpu` for tiny control messages). Ring backend: send/recv only between ring
  neighbours (fine for 2 nodes).
- `mlx.nn.average_gradients(gradients, group=None, all_reduce_size=32 MiB, communication_stream=None)`.
- `mlx.launch` flags: `--hosts`, `--hostfile`, `--backend`, `--env`, `--cwd`, `--python`,
  `--starting-port`, `--connections-per-ip`, `-n/--repeat-hosts`, `--mpi-arg`, `--nccl-port`.
  It builds a bash script per rank: `cd <cwd>; export MLX_RANK=<r>; export MLX_HOSTFILE=<tmp json>`
  (ring) or `MLX_JACCL_COORDINATOR` + `MLX_IBV_DEVICES` (jaccl), then `exec <python> <command>`.
  With `--python` the command must NOT start with `python` (the interpreter is prepended).
  Local runs use host `127.0.0.1`; remote runs `ssh -tt`.
- `mlx.distributed_config --hosts A,B --over thunderbolt|ethernet --backend ring|jaccl|jaccl-ring
  [--auto-setup] [--output-hostfile F] [--dot] [--ignore-unreachable] [--env K=V]`.
- Hostfile: list of `{"ssh": host, "ips": [...], "rdma": [...]}` or
  `{"backend": ..., "envs": [...], "hosts": [...]}`.
- Memory: `mx.get_active_memory() get_peak_memory() reset_peak_memory() get_memory_limit()
  set_memory_limit()`, `mx.metal.device_info()` (keys incl. `max_recommended_working_set_size`,
  `memory_size`), `mx.metal.is_available()`.
- Training: `nn.value_and_grad(model, fn)`, `nn.losses.cross_entropy(logits, targets, reduction=)`,
  `optim.AdamW(learning_rate=callable_or_float, weight_decay=)`, `optim.clip_grad_norm(grads, max)`,
  `optim.linear_schedule / cosine_decay / join_schedules`, `mx.compile(fn, inputs=state, outputs=state)`,
  `mx.vjp(fn, primals: list[array], cotangents: list[array])` (flat arrays only; we flatten
  module params with `mlx.utils.tree_flatten` / `tree_unflatten`).
- Layers: `nn.RoPE(dims, traditional=False, base=10000)`,
  `mx.fast.scaled_dot_product_attention(q, k, v, scale=, mask="causal")`, `nn.Conv2d` is NHWC.
- mlx-lm (0.32.0, optional extra `llm`) exposes `sharded_load(repo, pipeline_group, tensor_group)`
  and `--pipeline` flags in `mlx_lm.chat/server/benchmark` for pretrained models; not used by
  this repo's experiment (which trains its own models) but handy for the optional Phase 5.
