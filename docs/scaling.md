# How it scales, and where it stops

**Model size on one machine** is bounded by the 37.4 GB the GPU can address. The 2B model in bfloat16 uses 18.9 GiB, so roughly twice that size fits at the same batch; beyond it the model has to be split across machines.

**Data size** is not bounded by memory. The claims arrays are memory-mapped and read in contiguous member blocks, so the 109 GiB sequence array streams at 0.26 GiB/s while the GPU holds 6 GiB. The limit is disk throughput, and at these rates the model, not the disk, is the bottleneck.

**Machines.** Two paths, with different economics:

- *Pipeline parallel* adds memory linearly: two 48 GB Macs trained a 60 GiB model, which neither could hold. Steps are slow (11 to 14 s) because the stages run one after the other, so this is for capacity, not speed.
- *Data parallel* adds speed only when each step carries far more compute than communication. Every exchange moves 4 bytes per parameter: 4 GiB at 1B, which costs about 460 ms over the 9.3 GB/s link, as much as the Studio's own compute at these batch sizes. Accumulating 16 micro-batches per exchange cuts that to 6%, and splitting the batch 2:1 keeps the faster machine busy; that is where the 1.16x comes from. The ceiling with this pair is the MacBook's half-size GPU. Two Studios would remove it: the same arithmetic gives about 1.8x to 1.9x when each Studio carries a full batch, and about 1.9x on the claims model. That is an estimate from the measured components, not a measurement.

**Which models gain from two Macs in MLX.** The test is how much compute a step carries for each parameter it must exchange. Each exchange costs about 0.43 ms per million parameters over the link (4 bytes per parameter at 9.3 GB/s), so a model gains when its step time dwarfs that.

- *Models too large for one machine* (above about 4B parameters in bfloat16, or a 60 GiB float32 model like the 16B test): pipeline parallel. This is the only way to train them on Apple Silicon at all, but it is slow, 11 to 14 s a step. Use it for fine-tuning or occasional large runs, not iteration.
- *Convolutional sequence models with few parameters and long inputs*, like the claims TCN: data parallel, measured 1.6x with the MacBook as the second node. The 7.9M-parameter model exchanges 32 MB, about 3 ms, against roughly a second of compute per step. Two Studios: about 1.9x (estimated). Wider or deeper convolutional variants, and other members-by-time models on the same arrays, fall in this group.
- *Large transformers that fit on one machine* (about 1B to 4B parameters): a gain only with gradient accumulation and a batch split weighted to the faster Mac, measured 1.16x at 1B and 1.17x at 2B; about 1.8x to 1.9x with two Studios (estimated).

No gain, run on one machine:

- *Small or mid-size transformers trained with one exchange per step*: the pair was slower at every size, 0.92x at 10M down to 0.45x at 2B.
- *Recurrent models (LSTM, GRU)*: MLX has no fused recurrent kernel, so they stay on PyTorch, which cannot span two Macs.
- *Anything trained in PyTorch*: multi-machine work is MLX only.

**Limits found.** PyTorch cannot span two Macs at all, so multi-machine work is MLX only. MLX has no fused recurrent kernel: an LSTM written as a loop over time steps ran 1.5x to 12x slower than PyTorch's and needed far more memory, so recurrent models stay on PyTorch. The claims-model rows compare PyTorch's bf16-mixed default with MLX in float32; an MLX bfloat16 run would make that comparison exact. And the synthetic arrays say nothing about predictive value, which only a real, de-identified extract can answer.
