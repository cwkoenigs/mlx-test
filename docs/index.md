# Apple Silicon for Claims Deep Learning

Can a machine on our desks stand in for the cloud GPUs we have struggled to obtain? Yes: one Mac Studio (M5 Max, 48 GB) trains our deep learning workloads in both frameworks we use, and it scales. Measured on 2026-10-08 on the machines on our desks, every figure comes from a committed run log in [`results/`](https://github.com/cwkoenigs/mlx-test/tree/main/results).

- **It runs our stack.** PyTorch (Apple's MPS backend) and Apple's MLX both train the same models to the same loss. MLX is 1.9x to 5.1x faster on transformers; PyTorch is faster on our claims model and on recurrent networks.
- **It handles our data shape.** The member-level claims model at production schema (863k members, 105 time bins x 324 features, a 109 GiB sequence array) trains a full epoch in 3.7 minutes on one Studio, streaming the array from disk at 0.26 GiB/s.
- **It scales.** A 2-billion-parameter transformer trains on one Studio with half its GPU memory free. A 16-billion-parameter model (60 GiB of weights) trains split across two Macs over a Thunderbolt 5 RDMA cable. Two machines add 1.6x on the claims model and 1.16x on large transformers.
- **Recommendation.** Adopt one Studio as the training machine now, validate on a real claims extract, and add a second Studio when a model exceeds 48 GB or when its 1.9x data-parallel speedup (estimated) is worth a second machine.

## The question

Our member-level claims models need GPU training, and cloud GPU capacity has been hard to get. This evaluates an alternative we can own and run ourselves: a training machine that is not a laptop and not a data centre. The arrays Denis specified are 305 GiB: 863,154 training members and 847,568 forward-test members, each a 105 x 324 float32 sequence plus 255 static features, 18 plan categoricals, 50 code families and 12 monthly cost targets. The question is whether an Apple Silicon workstation can be that machine, in our environment: our two frameworks, our data shape and size, one engineer operating it, and a path to grow. Four sub-questions were measured: does the stack run, does the data stream and fit, how large a model does one machine train, and what a second machine adds.

## Recommendation

Adopt one Mac Studio as the team's training machine. It runs both frameworks, trains the claims model at production shape in minutes per epoch, holds transformers to about 4B parameters, needs no cloud GPU allocation or data centre, and one engineer operates it from a terminal with every run logged. Scale in two stages when the work demands it: a second Studio over Thunderbolt for models above 48 GB or for an estimated 1.9x on data-parallel training, and only then cloud GPUs, if capacity has opened up.

Next steps, in order:

1. Train the claims model on a de-identified real extract on the Studio. Throughput and memory are settled; predictive value is the one open question, and it is a data-access decision rather than a compute one.
2. Measure the MLX claims model in bfloat16 to make the framework comparison exact, about ten minutes.
3. When a second Studio is available, rerun the existing ladder, matrix and proof scripts on the pair: one hour, no new code, and it replaces the 1.9x estimate with a measurement.
