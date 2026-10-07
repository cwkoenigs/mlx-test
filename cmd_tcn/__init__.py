"""PyTorch Lightning temporal convolutional network on the synthetic CMD arrays.

Predicts ``y`` (12 months of allowed cost) from ``X_seq`` (105 time bins x 324 features) plus the
historical ``enrollment_mask`` and ``X_static``. Streams the arrays from disk with memory maps in
contiguous member blocks, so the 305 GiB full-scale dataset trains on a 36 GB Mac.

    .venv/bin/python -m cmd_tcn.train --data-root ~/data/cmd --max-steps 200
"""
