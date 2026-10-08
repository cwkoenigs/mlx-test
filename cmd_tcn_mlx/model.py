"""Temporal convolutional network for the CMD sequences, in MLX. Layer-for-layer the same as
``cmd_tcn/model.py`` (PyTorch): five residual blocks, dilations 1/2/4/8/16, two weight-normalised
causal dilated Conv1d per block (kernel 3, ReLU, dropout), 1x1 residual projection when the channel
count changes, receptive field 125 bins; last causal step concatenated with a projection of the
static features and mapped to the 12 monthly log1p-cost outputs. ``cmd_tcn_mlx.parity`` checks
that both frameworks produce the same outputs for the same weights.

MLX convolutions are channels-last (NLC), so X_seq [B, 105, 324] is used as-is.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

DEFAULT_DILATIONS = (1, 2, 4, 8, 16)


class WNCausalConv1d(nn.Module):
    """Causal dilated Conv1d with weight normalisation: w = g * v / ||v|| (norm over kernel x in)."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.dilation = dilation
        scale = 1.0 / math.sqrt(in_ch * kernel)
        self.v = mx.random.uniform(-scale, scale, (out_ch, kernel, in_ch))
        self.g = mx.sqrt(mx.sum(self.v * self.v, axis=(1, 2)))  # initial effective weight == v (as torch's weight_norm)
        self.bias = mx.random.uniform(-scale, scale, (out_ch,))

    def weight(self) -> mx.array:
        norm = mx.sqrt(mx.sum(self.v * self.v, axis=(1, 2), keepdims=True))
        return self.g[:, None, None] * self.v / norm

    def __call__(self, x: mx.array) -> mx.array:  # [B, T, C_in] -> [B, T, C_out]
        x = mx.pad(x, [(0, 0), (self.pad, 0), (0, 0)])
        return mx.conv1d(x, self.weight(), dilation=self.dilation) + self.bias


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = WNCausalConv1d(in_ch, out_ch, kernel, dilation)
        self.drop1 = nn.Dropout(dropout)
        self.conv2 = WNCausalConv1d(out_ch, out_ch, kernel, dilation)
        self.drop2 = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def __call__(self, x: mx.array) -> mx.array:
        h = self.drop1(nn.relu(self.conv1(x)))
        h = self.drop2(nn.relu(self.conv2(h)))
        res = x if self.downsample is None else self.downsample(x)
        return nn.relu(h + res)


class TCN(nn.Module):
    def __init__(self, in_ch: int, channels: int, kernel: int = 3,
                 dilations: tuple[int, ...] = DEFAULT_DILATIONS, dropout: float = 0.1):
        super().__init__()
        blocks, c = [], in_ch
        for d in dilations:
            blocks.append(TemporalBlock(c, channels, kernel, d, dropout))
            c = channels
        self.blocks = blocks
        self.receptive_field = 1 + 2 * (kernel - 1) * sum(dilations)

    def __call__(self, x: mx.array) -> mx.array:  # [B, T, C] -> [B, T, channels]
        for b in self.blocks:
            x = b(x)
        return x


class CMDTCNRegressor(nn.Module):
    """X_seq [B,105,324] + enrollment_mask [B,105] + X_static [B,255] -> 12 monthly cost predictions (log1p space)."""

    def __init__(self, n_seq_features: int = 324, n_static: int = 255, out_len: int = 12,
                 channels: int = 128, kernel: int = 3, dilations: tuple[int, ...] = DEFAULT_DILATIONS,
                 dropout: float = 0.1, static_dim: int = 64, head_hidden: int = 256):
        super().__init__()
        self.tcn = TCN(n_seq_features + 1, channels, kernel, tuple(dilations), dropout)  # +1: enrollment mask channel
        self.static_proj = nn.Linear(n_static, static_dim)
        self.static_drop = nn.Dropout(dropout)
        self.head1 = nn.Linear(channels + static_dim, head_hidden)
        self.head_drop = nn.Dropout(dropout)
        self.head2 = nn.Linear(head_hidden, out_len)

    def __call__(self, x_seq: mx.array, enroll: mx.array, x_static: mx.array) -> mx.array:
        x = mx.concatenate([x_seq, enroll[..., None]], axis=-1)        # [B, 105, 325]
        h = self.tcn(x)[:, -1, :]                                        # last (causal) step: [B, channels]
        s = self.static_drop(nn.relu(self.static_proj(x_static)))
        z = self.head_drop(nn.relu(self.head1(mx.concatenate([h, s], axis=-1))))
        return self.head2(z)                                             # [B, 12]


def count_params(m: nn.Module) -> int:
    return sum(v.size for _, v in tree_flatten(m.parameters()))


def masked_huber(pred: mx.array, y: mx.array, mask: mx.array, delta: float = 1.0):
    """Masked Huber regression on log1p(monthly allowed cost); returns (loss, mae_log1p, mae_usd)."""
    target = mx.log1p(mx.maximum(y, 0.0))
    denom = mx.maximum(mask.sum(), 1.0)
    loss = (nn.losses.huber_loss(pred, target, delta, reduction="none") * mask).sum() / denom
    mae_log = (mx.abs(pred - target) * mask).sum() / denom
    mae_usd = (mx.abs(mx.expm1(pred) - y) * mask).sum() / denom
    return loss, mae_log, mae_usd
