"""Temporal convolutional network (Bai, Kolter & Koltun 2018) for the CMD sequences.

Five residual blocks with dilations 1, 2, 4, 8, 16; each block has two causal dilated Conv1d
layers (kernel 3, weight-normalised, ReLU, dropout) and a 1x1 residual projection when the channel
count changes. Receptive field = 1 + 2 * (k-1) * sum(dilations) = 125 time bins with k=3, so the
last output step sees the whole 105-bin history. The last step is concatenated with a projection
of the static features and mapped to the 12 monthly outputs.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

DEFAULT_DILATIONS = (1, 2, 4, 8, 16)


class CausalConv1d(nn.Module):
    """Conv1d padded on the left only, so output[t] depends on input[<= t]."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int):
        super().__init__()
        self.pad = (kernel - 1) * dilation
        self.conv = weight_norm(nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, C, T]
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_ch, out_ch, kernel, dilation), nn.ReLU(), nn.Dropout(dropout),
            CausalConv1d(out_ch, out_ch, kernel, dilation), nn.ReLU(), nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + self.downsample(x))


class TCN(nn.Module):
    def __init__(self, in_ch: int, channels: int, kernel: int = 3,
                 dilations: tuple[int, ...] = DEFAULT_DILATIONS, dropout: float = 0.1):
        super().__init__()
        blocks, c = [], in_ch
        for d in dilations:
            blocks.append(TemporalBlock(c, channels, kernel, d, dropout))
            c = channels
        self.blocks = nn.Sequential(*blocks)
        self.receptive_field = 1 + 2 * (kernel - 1) * sum(dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, C, T] -> [B, channels, T]
        return self.blocks(x)


class CMDTCNRegressor(nn.Module):
    """X_seq [B,105,324] + enrollment_mask [B,105] + X_static [B,255] -> 12 monthly cost predictions (log1p space)."""

    def __init__(self, n_seq_features: int = 324, n_static: int = 255, out_len: int = 12,
                 channels: int = 128, kernel: int = 3, dilations: tuple[int, ...] = DEFAULT_DILATIONS,
                 dropout: float = 0.1, static_dim: int = 64, head_hidden: int = 256):
        super().__init__()
        self.tcn = TCN(n_seq_features + 1, channels, kernel, dilations, dropout)  # +1: enrollment mask channel
        self.static = nn.Sequential(nn.Linear(n_static, static_dim), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Sequential(
            nn.Linear(channels + static_dim, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, out_len),
        )

    def forward(self, x_seq: torch.Tensor, enroll: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_seq, enroll.unsqueeze(-1)], dim=-1).transpose(1, 2)  # [B, 325, 105]
        h = self.tcn(x)[:, :, -1]                                             # last (causal) step: [B, channels]
        return self.head(torch.cat([h, self.static(x_static)], dim=-1))     # [B, 12]


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
