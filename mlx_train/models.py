"""Models: a small decoder-only transformer (configurable) and a CNN baseline.

The same ``Block`` is reused by ``mlx_cluster.prove_bigger_than_one`` to build a
pipeline-parallel model that spans both Macs.

Verified against mlx 0.32.3: nn.RoPE, mx.fast.scaled_dot_product_attention(mask="causal"),
nn.RMSNorm, nn.Conv2d (NHWC), mlx.utils.tree_flatten.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


def ffn_dim(d_model: int, d_ff: int | None = None) -> int:
    """SwiGLU hidden size: 8/3 * d_model rounded to a multiple of 64 (as in Llama)."""
    if d_ff:
        return d_ff
    return max(64, int(round(8 * d_model / 3 / 64)) * 64)


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.wqkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=10000.0)

    def __call__(self, x: mx.array) -> mx.array:
        B, L, _ = x.shape
        q, k, v = mx.split(self.wqkv(x), 3, axis=-1)
        q = q.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(B, L, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        q = self.rope(q)
        k = self.rope(k)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal")
        return self.wo(o.transpose(0, 2, 1, 3).reshape(B, L, -1))


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    """Pre-norm transformer block: x + attn(norm(x)); x + mlp(norm(x))."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int | None = None):
        super().__init__()
        self.attn_norm = nn.RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        self.mlp_norm = nn.RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, ffn_dim(d_model, d_ff))

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.attn_norm(x))
        return x + self.mlp(self.mlp_norm(x))


class DecoderTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, d_ff: int | None = None):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = [Block(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.norm = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def __call__(self, tokens: mx.array) -> mx.array:
        h = self.embed(tokens)
        for layer in self.layers:
            h = layer(h)
        return self.head(self.norm(h))


class SimpleCNN(nn.Module):
    """Small conv net for the synthetic image task. Input is NHWC."""

    def __init__(self, channels: int, width: int, n_classes: int):
        super().__init__()
        self.c1 = nn.Conv2d(channels, width, 3, padding=1)
        self.c2 = nn.Conv2d(width, 2 * width, 3, stride=2, padding=1)
        self.c3 = nn.Conv2d(2 * width, 4 * width, 3, stride=2, padding=1)
        self.c4 = nn.Conv2d(4 * width, 4 * width, 3, padding=1)
        self.fc = nn.Linear(4 * width, n_classes)

    def __call__(self, x: mx.array) -> mx.array:
        x = nn.relu(self.c1(x))
        x = nn.relu(self.c2(x))
        x = nn.relu(self.c3(x))
        x = nn.relu(self.c4(x))
        return self.fc(x.mean(axis=(1, 2)))


def build_model(cfg) -> nn.Module:
    """cfg is an ``mlx_train.config.ModelConfig``."""
    if cfg.type == "transformer":
        return DecoderTransformer(cfg.vocab_size, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.d_ff)
    if cfg.type == "cnn":
        return SimpleCNN(cfg.channels, cfg.width, cfg.n_classes)
    raise ValueError(f"unknown model type {cfg.type!r}")


def count_params(model: nn.Module) -> int:
    return sum(v.size for _, v in tree_flatten(model.parameters()))


def param_bytes(model: nn.Module) -> int:
    return sum(v.nbytes for _, v in tree_flatten(model.parameters()))


def param_checksum(model: nn.Module) -> float:
    """Deterministic scalar fingerprint of all parameters (used to verify identical init across ranks)."""
    parts = [mx.sum(v.astype(mx.float32)) for _, v in tree_flatten(model.parameters())]
    s = mx.sum(mx.stack(parts))
    mx.eval(s)
    return float(s.item())


def transformer_param_count(vocab_size: int, d_model: int, n_layers: int, d_ff: int | None = None) -> int:
    """Analytic parameter count matching ``DecoderTransformer`` (no need to build it)."""
    return block_param_count(d_model, d_ff) * n_layers + 2 * vocab_size * d_model + d_model


def block_param_count(d_model: int, d_ff: int | None = None) -> int:
    ff = ffn_dim(d_model, d_ff)
    return 4 * d_model * d_model + 3 * d_model * ff + 2 * d_model
