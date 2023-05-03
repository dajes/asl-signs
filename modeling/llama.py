# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # output = self._norm(x.float()).type_as(x)
        output = self._norm(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return torch.view_as_real(freqs_cis)


@torch.jit.export
def complex_multiply(a, b):
    return torch.stack([
        a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1],
        a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0],
    ], dim=-1)


@torch.jit.export
def apply_rotary_emb(
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq = xq.reshape(*xq.shape[:-1], -1, 2)
    xk = xk.reshape(*xk.shape[:-1], -1, 2)

    xq_out = complex_multiply(xq, freqs_cis).flatten(3)
    xk_out = complex_multiply(xk, freqs_cis).flatten(3)

    return xq_out, xk_out


class Attention(nn.Module):
    def __init__(self, dim, max_len, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.head_dim = dim // num_heads
        self.num_heads = num_heads
        self.scale = np.float32(self.head_dim ** -0.5)

        self.dropout = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.register_buffer("bias", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len))

    def forward(self, x, freqs_cis, causal=False, attn_mask=None):
        bsz, seqlen, _ = x.shape
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        xq = qkv[0]
        xk = qkv[1]
        xv = qkv[2]

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if self.flash:
            if attn_mask is not None:
                if causal:
                    attn_mask = None
                else:
                    attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            x = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=attn_mask, dropout_p=self.dropout, is_causal=causal)
        else:
            attn = (xq @ xk.transpose(-2, -1)) * self.scale
            if causal:
                attn = attn.masked_fill(self.bias[:, :, :N, :N] == 0, float('-inf'))
            elif attn_mask is not None:
                attn = attn.masked_fill(~attn_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ xv

        x = x.transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class FeedForward(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, bias=False, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.drop = nn.Dropout(drop)
        self.fc3 = nn.Linear(in_features, hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.export = False

    def forward(self, x):
        return self.drop(self.fc2(F.silu(self.fc1(x)) * self.fc3(x)))


class SwiGLUFFNFused(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, bias=False, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = (int(hidden_features * 2 / 3) + 7) // 8 * 8
        self.fc1 = nn.Linear(in_features, hidden_features * 2, bias=bias)
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.export = False

    def forward(self, x):
        out = self.fc1(x)
        if self.export:
            features = out.reshape(out.shape[0], out.shape[1], 2, -1)
            return self.drop(self.fc2(F.silu(features[:, :, 0]) * features[:, :, 1]))
        features = out.shape[-1] // 2
        return self.drop(self.fc2(F.silu(out[..., :features]) * out[..., features:]))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, n_heads, max_len, hidden_dim, mlp_ratio, norm_eps=1e-5, drop=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = hidden_dim
        self.head_dim = hidden_dim // n_heads
        self.attention = Attention(hidden_dim, max_len, n_heads, attn_drop=drop, proj_drop=drop)
        # self.feed_forward = FeedForward(in_features=hidden_dim, hidden_features=int(hidden_dim * mlp_ratio), drop=drop)
        self.feed_forward = SwiGLUFFNFused(in_features=hidden_dim, hidden_features=int(hidden_dim * mlp_ratio),
                                           drop=drop)
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(hidden_dim, eps=norm_eps)
        self.ffn_norm = RMSNorm(hidden_dim, eps=norm_eps)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, causal: bool, mask: Optional[torch.Tensor]):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cis, causal, mask)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class TransformerLlama(nn.Module):
    def __init__(self, n_features, n_heads, max_len, drop_rate, depth, mlp_ratio=4, norm_eps=1e-5):
        super().__init__()
        self.n_layers = depth

        self.layers = torch.nn.ModuleList([
            TransformerBlock(layer_id, n_heads, max_len, n_features, mlp_ratio, norm_eps, drop_rate)
            for layer_id in range(depth)
        ])

        self.norm = RMSNorm(n_features, eps=norm_eps)
        self.register_buffer("freqs_cis", precompute_freqs_cis(
            n_features // n_heads, max_len * 2
        ))

    def forward(self, x, causal=False, attn_mask=None):
        seqlen = x.shape[1]
        freqs_cis = self.freqs_cis[:seqlen]

        for layer in self.layers:
            x = layer(x, freqs_cis, causal, attn_mask)
        x = self.norm(x)
        return x
