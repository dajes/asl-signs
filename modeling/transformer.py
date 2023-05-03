import math
import numpy as np
import torch
from torch import nn

from modeling.llama import RMSNorm


class Attention(nn.Module):
    def __init__(self, dim, max_len, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = np.float32(self.head_dim ** -0.5)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.dropout = attn_drop
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        self.register_buffer("bias", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len))

    def forward(self, x, causal=False, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        q = qkv[0]
        k = qkv[1]
        v = qkv[2]

        if self.flash:
            if attn_mask is not None:
                if causal:
                    attn_mask = None
                else:
                    attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.dropout, is_causal=causal)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if causal:
                attn = attn.masked_fill(self.bias[:, :, :N, :N] == 0, float('-inf'))
            elif attn_mask is not None:
                attn = attn.masked_fill(~attn_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if not isinstance(bias, tuple):
            bias = (bias, bias)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.export = False

    def forward(self, x):
        x = self.fc1(x)
        if self.export:
            x = gelu(x)
        else:
            # a lot less memory consumption and faster training
            x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):

    def __init__(
            self,
            dim,
            num_heads,
            max_len,
            mlp_ratio=4.,
            qkv_bias=False,
            drop=0.,
            attn_drop=0.,
            norm_layer=RMSNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, max_len, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x, causal=False, attn_mask=None):
        x = x + self.attn(self.norm1(x), causal=causal, attn_mask=attn_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, n_features, n_heads, max_len, drop_rate, depth, mlp_ratio=4,
                 qkv_bias=False):
        super().__init__()
        self.n_features = n_features
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.Sequential(*[
            Block(
                dim=n_features,
                num_heads=n_heads,
                max_len=max_len,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=drop_rate,
                norm_layer=RMSNorm,
            )
            for _ in range(depth)
        ])

    def forward(self, x, causal=False, attn_mask=None):
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x, causal, attn_mask)
        return x
