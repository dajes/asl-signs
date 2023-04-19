from typing import List

import torch
from torch import nn

from architecture.base import BaseArchitecture
from modeling.transformer import TransformerEncoder


class TransformerArchitecture(BaseArchitecture):
    def create_body(self, n_features: int, max_len: int, drop_rate: float = 0.1, depth: int = 12, n_heads: int = 8,
                    mult_factor: int = 4):
        return TransformerEncoder(n_features, n_heads, max_len, drop_rate, depth, mult_factor)

    def __init__(self, in_features: int, n_features: int, n_outputs: List[int], max_len: int, drop_rate: float = 0.1,
                 depth: int = 12, n_heads: int = 8, mult_factor: int = 4, causal_foresight: int = 1):
        super().__init__(in_features, n_features, n_outputs, max_len, drop_rate, depth, n_heads, mult_factor,
                         causal_foresight)
        self.pos_enc = nn.Embedding(max_len, n_features)
        self.cls_enc = nn.Embedding(1, n_features)

    def forward_body(self, x, causal=False, attn_mask=None):
        x = x + self.pos_enc.weight[:x.shape[1], :].unsqueeze(0)
        if not causal:
            x = torch.cat((self.cls_enc.weight.repeat(x.shape[0], 1).unsqueeze(1), x), dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat((attn_mask.new_ones((len(attn_mask), 1)), attn_mask), dim=1)
        x = self.body(x, causal=causal, attn_mask=attn_mask)
        if not causal:
            x = x[:, 0]
        return x
