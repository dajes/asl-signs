import torch
from torch import nn

from architecture.base import BaseArchitecture
from modeling.transformer import TransformerEncoder


class TransformerArchitecture(BaseArchitecture):
    def create_body(self, n_features: int, max_len: int, drop_rate: float = 0.1, depth: int = 12, n_heads: int = 8,
                    mult_factor: int = 4):
        return TransformerEncoder(n_features, n_heads, drop_rate, nn.SiLU, depth, mult_factor)

    def __init__(self, in_features: int, n_features: int, n_classes: int, max_len: int, drop_rate: float = 0.1,
                 depth: int = 12, n_heads: int = 8, mult_factor: int = 4):
        super().__init__(in_features, n_features, n_classes, max_len, drop_rate, depth, n_heads, mult_factor)
        self.pos_enc = nn.Parameter(torch.normal(0, .02, size=(1, max_len, n_features)))
        self.cls_token = nn.Parameter(torch.normal(0, .02, size=(1, 1, n_features)))

    def forward_body(self, x):
        x = x + self.pos_enc[:, :x.shape[1], :]
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.body(x).mean(1)
        return x
