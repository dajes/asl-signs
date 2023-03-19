from torch import nn

from architecture.base import BaseArchitecture


class LSTMArchitecture(BaseArchitecture):
    def create_body(self, n_features: int, max_len: int, drop_rate: float = 0.1, depth: int = 12, n_heads: int = 8,
                    mult_factor: int = 4):
        return nn.GRU(n_features, n_features, depth, batch_first=True, dropout=drop_rate)

    def forward_body(self, x):
        x, _ = self.body(x)
        return x[:, -1, :]
