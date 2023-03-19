from torch import nn

from architecture.base import BaseArchitecture


class LinearArchitecture(BaseArchitecture):
    def create_body(self, n_features: int, max_len: int, drop_rate: float = 0.1, depth: int = 12, n_heads: int = 8,
                    mult_factor: int = 4):
        return nn.Identity()

    def forward_body(self, x):
        return self.body(x[:, -1, :])
