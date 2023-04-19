from typing import List

from torch import nn
from torch.nn import functional as F


class BaseArchitecture(nn.Module):
    def fixed_length(self, x):
        # uses 4D interpolation because 3D interpolation is not supported for export
        return F.interpolate(
            x.transpose(1, 2)[..., None],
            size=(self.max_len, 1), mode='nearest', align_corners=True
        )[..., 0].transpose(1, 2)

    def create_body(self, n_features: int, max_len: int, drop_rate: float = 0.1, depth: int = 12, n_heads: int = 8,
                    mult_factor: int = 4):
        raise NotImplementedError()

    def __init__(self, in_features: int, n_features: int, n_outputs: List[int], max_len: int, drop_rate: float = 0.1,
                 depth: int = 12, n_heads: int = 8, mult_factor: int = 4, causal_foresight: int = 1):
        super().__init__()
        self.in_features = in_features
        self.n_features = n_features
        self.n_classes = n_outputs
        self.max_len = max_len
        self.drop_rate = drop_rate
        self.depth = depth
        self.causal_foresight = causal_foresight

        self.in_drop = nn.Dropout(drop_rate)
        self.tail = nn.Linear(in_features, n_features)
        self.tail_norm = nn.BatchNorm1d(n_features)

        self.body = self.create_body(n_features, max_len, drop_rate, depth, n_heads, mult_factor)

        self.head_norm = nn.BatchNorm1d(n_features)
        self.head_drop = nn.Dropout(.5)
        self.heads = nn.ModuleList(
            [
                # Use big head for causal prediction to relax the constraints on the main part of model
                nn.Sequential(
                    nn.Linear(n_features, n_features),
                    nn.GELU(),
                    nn.LayerNorm(n_features),
                    nn.Linear(n_features, n_features),
                    nn.GELU(),
                    nn.LayerNorm(n_features),
                    nn.Linear(n_features, in_features * causal_foresight),
                ) if causal_foresight else nn.Identity(),
                *[
                    nn.Linear(n_features, n_classes)
                    for n_classes in n_outputs
                ]
            ]
        )

    def forward_body(self, x, causal=False, attn_mask=None):
        return self.body(x, causal=causal, attn_mask=attn_mask)

    def forward(self, x, task_n=1, attn_mask=None):
        x = self.tail(x)
        x = self.tail_norm(x.transpose(1, 2)).transpose(1, 2)
        x = self.forward_body(x, causal=task_n == 0, attn_mask=attn_mask)
        if len(x.shape) == 3:
            x = self.head_norm(x.transpose(1, 2)).transpose(1, 2)
        else:
            x = self.head_norm(x)
        x = self.head_drop(x)
        if self.training and task_n:
            result = []
            for head in self.heads[1:]:
                result.append(head(x))
            x = result
        else:
            x = self.heads[task_n](x)
            if not task_n:
                x = x.reshape(x.shape[0], x.shape[1], self.causal_foresight, -1)
        return x
