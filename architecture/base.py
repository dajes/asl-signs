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

    def __init__(self, in_features: int, n_features: int, n_classes: int, max_len: int, drop_rate: float = 0.1,
                 depth: int = 12, n_heads: int = 8, mult_factor: int = 4):
        super().__init__()
        self.in_features = in_features
        self.n_features = n_features
        self.n_classes = n_classes
        self.max_len = max_len
        self.drop_rate = drop_rate
        self.depth = depth

        self.in_drop = nn.Dropout(drop_rate)
        self.tail = nn.Linear(in_features, n_features)
        self.tail_norm = nn.LayerNorm(n_features)

        self.body = self.create_body(n_features, max_len, drop_rate, depth, n_heads, mult_factor)

        self.head_norm = nn.LayerNorm(n_features)
        self.head = nn.Linear(n_features, n_classes)
        # self.head.bias.data[:] = math.log(5 / n_classes)

    def forward_body(self, x):
        return self.body(x)

    def forward(self, x):
        x = self.in_drop(x)
        x = self.tail(x)
        x = self.tail_norm(x)
        x = self.forward_body(x)
        x = self.head_norm(x)
        x = self.head(x)
        return x
