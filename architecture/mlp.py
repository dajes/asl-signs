from torch import nn

from architecture.base import BaseArchitecture


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.SiLU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        if not isinstance(bias, tuple):
            bias = (bias, bias)
        if not isinstance(drop, tuple):
            drop = (drop, drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """

    def __init__(
            self, dim, seq_len, mlp_ratio=(2.0, 4.0), mlp_layer=Mlp,
            act_layer=nn.SiLU, drop=0.):
        super().__init__()
        tokens_dim, channels_dim = [int(x * dim) for x in mlp_ratio]
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp_tokens = mlp_layer(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        x = x + self.mlp_channels(self.norm2(x))
        return x


class MLPArchitecture(BaseArchitecture):
    def create_body(self, n_features: int, max_len: int, drop_rate: float = 0.1, depth: int = 12, n_heads: int = 8,
                    mult_factor: int = 4):
        return nn.Sequential(
            *[MixerBlock(n_features, max_len, drop=drop_rate) for _ in range(depth)]
        )

    def forward_body(self, x):
        x = self.fixed_length(x)
        return self.body(x).mean(dim=1)
