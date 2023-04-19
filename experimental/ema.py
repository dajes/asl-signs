from copy import deepcopy

import torch
from torch import nn


class EMA(nn.Module):
    """ Model Exponential Moving Average V2 from timm"""

    def __init__(self, model, decay=0.9999, updates=0):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)

        self.updates = updates  # number of EMA updates
        self.module.eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self.updates += 1
        decay = min(self.decay, (1 + self.updates) / (10 + self.updates))
        self._update(model, update_fn=lambda e, m: decay * e + (1. - decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)
