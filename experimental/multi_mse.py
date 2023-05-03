from typing import List

import torch
from torch import nn


class MultiMSELoss(nn.MSELoss):
    def forward(self, preds: List[torch.Tensor], target: torch.Tensor, ds_nums: torch.Tensor):
        if self.reduction == 'none':
            loss = torch.zeros(target.shape[0], device=target.device)
        else:
            loss = 0
        for ds_num in torch.unique(ds_nums):
            mask = ds_nums == ds_num
            y_hat = preds[ds_num - 1][mask]
            y = target[mask]
            if y.dtype == torch.long:
                assert y.max() < y_hat.shape[-1]
                assert y.min() >= 0
                one_hot = torch.zeros_like(y_hat)
                one_hot.scatter_(1, y.unsqueeze(1), 1)
                y = one_hot
            if self.reduction == 'none':
                loss[mask] += super().forward(y_hat, y).sum(-1)
            else:
                loss += super().forward(y_hat, y)
        return loss / target.shape[0]
