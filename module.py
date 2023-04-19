import os
import warnings
from typing import List

import numpy as np
import pytorch_lightning as pl
import torch

import constants
from architecture.linear import LinearArchitecture
from architecture.lstm import LSTMArchitecture
from architecture.mlp import MLPArchitecture
from architecture.transformer import TransformerArchitecture
from experimental.ema import EMA
from experimental.lion import Lion
from experimental.mixup import Mixup
from experimental.multi_crossentropy import MultiCrossentropyLoss
from state import state

for warn_pattern in [
    r".*Your compiler for AOTAutograd is returning a a function that doesn't take boxed\.*",
    r".*Trying to infer the `batch_size` from an ambiguous collection.*",
    r".*The `batch_size` could not be inferred from the dataloader.*",
    r".*Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\).*",
    r'.*Please use DTensor instead and we are deprecating ShardedTensor.*',
    r'.*nn\.Module hooks are not fully supported, they may be ignored.*',
]:
    warnings.filterwarnings(
        "ignore", warn_pattern
    )
state['silent'] = os.environ.get('LOCAL_RANK', '0') != '0'
state['master'] = os.environ.get('LOCAL_RANK', '0') == '0'


class Module(pl.LightningModule):

    def __init__(
            self, working_dir: str, epochs: int, steps_per_epoch: int, lr: float,
            in_features: int, n_features: int, n_outputs: List[int], max_len: int, drop_rate: float = 0.1,
            depth: int = 6, num_heads: int = 8, mlp_ratio=4.,
            model_type: str = 'transformer', causal_foresight: int = 1, use_ema: bool = True,
            label_smoothing: float = 0., mixup: float = .2, n_coords: int = 2
    ):
        super().__init__()
        self.working_dir = working_dir
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.lr = lr
        self.causal_foresight = causal_foresight
        self.n_coords = n_coords
        self.teacher = None
        self.save_hyperparameters()

        self.model = {
            'linear': LinearArchitecture,
            'mlp': MLPArchitecture,
            'lstm': LSTMArchitecture,
            'transformer': TransformerArchitecture,
        }[model_type](in_features, n_features, n_outputs, max_len, drop_rate, depth, num_heads, mlp_ratio,
                      causal_foresight)
        self.model_ema = EMA(self.model) if use_ema else None
        self.criterion = MultiCrossentropyLoss(reduction='sum', label_smoothing=label_smoothing)
        self.regression = torch.nn.SmoothL1Loss(reduction='sum') if self.causal_foresight > 0 else None
        self.mixup = Mixup(self.n_coords, mixup)

    def forward(self, x, task_n, attn_mask):
        if not self.training and self.model_ema is not None:
            return self.model_ema.module(x, task_n, attn_mask)
        return self.model(x, task_n, attn_mask)

    def apply_criterion(self, y_hats, y, teacher_y, ds_nums, lam, index):
        loss = self.mixup.apply_criterion(self.criterion, y_hats, teacher_y, ds_nums, lam, index)
        acc = 0
        for ds_num in torch.unique(ds_nums):
            mask = ds_nums == ds_num
            y_hat = y_hats[ds_num - 1][mask]
            target = y[mask]
            with torch.no_grad():
                acc += (y_hat.argmax(dim=-1) == target).float().sum()
        acc /= len(y)
        return loss, acc

    def forward_loss(self, x, attn_mask, y, causal, ds_nums):
        x, attn_mask, lam, index = self.mixup(x, attn_mask, self.training and not causal)
        y_hat = self(x, int(not causal), attn_mask)

        if self.training:
            y_hats = y_hat
        else:
            y_hats = [y_hat]

        if self.teacher is not None:
            self.teacher.eval()
            with torch.no_grad():
                teacher_y = self.teacher(x, int(not causal), attn_mask).softmax(-1)
            categorical_teacher = teacher_y.argmax(-1)
            loss, acc = self.apply_criterion(
                y_hats, categorical_teacher if self.training else y, teacher_y, ds_nums, lam, index)

        elif not causal:
            loss, acc = self.apply_criterion(y_hats, y, y, ds_nums, lam, index)
        else:
            acc = None
            x_hat = y_hat[:, :-1]
            x_true = x[:, 1:]

            preds = torch.cat([x_hat[:, i, :x_hat.shape[1] - i] for i in range(x_hat.shape[1])], 1)
            targets = torch.cat([x_true[:, i:i + x_hat.shape[2]] for i in range(x_true.shape[1])], 1)

            mask = targets != 0
            if attn_mask is not None:
                attn_mask_ = torch.cat([
                    attn_mask[:, i:i + x_hat.shape[2]] for i in range(1, 1 + x_true.shape[1])], 1).unsqueeze(2)
                mask = mask & attn_mask_
            if torch.any(mask):
                loss = self.regression(preds[mask], targets[mask]) / np.prod(preds.shape) * 2000
            else:
                loss = 0 * y_hat[0, 0, 0]
        return loss, y_hat, acc

    def calculate_loss(self, batch, batch_idx):
        features, attn_mask, y, ds_nums = batch
        features = features.to(self.device).to(self.dtype)
        y = y.to(self.device).long()

        if constants.CAUSAL_FRACTION == 0:
            causal = False
        elif constants.CAUSAL_FRACTION == 1:
            causal = True
        elif constants.CAUSAL_FRACTION > .5:
            every_n = int(1 / (1 - constants.CAUSAL_FRACTION))
            causal = (1 + batch_idx + self.local_rank) % every_n != 0
        else:
            every_n = int(1 / constants.CAUSAL_FRACTION)
            causal = (batch_idx + self.local_rank) % every_n == 0

        loss, y_hat, acc = self.forward_loss(features, attn_mask, y, causal, ds_nums)

        return loss, acc

    def calculate_loss_val(self, batch):
        features, attn_mask, y, ds_num = batch
        features = features.to(self.device).to(self.dtype)
        y = y.to(self.device).long()

        loss, y_hat, acc = self.forward_loss(features, attn_mask, y, False, ds_num)
        if constants.CAUSAL_FRACTION:
            causal, _, _ = self.forward_loss(features, attn_mask, y, True, ds_num)
        else:
            causal = None
        return loss, causal, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.calculate_loss(batch, batch_idx)
        if acc is None:
            self.log(f'train/causal', loss.detach(), on_step=False, on_epoch=True, prog_bar=False,
                     sync_dist=True)
        else:
            self.log(f'train/loss', loss.detach(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f'train/acc', acc, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def on_before_backward(self, loss: torch.Tensor) -> None:
        if self.model_ema:
            self.model_ema.update(self.model)

    def validation_step(self, batch, batch_idx):
        loss, causal, acc = self.calculate_loss_val(batch)
        self.log('val/loss', loss.detach(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        if causal is not None:
            self.log('val/causal', causal.detach(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self, kind=constants.OPTIMIZER, scheduler_type=constants.SCHEDULER):
        params = sum(p.numel() for p in self.trainer.model.parameters() if p.requires_grad)
        heads_parameters = [sum(p.numel() for p in m.parameters() if p.requires_grad) for m in self.model.heads]
        subtract = sum(heads_parameters) - heads_parameters[1]
        self.logger.log_hyperparams({
            'hyp/lr': self.lr,
            'model/parameters': params - subtract,
            'model/heads': len(self.model.heads),
            'model/heads_parameters': subtract,
        })
        self.print(f'Initializing optimizer with LR: {self.lr}')
        if kind == 'sgd':
            optimizer = torch.optim.SGD(
                self.trainer.model.parameters(), lr=self.lr, nesterov=True, weight_decay=constants.WEIGHT_DECAY,
                momentum=.937)
        elif kind == 'adamw':
            optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=self.lr,
                                          weight_decay=constants.WEIGHT_DECAY)
        elif kind == 'lion':
            optimizer = Lion(self.trainer.model.parameters(), lr=self.lr, weight_decay=constants.WEIGHT_DECAY * 10)
        else:
            raise ValueError(f'Unknown optimizer: {kind}')

        if scheduler_type == '1cycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=self.lr, steps_per_epoch=self.steps_per_epoch, epochs=self.epochs,
                anneal_strategy='cos', base_momentum=0.8, max_momentum=0.937, pct_start=1e-8, div_factor=10
            )
        elif scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, self.epochs * self.steps_per_epoch, 2, self.lr * .1)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }
