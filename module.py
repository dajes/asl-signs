import os
import warnings
from typing import List

import pytorch_lightning as pl
import torch

from architecture.linear import LinearArchitecture
from architecture.lstm import LSTMArchitecture
from architecture.mlp import MLPArchitecture
from architecture.transformer import TransformerArchitecture
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
    CAUSAL_FRACTION = 1 / 8
    CAUSAL_FORESIGHT = 1 / 4

    def __init__(
            self, working_dir: str, epochs: int, steps_per_epoch: int, lr: float,
            in_features: int, n_features: int, n_outputs: List[int], max_len: int, drop_rate: float = 0.1,
            depth: int = 6, num_heads: int = 8, mlp_ratio=4.,
            model_type: str = 'mlp'
    ):
        super().__init__()
        self.working_dir = working_dir
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.lr = lr
        self.save_hyperparameters()

        self.model = {
            'linear': LinearArchitecture,
            'mlp': MLPArchitecture,
            'lstm': LSTMArchitecture,
            'transformer': TransformerArchitecture,
        }[model_type](in_features, n_features, n_outputs, max_len, drop_rate, depth, num_heads, mlp_ratio)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.regression = torch.nn.SmoothL1Loss()

    def forward(self, x, task_n):
        return self.model(x, task_n)

    @torch.compile
    def forward_loss(self, x, y, ds_num):
        y_hat = self(x, ds_num)

        if ds_num:
            loss = self.criterion(y_hat, y)
        else:
            pad = int(self.CAUSAL_FORESIGHT * x.shape[1]) or 1
            x_hat = y_hat[:, :-pad]
            x_true = x[:, pad:]
            mask = x_true != 0
            if torch.any(mask):
                loss = self.regression(x_hat[mask], x_true[mask]) * 20
            else:
                loss = 0 * y_hat[0, 0, 0]
        return loss, y_hat

    def calculate_loss(self, batch, batch_idx):
        features, y, ds_num = batch
        features = features.to(self.device).to(self.dtype)
        y = y.to(self.device).long()

        if self.CAUSAL_FRACTION == 0:
            causal = False
        elif self.CAUSAL_FRACTION == 1:
            causal = True
        elif self.CAUSAL_FRACTION > .5:
            every_n = int(1 / (1 - self.CAUSAL_FRACTION))
            causal = (1 + batch_idx + self.local_rank) % every_n != 0
        else:
            every_n = int(1 / self.CAUSAL_FRACTION)
            causal = (batch_idx + self.local_rank) % every_n == 0

        loss, y_hat = self.forward_loss(features, y, 0 if causal else ds_num)
        if causal:
            return loss, None, ds_num

        with torch.no_grad():
            acc = (y_hat.argmax(dim=-1) == y).float().mean()
        return loss, acc, ds_num

    def calculate_loss_val(self, batch):
        features, y, ds_num = batch
        features = features.to(self.device).to(self.dtype)
        y = y.to(self.device).long()

        loss, y_hat = self.forward_loss(features, y, ds_num)
        causal, _ = self.forward_loss(features, y, 0)
        with torch.no_grad():
            acc = (y_hat.argmax(dim=-1) == y).float().mean()
        return loss, causal, acc, ds_num

    def training_step(self, batch, batch_idx):
        loss, acc, ds_num = self.calculate_loss(batch, batch_idx)
        if acc is None:
            self.log(f'train/causal{ds_num}', loss.detach(), on_step=False, on_epoch=True, prog_bar=False,
                     sync_dist=True)
        else:
            self.log(f'train/loss{ds_num}', loss.detach(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
            self.log(f'train/acc{ds_num}', acc, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, causal, acc, ds_num = self.calculate_loss_val(batch)
        self.log('val/loss', loss.detach(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val/causal', causal.detach(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def configure_optimizers(self, kind='adamw'):
        params = sum(p.numel() for p in self.trainer.model.parameters() if p.requires_grad)
        self.logger.log_hyperparams({
            'hyp/lr': self.lr,
            'model/parameters': params,
        })
        self.logger.log_metrics({
            'model/parameters': params,
        })
        self.print(f'Initializing optimizer with LR: {self.lr}')
        if kind == 'sgd':
            optimizer = torch.optim.SGD(
                self.trainer.model.parameters(), lr=self.lr, nesterov=True, weight_decay=5e-4, momentum=.937)
        elif kind == 'adamw':
            optimizer = torch.optim.AdamW(self.trainer.model.parameters(), lr=self.lr, weight_decay=5e-4,
                                          betas=(.937, .999))

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=self.lr, steps_per_epoch=self.steps_per_epoch, epochs=self.epochs,
                    anneal_strategy='cos', base_momentum=0.8, max_momentum=0.937,
                ),
                "interval": "step",
                "frequency": 1,
            }
        }
