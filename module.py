import os
import warnings

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
    def __init__(
            self, working_dir: str, epochs: int, steps_per_epoch: int, lr: float,
            in_features: int, n_features: int, n_classes: int, max_len: int, drop_rate: float = 0.1,
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
        }[model_type](in_features, n_features, n_classes, max_len, drop_rate, depth, num_heads, mlp_ratio)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def forward_loss(self, x, y):
        y_hat = self(x)

        loss = self.criterion(y_hat, y)
        return loss, y_hat

    def calculate_loss(self, batch):
        features, y = batch
        features = features.to(self.device).to(self.dtype)
        y = y.to(self.device).long()
        loss, y_hat = self.forward_loss(features, y)
        with torch.no_grad():
            acc = (y_hat.argmax(dim=-1) == y).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self.calculate_loss(batch)
        self.log('train/loss', loss.detach(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log('train/acc', acc, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self.calculate_loss(batch)
        self.log('val/loss', loss.detach(), on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
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
