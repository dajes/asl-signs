import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, RandomSampler

from dataset.basic import BasicDataset


class LightData(pl.LightningDataModule):

    def __init__(self, path, n_coords, max_len, batch_size, steps_per_epoch, epochs, num_workers):
        super().__init__()
        ds = BasicDataset.from_csv(path, n_coords, max_len)
        self.train_ds, self.val_ds = ds.random_split(19 / 21)
        assert np.unique(self.train_ds.labels).shape[0] == 250, "Train dataset should have 250 classes"
        self.train_ds.load_into_memory()
        self.train_ds.train = True
        self.n_features = ds.n_features
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.epochs = epochs
        self.num_works = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            sampler=RandomSampler(self.train_ds, num_samples=self.batch_size * self.steps_per_epoch, replacement=True),
            batch_size=self.batch_size,
            collate_fn=self.train_ds.collate_fn,
            num_workers=self.num_works,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_works > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=256,
            collate_fn=self.val_ds.collate_fn,
            num_workers=16 if self.num_works else 0,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_works > 0,
        )
