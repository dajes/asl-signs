import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, RandomSampler

from dataset.basic import BasicDataset
from dataset.ensemble import DatasetEnsemble


class LightData(pl.LightningDataModule):

    def __init__(self, path, n_coords, max_len, batch_size, steps_per_epoch, epochs, num_workers,
                 external_datasets):
        super().__init__()
        ds = BasicDataset.from_csv(path, n_coords, max_len)
        main_train_ds, self.val_ds = ds.random_split(19 / 21)
        if num_workers:
            main_train_ds.load_into_memory()
        main_train_ds.train = True
        datasets = [main_train_ds]
        for ds_num, external_dataset in enumerate(external_datasets, 2):
            external_dataset = BasicDataset.from_csv(external_dataset, n_coords, max_len)
            external_dataset.train = True
            external_dataset.ds_num = ds_num
            if num_workers:
                external_dataset.load_into_memory()
            datasets.append(external_dataset)
        self.train_ds = DatasetEnsemble(datasets)
        self.n_outputs = self.train_ds.n_outputs
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
