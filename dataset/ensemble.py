import numpy as np

from torch.utils.data import Dataset


class DatasetEnsemble(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.borders = np.cumsum([0] + list(map(len, self.datasets)))

    def __len__(self):
        return self.borders[-1]

    def __getitem__(self, idx):
        for i, border in enumerate(self.borders[1:], 1):
            if idx < border:
                return self.datasets[i - 1][idx - self.borders[i - 1]]
        raise IndexError(f'Index {idx} is out of range. Dataset length is {len(self)}')

    @property
    def n_outputs(self):
        return [ds.n_outputs for ds in self.datasets]

    @property
    def collate_fn(self):
        return self.datasets[0].collate_fn
