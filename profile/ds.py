import cProfile
import os
import pstats

import constants
from dataset.basic import BasicDataset
from utils import numerated_folder, seed_everything

seed_everything()
print('Loading dataset...')
seed_everything()

train_ds = BasicDataset.from_csv(os.path.join(constants.DATASET_PATH, 'asl-signs/train.csv')).random_split(1 / 21)[0]
train_ds.train = True
train_ds.load_into_memory()

folder = os.path.dirname(__file__)
filename = numerated_folder(os.path.join(folder, 'results', 'ds'), create=False) + '.prof'


def collate_fn(batch):
    return train_ds.collate_fn(batch)


print('Warming up...')
for i in range(10):
    collate_fn([train_ds.__getitem__(i) for _ in range(16)])

print('Profiling...')
with cProfile.Profile() as pr:
    for i in range(10, 110):
        collate_fn([train_ds.__getitem__(i) for _ in range(16)])

print('Saving...')
stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.dump_stats(filename=filename)
print(filename[len(os.path.dirname(folder)) + 1:])
