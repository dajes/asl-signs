import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

import constants
from dataset.basic import BasicDataset


def rewrite_parquet(path):
    data = pd.read_parquet(path, columns=['x', 'y', 'z']).values.astype(np.float16)
    np_path = f'{os.path.splitext(path)[0].replace("train_landmark_files", "train_landmark_arrays")}.fp16'
    os.makedirs(os.path.dirname(np_path), exist_ok=True)
    dumped = data.tobytes()
    with open(np_path, 'wb') as f:
        f.write(dumped)


if __name__ == '__main__':
    ds = BasicDataset.from_csv(os.path.join(constants.DATASET_PATH, 'asl-signs', 'train.csv'))

    paths = [f'{ds.prefix}{ds.parquets[i]}' for i in range(len(ds))]

    with Pool() as p:
        for _ in tqdm(p.imap(rewrite_parquet, paths), total=len(paths)):
            pass
