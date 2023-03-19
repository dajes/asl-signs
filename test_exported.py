import multiprocessing as mp
import os

import numpy as np
import pandas as pd
from tensorflow.lite.python.interpreter import Interpreter
from tqdm import tqdm

import constants
from dataset.basic import BasicDataset, read_parquet
from export import tflite_path
from utils import seed_everything

ROWS_PER_FRAME = 543

seed_everything()
ds = BasicDataset.from_csv(os.path.join(constants.DATASET_PATH, 'asl-signs', 'train.csv'), 3, 1)
train_ds, val_ds = ds.random_split(19 / 21)

data_columns = ["x", "y", "z"]


def load_relevant_data_subset(pq_path):
    try:
        data = read_parquet(pq_path)
    except FileNotFoundError:
        data = pd.read_parquet(pq_path, columns=data_columns).values
    n_frames = int(len(data) / ROWS_PER_FRAME)
    data = data.reshape(n_frames, ROWS_PER_FRAME, len(data_columns))
    return data.astype(np.float32)


interpreter = Interpreter(tflite_path)
interpreter.allocate_tensors()

found_signatures = list(interpreter.get_signature_list().keys())
prediction_fn = interpreter.get_signature_runner("serving_default")


def predict(data):
    pq_path, label = data
    inputs = load_relevant_data_subset(pq_path)

    output = prediction_fn(inputs=inputs)
    sign = np.argmax(output["outputs"])

    return sign, label


data_iter = ((f'{val_ds.prefix}{val_ds.parquets[i]}', val_ds.labels[i]) for i in range(len(val_ds)))

with mp.Pool() as pool:
    tk0 = tqdm(pool.imap(predict, data_iter), total=len(val_ds), smoothing=0)

    stats = {}

    for sign, label in tk0:
        stats['total'] = stats.get('total', 0) + 1
        stats['correct'] = stats.get('correct', 0) + (sign == label)
        tk0.set_postfix(acc=f"{stats['correct'] / stats['total']:.4f}")
