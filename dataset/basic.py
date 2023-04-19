import base64
import hashlib
import json
import os
import pickle
from multiprocessing import Pool

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

from augmentation.flippin import FlippinAugmentation
from augmentation.rotate import RotateAugmentation
from augmentation.scale import ScaleAugmentation
from augmentation.shift_augmentation import ShiftAugmentation
from augmentation.temporal import TemporalInterpolation
from modeling.preprocess import preprocess


def read_parquet(path):
    np_path = f'{os.path.splitext(path)[0]}.fp16'
    if os.path.exists(np_path):
        with open(np_path, 'rb') as f:
            raw = f.read()
        data = np.frombuffer(raw, dtype=np.float16).reshape(-1, 3).astype(np.float32)
    else:
        data = pd.read_parquet(path, columns=['x', 'y', 'z']).values.astype(np.float32)
    return data


class BasicDataset(Dataset):
    ROWS_PER_FRAME = 543
    idx_range_face = (0, 468)
    idx_range_hand_left = (468, 489)
    idx_range_pose = (489, 522)
    idx_range_hand_right = (522, 543)

    lips_upper_outer = [0, 37, 39, 40, 61, 185, 267, 269, 270, 291, 409]
    lips_lower_outer = [17, 61, 84, 91, 146, 181, 291, 314, 321, 375, 405]
    lips_upper_inner = [13, 78, 80, 81, 82, 191, 308, 310, 311, 312, 415]
    lips_lower_inner = [14, 78, 87, 88, 95, 178, 308, 317, 318, 324, 402]

    pose_hands = [500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513]

    relevant_ids = sorted({
        *list(range(idx_range_hand_left[0], idx_range_hand_left[1])),
        *list(range(idx_range_hand_right[0], idx_range_hand_right[1])),
        # *list(range(idx_range_face[0], idx_range_face[1])),
        # *list(range(idx_range_pose[0], idx_range_pose[1])),
        *lips_upper_outer, *lips_lower_outer, *lips_upper_inner, *lips_lower_inner,
        *pose_hands
    })
    rids = torch.tensor(relevant_ids)

    def random_split(self, split_point: float = 0.8):
        unique_participants = np.unique(self.participants)
        np.random.shuffle(unique_participants)
        split = int(len(unique_participants) * split_point)
        train_participants = unique_participants[:split]
        val_participants = unique_participants[split:]

        first = np.where(np.isin(self.participants, train_participants))[0]
        second = np.where(np.isin(self.participants, val_participants))[0]

        train = BasicDataset(
            self.prefix,
            [self.parquets[i] for i in first],
            [self.participants[i] for i in first],
            [self.labels[i] for i in first],
            self.n_coords,
            self.max_len
        )
        val = BasicDataset(
            self.prefix,
            [self.parquets[i] for i in second],
            [self.participants[i] for i in second],
            [self.labels[i] for i in second],
            self.n_coords,
            self.max_len
        )

        return train, val

    @classmethod
    def from_csv(cls, path, n_coords: int = 3, max_len: int = 64):
        table = pd.read_csv(path)

        with open(os.path.join(os.path.dirname(path), 'sign_to_prediction_index_map.json'), 'r') as f:
            mapping = json.load(f)
        assert len(mapping) == len(set(mapping.values()))
        assert 0 <= max(mapping.values()) < 250

        parquets = np.array(table['path']).tolist()
        max_prefix = os.path.commonprefix((max(parquets), min(parquets))).replace("train_landmark_files", "train_landmark_arrays")
        prefix = os.path.join(os.path.dirname(path), max_prefix)
        labels = np.array([mapping[n] for n in table['sign']], np.uint8)
        parquets = [p[len(max_prefix):] for p in parquets]
        participants = np.array(table['participant_id'], np.uint16)
        return cls(prefix, parquets, participants, labels, n_coords, max_len)

    def __init__(self, prefix, parquets, participants, labels, n_coords: int = 3, max_len: int = 64, datas=None,
                 lengths=None):
        self.prefix = prefix
        self.parquets = parquets
        self.participants = participants
        self.labels = labels
        self.n_outputs = len(np.unique(self.labels))
        self.ds_num = 1
        self.n_coords = n_coords
        self.max_len = max_len
        self.n_features = len(self.relevant_ids) * self.n_coords

        self.datas = datas
        self.lengths = lengths
        self.cumsum = None

        self.train = False
        self.temporal = TemporalInterpolation(max_len)
        self.flippin = FlippinAugmentation(BasicDataset, flip_prob=.5)
        self.rotatin = RotateAugmentation()
        self.scalin = ScaleAugmentation()
        self.shiftin = ShiftAugmentation()

    def load_into_memory(self):
        # WARNING: Peak RAM usage is 40GB
        paths = [f'{self.prefix}{p}' for p in self.parquets]
        hashing = hashlib.md5()
        for p in self.parquets:
            hashing.update(p.encode())
        digest = hashing.digest()
        encoded = base64.b64encode(digest).decode().replace('/', '_').replace('+', '-').replace('=', '')
        cache_path = os.path.join(os.path.dirname(self.prefix), '.cache', f'{encoded}')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                data = pickle.loads(f.read())
            self.datas = data['datas'].reshape(-1, self.ROWS_PER_FRAME, 3)[:, self.rids]
            self.lengths = data['lengths'] // self.ROWS_PER_FRAME
            self.cumsum = np.cumsum(self.lengths)
        else:
            datas = []
            with Pool(8) as p:
                for data in tqdm(p.imap(read_parquet, paths, 32), total=len(paths), desc='Loading data into memory'):
                    datas.append(data)
            lengths = np.array([len(d) for d in datas], np.uint32)
            datas = np.concatenate(datas, 0)
            dumped = pickle.dumps({
                'datas': datas,
                'lengths': lengths,
            })
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                f.write(dumped)
            self.datas = datas.reshape(-1, self.ROWS_PER_FRAME, 3)[:, self.rids]
            self.lengths = lengths // self.ROWS_PER_FRAME
            self.cumsum = np.cumsum(self.lengths)

        return

    def __len__(self):
        return len(self.parquets)

    def __getitem__(self, item):
        path = f'{self.prefix}{self.parquets[item]}'
        label = self.labels[item]

        if self.datas is not None:
            start = self.cumsum[item - 1] if item > 0 else 0
            end = start + self.lengths[item]
            data = self.datas[start:end]
            data = data.reshape((-1, len(self.relevant_ids), 3))
            relevant_data = data.astype(np.float32)
        else:
            data = read_parquet(path)
            n_frames = len(data) // self.ROWS_PER_FRAME
            data = data.reshape((n_frames, self.ROWS_PER_FRAME, 3))
            relevant_data = data[:, self.relevant_ids].astype(np.float32)

        if self.train:
            relevant_data = relevant_data
            if np.random.rand() < .8:
                relevant_data = self.temporal(relevant_data)
            relevant_data = self.flippin(relevant_data)
            relevant_data = self.shiftin(relevant_data)
            relevant_data = self.rotatin(relevant_data)
            relevant_data = self.scalin(relevant_data)
            relevant_data = self.shiftin(relevant_data)

        data = torch.zeros((len(relevant_data), len(self.relevant_ids), self.n_coords), dtype=torch.float32)
        data[:] = torch.from_numpy(relevant_data[:, :, :self.n_coords])

        return data, label, self.ds_num

    @property
    def collate_fn(self):
        max_len = self.max_len
        train = self.train

        def collate_fn(batch):
            features, labels, ds_nums = zip(*batch)
            local_max_len = min(max_len, max(len(f) for f in features))
            features_ = []
            for f in features:
                f = preprocess(f, local_max_len)
                if len(f) < local_max_len:
                    f = F.pad(f, (0, 0, 0, 0, 0, local_max_len - f.shape[0]), 'constant', 0)
                elif train:
                    start = np.random.randint(0, len(f) - local_max_len + 1)
                    f = f[start:start + local_max_len]
                elif len(f) > local_max_len:
                    f = f[:local_max_len]
                features_.append(f.reshape(f.shape[0], -1))
            features = features_
            features = torch.stack(features)
            labels = np.stack(labels)
            return features, torch.from_numpy(labels), ds_nums[0]

        return collate_fn
