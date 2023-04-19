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
    min_lips = [13, 14, 81, 95, 178, 191, 311, 324, 402, 415]

    eye_left = [362, 381, 374, 390, 263, 388, 386, 384]
    eye_right = [33, 163, 145, 154, 133, 157, 159, 161]

    pose_hands = [500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510, 511, 512, 513]

    pose_palms = [504, 505]
    pose_elbows = [502, 503]

    lips_center = 13
    min_left_hand = [468, 472, 473, 476, 480, 484, 485, 488]
    min_right_hand = [522, 526, 527, 530, 534, 538, 539, 542]

    relevant_ids = sorted({
        # baseline
        *list(range(idx_range_hand_left[0], idx_range_hand_left[1])),
        *list(range(idx_range_hand_right[0], idx_range_hand_right[1])),
        *min_lips,
        # *lips_upper_outer, *lips_lower_outer, *lips_upper_inner, *lips_lower_inner,

        *eye_left, *eye_right,
        *pose_elbows,
        *pose_palms,

        # minimalistic, about 5% worse but uses only 17 keypoints!
        # lips_center,
        # *min_left_hand,
        # *min_right_hand,

        # experimental
        # *list(range(idx_range_face[0], idx_range_face[1])),
        # *list(range(idx_range_pose[0], idx_range_pose[1])),
        # *pose_hands
    })
    CENTER_AROUND = sorted(set(range(*idx_range_face)) & set(relevant_ids))
    rids = torch.tensor(relevant_ids)

    NORMALIZE = True

    @staticmethod
    def get_center_around():
        return [BasicDataset.relevant_ids.index(idx) for idx in BasicDataset.CENTER_AROUND]

    def random_split(self, split_point: float = 0.8, by_participant: bool = True):
        state = np.random.RandomState(42)
        if by_participant:
            unique_participants = np.unique(self.participants)
            state.shuffle(unique_participants)
            split = int(len(unique_participants) * split_point)
            train_participants = unique_participants[:split]
            val_participants = unique_participants[split:]

            first = np.where(np.isin(self.participants, train_participants))[0]
            second = np.where(np.isin(self.participants, val_participants))[0]
        else:
            first, second = np.split(state.permutation(len(self)), [int(len(self) * split_point)])

        train = BasicDataset(
            self.name,
            self.prefix,
            [self.parquets[i] for i in first],
            [self.participants[i] for i in first],
            [self.labels[i] for i in first],
            self.n_coords,
            self.max_len
        )
        val = BasicDataset(
            self.name,
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
        name = os.path.basename(os.path.dirname(path))
        table = pd.read_csv(path)

        with open(os.path.join(os.path.dirname(path), 'sign_to_prediction_index_map.json'), 'r') as f:
            mapping = json.load(f)
        assert len(mapping) == len(set(mapping.values()))

        parquets = np.array(table['path']).tolist()
        max_prefix = os.path.commonprefix((max(parquets), min(parquets))).replace("train_landmark_files", "train_landmark_arrays")
        prefix = os.path.join(os.path.dirname(path), max_prefix)
        labels = np.array([mapping[n] for n in table['sign']], np.uint32)
        parquets = [p[len(max_prefix)-1:] for p in parquets]
        participants = np.array(table['participant_id'], np.uint16)
        return cls(name, prefix, parquets, participants, labels, n_coords, max_len)

    def __init__(self, name, prefix, parquets, participants, labels, n_coords: int = 3, max_len: int = 64, datas=None,
                 lengths=None):
        self.name = name
        self.prefix = prefix
        self.parquets = parquets
        self.participants = participants
        self.labels = labels
        self.n_outputs = len(np.unique(self.labels))
        self.ds_num = 1
        self.n_coords = n_coords
        self.max_len = max_len
        self.n_features = len(self.relevant_ids) * self.n_coords
        self.center_around = torch.tensor(self.get_center_around())

        self.datas = datas
        self.lengths = lengths
        self.cumsum = None

        self.train = False
        self.temporal = TemporalInterpolation(max_len)
        self.flippin = FlippinAugmentation(BasicDataset, flip_prob=.5)
        self.rotatin = RotateAugmentation()
        self.scalin = ScaleAugmentation()
        self.shiftin = ShiftAugmentation(BasicDataset)

    def __str__(self):
        return f'{self.name}({len(self)}, {self.n_outputs})'

    def __repr__(self):
        return str(self)

    def load_into_memory(self):
        paths = [f'{self.prefix}{p}' for p in self.parquets]
        hashing = hashlib.md5()
        for p in self.parquets:
            hashing.update(p.encode())
        for p in map(str, sorted(self.relevant_ids)):
            hashing.update(p.encode())
        digest = hashing.digest()
        encoded = base64.b64encode(digest).decode().replace('/', '_').replace('+', '-').replace('=', '')
        cache_path = os.path.join(os.path.dirname(self.prefix), '.cache', f'{encoded}')
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                data = pickle.loads(f.read())
            self.datas = data['datas'].reshape(-1, len(self.rids), 3)
            self.lengths = data['lengths'] // len(self.rids)
            self.cumsum = np.cumsum(self.lengths)
        else:
            datas = []
            with Pool(8) as p:
                for data in tqdm(p.imap(read_parquet, paths, 32), total=len(paths), desc=f'Loading {self} into memory'):
                    datas.append(data.reshape(-1, self.ROWS_PER_FRAME, 3)[:, self.rids].reshape(-1, 3)
                                 .astype(np.float16))
            lengths = np.array([len(d) for d in datas], np.uint32)
            datas = np.concatenate(datas, 0)
            dumped = pickle.dumps({
                'datas': datas,
                'lengths': lengths,
            })
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'wb') as f:
                f.write(dumped)
            self.datas = datas.reshape(-1, len(self.rids), 3)
            self.lengths = lengths // len(self.rids)
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
            if self.NORMALIZE:
                non_nan = relevant_data[~np.isnan(relevant_data).any(axis=2)]
                if len(non_nan):
                    relevant_data -= non_nan.mean(0, keepdims=True)[None]
            relevant_data = self.temporal(relevant_data)
            relevant_data = self.flippin(relevant_data)
            relevant_data = self.shiftin(relevant_data)
            relevant_data = self.rotatin(relevant_data)
            relevant_data = self.scalin(relevant_data)

        data = torch.zeros((len(relevant_data), len(self.relevant_ids), self.n_coords), dtype=torch.float32)
        data[:] = torch.from_numpy(relevant_data[:, :, :self.n_coords])

        if self.NORMALIZE:
            data_subset = data[:, self.center_around]
            m = ~torch.isnan(data_subset).any(dim=2)
            non_nan = data_subset[m]
            if len(non_nan) > 0:
                data -= non_nan.mean(0, keepdims=True)[None]

        return data, int(label), self.ds_num

    @property
    def collate_fn(self):
        max_len = self.max_len
        train = self.train

        def collate_fn(batch):
            features, labels, ds_nums = zip(*batch)
            features_lengths = [len(f) for f in features]
            local_max_len = min(max_len, max(features_lengths))
            starts = [0] * len(features)
            features_ = []
            for f, start_ in zip(features, starts):
                f = preprocess(f, local_max_len)
                if len(f) < local_max_len:
                    f = F.pad(f, (0, 0, 0, 0, start_, local_max_len - f.shape[0] - start_), 'constant', 0)
                elif train:
                    start = np.random.randint(0, len(f) - local_max_len + 1)
                    f = f[start:start + local_max_len]
                elif len(f) > local_max_len:
                    f = f[:local_max_len]
                features_.append(f.reshape(f.shape[0], -1))
            features = features_
            features = torch.stack(features)
            labels = np.stack(labels)
            attn_mask = torch.zeros((len(features), local_max_len), dtype=torch.bool)
            for i, (l, s) in enumerate(zip(features_lengths, starts)):
                attn_mask[i, s:s + l] = 1
            return features, attn_mask, torch.from_numpy(labels), torch.from_numpy(np.array(ds_nums))

        return collate_fn


def __visualize_keypoints():
    from matplotlib import pyplot as plt
    from utils import seed_everything
    import constants

    seed_everything(42)
    train_ds = BasicDataset.from_csv(os.path.join(constants.DATASET_PATH, 'asl-signs/train.csv')).random_split(1 / 21)[
        0]
    train_ds.relevant_ids = list(range(0, 543))

    example = train_ds[0]
    data = example[0]
    data = data[data.shape[0] // 2].numpy()

    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    ax.set_aspect('equal')
    ax.scatter(data[:468, 0], -data[:468, 1], s=1)
    for i, (x, y) in enumerate(zip(data[:468, 0], -data[:468, 1])):
        ax.text(x, y, str(i), fontsize=6, c='red' if i in BasicDataset.relevant_ids else 'black')
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.subplots_adjust(0, 0, 1, 1, wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()

    relevant = [idx for idx in BasicDataset.relevant_ids if idx >= 468]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.set_aspect('equal')
    ax.scatter(data[468:, 0], -data[468:, 1], s=3)
    ax.scatter(data[relevant, 0],
               -data[relevant, 1], s=10)

    for i, (x, y, _) in zip(relevant, data[relevant]):
        ax.text(x, -y, str(i), fontsize=6, color='orange')

    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.subplots_adjust(0, 0, 1, 1, wspace=0, hspace=0)
    plt.tight_layout()
    plt.show()
    return


def __visualize_augmentations():
    from matplotlib import pyplot as plt
    from utils import seed_everything
    import constants

    seed_everything(42)
    train_ds = BasicDataset.from_csv(os.path.join(constants.DATASET_PATH, 'asl-signs/train.csv')).random_split(1 / 21)[
        0]

    interesting_points = [
        train_ds.relevant_ids.index(train_ds.idx_range_hand_left[0] + 5),  # center of left palm
        train_ds.relevant_ids.index(train_ds.idx_range_hand_right[0] + 5),  # center of right palm
        train_ds.relevant_ids.index(train_ds.idx_range_face[0] + 13),  # center of upper lip
    ]
    colors = ['r', 'g', 'b']

    collate_fn = train_ds.collate_fn

    for idx in [0, len(train_ds) // 3]:
        for i in range(100)[::-1]:
            train_ds.train = i > 0
            features, mask, labels, ds_num = collate_fn([train_ds[idx]])

            trajectories = features.reshape(
                features.shape[1], -1, train_ds.n_coords)[:, interesting_points].permute(1, 0, 2)[..., :2].numpy()

            nan_mask = trajectories == 0
            trajectories[nan_mask] = np.nan

            for trajectory, c in zip(trajectories, colors):
                plt.plot(trajectory[:, 0], -trajectory[:, 1], c=c, alpha=.2 if i > 0 else 1)
        plt.show()


def __visualize_mixup():
    from matplotlib import pyplot as plt
    from utils import seed_everything
    import constants

    seed_everything(42)
    train_ds = BasicDataset.from_csv(os.path.join(constants.DATASET_PATH, 'asl-signs/train.csv')).random_split(1 / 21)[
        0]

    interesting_points = [
        train_ds.relevant_ids.index(train_ds.idx_range_hand_left[0] + 5),  # center of left palm
        train_ds.relevant_ids.index(train_ds.idx_range_hand_right[0] + 5),  # center of right palm
        train_ds.relevant_ids.index(train_ds.idx_range_face[0] + 13),  # center of upper lip
    ]
    colors = ['r', 'g', 'b']

    collate_fn = train_ds.collate_fn

    for idx in [0, len(train_ds) // 3]:
        features, mask, labels, ds_num = collate_fn([train_ds[idx], train_ds[idx + 1]])
        trajectories = features.reshape(
            features.shape[0], features.shape[1], -1, train_ds.n_coords
        )[:, :, interesting_points].permute(0, 2, 1, 3)[..., :2].numpy()

        nan_mask = trajectories == 0
        trajectories[nan_mask] = np.nan
        for mixup in np.linspace(0, 1, 5)[1:-1]:
            trajectories_ = trajectories[0] * (1 - mixup) + trajectories[1] * mixup
            for trajectory, c in zip(trajectories_, colors):
                plt.plot(trajectory[:, 0], -trajectory[:, 1], c=c, alpha=.33)

        for trajectory, c in zip(trajectories[0], colors):
            plt.plot(trajectory[:, 0], -trajectory[:, 1], c=c)
        for trajectory, c in zip(trajectories[1], colors):
            plt.plot(trajectory[:, 0], -trajectory[:, 1], c=c)
        plt.show()


if __name__ == '__main__':
    __visualize_keypoints()
    # __visualize_augmentations()
    # __visualize_mixup()
