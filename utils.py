import os
import random

import numpy as np
import torch


def numerated_folder(folder_path: str, create: bool = True) -> str:
    dst_path, folder_name = os.path.split(folder_path)
    os.makedirs(dst_path, exist_ok=True)

    taken_names = [os.path.splitext(n)[0] for n in os.listdir(dst_path) if n.startswith(folder_name)]

    taken_idxes = {int(n[len(folder_name) + 1:]) for n in taken_names if n[len(folder_name) + 1:].isdigit()}

    idx = next(i for i in range(1, max(taken_idxes or [0]) + 2) if i not in taken_idxes)

    result = os.path.join(dst_path, f'{folder_name}_{idx:04d}')
    if create:
        os.makedirs(result)

    return result


def seed_everything(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
