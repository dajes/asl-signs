import json
import os
from collections import Counter

import pandas as pd

from gathering import signasl

if __name__ == '__main__':
    folder = os.path.dirname(signasl.videos_folder)
    with open(os.path.join(folder, 'signasl.csv'), 'r') as f:
        orig_table = f.read().splitlines()
    lines = [l.split(',') for l in orig_table]

    paths = [os.path.join('keypoints', l[-2] + '.fp16') for l in lines[1:]]
    signs = [l[1].lower() for l in lines[1:]]

    m = [os.path.exists(os.path.join(folder, p)) for p in paths]

    print(f'Number of missing files: {len(m) - sum(m)} ({1 - sum(m) / len(m):%})')

    paths = [p for p, include in zip(paths, m) if include]
    signs = [s for s, include in zip(signs, m) if include]

    counter = dict(sorted(Counter(signs).items(), key=lambda x: x[1], reverse=True))
    sign2idx = {k: i for i, k in enumerate(counter.keys())}

    with open(os.path.join(folder, 'sign_to_prediction_index_map.json'), 'w') as f:
        json.dump(sign2idx, f)

    table = pd.DataFrame([
        [n, -1, -1, sign]
        for n, sign in zip(paths, signs)
    ], columns=['path', 'participant_id', 'sequence_id', 'sign'])

    table.to_csv(os.path.join(folder, 'train.csv'), index=False)
