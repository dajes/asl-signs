import json
import os
from collections import Counter

import pandas as pd

from gathering import wlasl

if __name__ == '__main__':
    folder = os.path.dirname(wlasl.videos_folder)
    with open(os.path.join(folder, 'wlasl.csv'), 'r') as f:
        orig_table = f.read().splitlines()
    lines = [l.split(',') for l in orig_table][1:]

    paths = [os.path.join('keypoints', l[-1] + '.fp16') for l in lines[1:]]
    signs = [l[2].lower() for l in lines[1:]]

    counter = dict(sorted(Counter(signs).items(), key=lambda x: x[1], reverse=True))
    sign2idx = {k: i for i, k in enumerate(counter.keys())}

    with open(os.path.join(folder, 'sign_to_prediction_index_map.json'), 'w') as f:
        json.dump(sign2idx, f)

    table = pd.DataFrame([
        [n, -1, -1, sign]
        for n, sign in zip(paths, signs)
    ], columns=['path', 'participant_id', 'sequence_id', 'sign'])

    table.to_csv(os.path.join(folder, 'train.csv'), index=False)
