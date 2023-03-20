import os
import json
import pandas as pd
from collections import Counter

from gathering import signingsavvy


if __name__ == '__main__':
    folder = os.path.dirname(signingsavvy.DST_PATH)

    videos = os.listdir(signingsavvy.videos_folder)

    labels = [n[:-4].split('_', 2)[2] for n in videos]
    counter = dict(sorted(Counter(labels).items(), key=lambda x: x[1], reverse=True))
    sign2idx = {k: i for i, k in enumerate(counter.keys())}

    with open(os.path.join(folder, 'sign_to_prediction_index_map.json'), 'w') as f:
        json.dump(sign2idx, f)

    table = pd.DataFrame([
        [os.path.join('keypoints', os.path.splitext(n)[0] + '.fp16'), -1, -1, sign]
        for n, sign in zip(videos, labels)
    ], columns=['path', 'participant_id', 'sequence_id', 'sign'])

    table.to_csv(os.path.join(folder, 'train.csv'), index=False)
