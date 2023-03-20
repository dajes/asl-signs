import json
import os

import pandas as pd

from gathering import lsfb

if __name__ == '__main__':
    folder = os.path.dirname(lsfb.videos_folder)
    lemmes = pd.read_csv(os.path.join(folder, 'lemmes.csv'))
    clips = pd.read_csv(os.path.join(folder, 'clips.csv'))

    paths = [e.replace('videos/', 'keypoints/').replace('.mp4', '.fp16') for e in clips['video']]
    signs = [e for e in clips['lemme']]

    sign2idx = {k: i for i, k in enumerate(lemmes['lemme'])}

    with open(os.path.join(folder, 'sign_to_prediction_index_map.json'), 'w') as f:
        json.dump(sign2idx, f)

    table = pd.DataFrame([
        [n, -1, -1, sign]
        for n, sign in zip(paths, signs)
    ], columns=['path', 'participant_id', 'sequence_id', 'sign'])

    table.to_csv(os.path.join(folder, 'train.csv'), index=False)
