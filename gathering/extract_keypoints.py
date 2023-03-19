import multiprocessing
import os
import sys
from itertools import repeat

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from tqdm import tqdm

from gathering import mp


def extract_video(args):
    name, videos_folder = args
    video_path = os.path.join(videos_folder, name)
    keypoints = list(mp.process_video(video_path))
    return keypoints, name


def extract_keypoints(videos_folder):
    files = os.listdir(videos_folder)
    if os.path.basename(videos_folder) == 'videos':
        dst_folder = os.path.join(os.path.dirname(videos_folder), 'keypoints')
    else:
        dst_folder = videos_folder.rstrip('/') + '_keypoints'

    os.makedirs(dst_folder, exist_ok=True)
    extracted = {os.path.splitext(f)[0] for f in os.listdir(dst_folder)}
    files = [f for f in files if os.path.splitext(f)[0] not in extracted]

    with multiprocessing.Pool(4) as pool:
        for keypoints, name in tqdm(
                pool.imap(extract_video, zip(files, repeat(videos_folder))),
                desc=f'[Extracting {videos_folder}]',
                total=len(files), unit='videos', smoothing=0,
                initial=len(extracted)
        ):
            save_path = os.path.join(dst_folder, os.path.splitext(name)[0] + '.fp16')
            keypoints = np.array(keypoints, dtype=np.float16)

            assert keypoints.shape[1] == 543, keypoints.shape
            assert keypoints.shape[2] == 3, keypoints.shape

            dumped = keypoints.tobytes()
            with open(save_path, 'wb') as f:
                f.write(dumped)

    return


if __name__ == '__main__':
    import signingsavvy

    extract_keypoints(signingsavvy.videos_folder)
