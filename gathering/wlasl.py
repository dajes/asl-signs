# Download data from https://github.com/dxli94/WLASL
import os
import sys

from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import cv2
import json
import requests

import constants

from gathering.utility import hash_url, download_video
from gathering import signasl

DATA_URL = 'https://raw.githubusercontent.com/dxli94/WLASL/ac00e6be631c1a2a486621b65f202219f3964d6b/start_kit/WLASL_v0.3.json'
DST_PATH = os.path.join(constants.DATASET_PATH, 'external', 'wlasl')
full_folder = os.path.join(DST_PATH, 'full_videos')
videos_folder = os.path.join(DST_PATH, 'videos')


def download_wsasl():
    os.makedirs(DST_PATH, exist_ok=True)
    data_path = os.path.join(DST_PATH, 'WLASL_v0.3.json')
    if os.path.exists(data_path):
        with open(data_path, 'r') as f:
            data = json.load(f)
    else:
        data = requests.get(DATA_URL).content
        with open(data_path, 'wb') as f:
            f.write(data)
        data = json.loads(data)

    video2annotations = {}
    for sign in data:
        for instance in sign['instances']:
            instance['sign'] = sign['gloss']
            del instance['instance_id']
            del instance['video_id']
            lst = video2annotations.setdefault(instance['url'], [])
            if instance not in lst:
                lst.append(instance)
    sorted_videos = sorted(video2annotations.keys())

    db_path = os.path.join(DST_PATH, 'wlasl.csv')
    if os.path.exists(db_path):
        with open(db_path, 'r') as f:
            lines = [line.split(',') for line in f.read().splitlines()[1:]]
        max_i = max(int(line[0]) for line in lines) if lines else 0
    else:
        max_i = 0
        with open(db_path, 'a') as f:
            f.write('i,j,label,signer_id,video_src')

    os.makedirs(full_folder, exist_ok=True)
    possibly_cached = [signasl.videos_folder, full_folder]

    pbar = tqdm(total=len(sorted_videos), initial=max_i, desc='Downloading videos')

    for i, video_url in enumerate(sorted_videos[max_i:], start=max_i):
        video_id = hash_url(video_url)
        annotations = video2annotations[video_url]

        video_path = None
        for folder in possibly_cached:
            video_path_ = os.path.join(folder, f'{video_id}.mp4')
            if os.path.exists(video_path_):
                video_path = video_path_
                break
        if video_path is None:
            video_data = download_video(video_url)
            if video_data is None:
                continue
            video_path = os.path.join(full_folder, f'{video_id}.mp4')
            with open(video_path, 'wb') as f:
                f.write(video_data)

        intervals = {}
        for annotation in annotations:
            interval = annotation['frame_start'], annotation['frame_end']
            if interval in intervals:
                continue
            intervals[interval] = (annotation['sign'], annotation['signer_id'], annotation['fps'])

        cap = cv2.VideoCapture(video_path)
        writers = {}

        os.makedirs(videos_folder, exist_ok=True)
        j = 1
        read, frame = cap.read()
        while read:
            for (start, end), (label, signer_id, fps) in intervals.items():
                if j == start:
                    writers[(start, end)] = cv2.VideoWriter(
                        os.path.join(videos_folder, f'{video_id}_{start}_{end}.mp4'),
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps, (frame.shape[1], frame.shape[0])
                    )
                if j >= start and (j <= end or end == -1):
                    writers[(start, end)].write(frame)
                if j == end:
                    writers[(start, end)].release()
                    del writers[(start, end)]
                    with open(db_path, 'a') as f:
                        f.write(f'{i},{j},{label},{signer_id},{video_id}_{start}_{end}')

            read, frame = cap.read()
            j += 1
        cap.release()
        for (start, end), writer in writers.items():
            writer.release()
            label, signer_id, fps = intervals[(start, end)]
            with open(db_path, 'a') as f:
                f.write(f'\n{i},{j},{label},{signer_id},{video_id}_{start}_{end}')
        writers.clear()
        pbar.update()
    pbar.close()


if __name__ == '__main__':
    download_wsasl()
