# Parses signingsavvy.com
import sys
import os
import traceback

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import random
import re
import time

import requests
from tqdm import tqdm

import constants


def parse_response(response):
    if response.status_code != 200:
        raise ValueError(f'Error {response.status_code} for {response.url}')
    text = ' '.join(response.text.split())
    if re.findall(r'<strong>ERROR:<\/strong>', text) or re.findall(r'<\/i>Mature Word<\/h3>', text):
        raise FileNotFoundError(f'Error for {response.url}')
    plabel = re.findall(r"=\"signing_header\"> <h2>([\w '-\.’:\"]+)<\/h2> <ul>", text)
    if not plabel:
        plabel = [' '.join(e) for e in re.findall(r"<h2>([\w '-\.’:\"]+)<\/h2><h3><em>\(([^\)]+)", text)]
    assert len(plabel) == 1
    label = plabel[0]
    label = label.replace(' ', '_').replace('"', '').replace('"', '').replace('.', '_').replace('’', '').replace(':', '_').replace('-', '_')
    variations = list(map(int, re.findall(r'>\w+<br>(\d+)<\/a>', text)))
    assert variations == list(range(1, len(variations) + 1)), 'variations are not consecutive'
    purl = re.findall(r'<link rel="preload" as="video" href="([^"]+)">', text)
    if len(purl) != 1:
        raise FileNotFoundError(f'No videos {response.url}')
    video_url = purl[0]
    return label, video_url, variations


def parse_signingsavvy():
    os.makedirs(DST_PATH, exist_ok=True)
    present_files = os.listdir(DST_PATH)
    if present_files:
        max_n = max(int(f.split('_')[0]) for f in present_files)
    else:
        max_n = 0

    n = max(1, max_n - 1)
    pbar = tqdm(initial=n, desc='Downloading signs')
    while True:
        i = 1
        url = URL.format(n=n, i=i)
        try:
            try:
                response = requests.get(url)
                label, video_url, variations = parse_response(response)
                while True:
                    dst_path = os.path.join(DST_PATH, f'{n}_{i}_{label}.mp4')
                    video_data = requests.get(video_url).content
                    with open(dst_path, 'wb') as f:
                        f.write(video_data)
                    if i in variations:
                        variations.remove(i)
                    if not variations:
                        break
                    i = variations[0]
                    url = URL.format(n=n, i=i)
                    response = requests.get(url)
                    label, video_url, _ = parse_response(response)
            except FileNotFoundError:
                print(f'Not found: {url}')

            pbar.update(1)
            n += 1
        except Exception as e:
            print(''.join(traceback.format_exception(type(e), e, e.__traceback__)))
            print(url)
            time.sleep(random.uniform(0.5, 1.5))


URL = 'https://www.signingsavvy.com/sign/_/{n}/{i}'
DST_PATH = os.path.join(constants.DATASET_PATH, 'external', 'signingsavvy', 'videos')
videos_folder = DST_PATH

if __name__ == '__main__':
    parse_signingsavvy()
