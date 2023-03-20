# Parses signasl.org
import base64
import hashlib
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import re

import requests
from lxml import etree
from tqdm import tqdm

from gathering.utility import download_video
import constants


def parse_page(letter, page):
    response = requests.get(URL_LIST.format(letter=letter, page=page))
    text = ' '.join(response.text.split())
    signs = re.findall(r'<a href="\/sign\/([^"]+)"', text)
    pages = list(map(int, re.findall(rf"<a href='\/dictionary\/{letter}\/(\d+)'>", text)))
    assert pages == list(range(1, max(pages) + 1)), 'pages are not consecutive'
    return signs, pages


def get_words():
    """Cached getter of all words from signasl.org"""
    os.makedirs(DST_PATH, exist_ok=True)
    words_path = os.path.join(DST_PATH, 'words.txt')
    if os.path.exists(words_path):
        with open(words_path, 'r') as f:
            words = set(f.read().split())
        start_letter = max(words)[0]
        if start_letter == 'z':
            return words
    else:
        start_letter = 'a'
        words = set()
    tk0 = tqdm(desc='Parsing signasl.org words', initial=len(words), unit='words')
    for letter in map(chr, range(ord(start_letter), ord('z') + 1)):
        i = 0
        pages = [1]
        while i < len(pages):
            page = pages[i]
            signs, pages = parse_page(letter, page)
            signs = {
                sign.replace(' ', '_')
                for sign in signs
            }
            added_signs = signs - words
            if added_signs:
                with open(words_path, 'a') as f:
                    f.write(' '.join(added_signs))
            words |= signs
            tk0.set_postfix(letter=letter, page=page)
            tk0.update(len(added_signs))
            i += 1
    tk0.close()
    return words


def parse_signasl():
    words = sorted(get_words())

    os.makedirs(video_folder, exist_ok=True)
    db_path = os.path.join(DST_PATH, 'signasl.csv')
    if os.path.exists(db_path):
        with open(db_path, 'r') as f:
            lines = [line.split(',') for line in f.read().splitlines()[1:]]
        added_ids = {line[2] for line in lines}
        max_i = max(int(line[0]) for line in lines) if lines else 0
    else:
        added_ids = set()
        max_i = 0
        with open(db_path, 'a') as f:
            f.write('i,label,video_id,video_src')
    # estimate total videos based on running mean value
    tk0 = tqdm(
        desc='Parsing signasl.org videos',
        unit='videos',
        initial=len(added_ids),
        total=int(round(len(words) * len(added_ids) / (max_i + 1))) or 1,
        smoothing=0
    )

    for i, word in enumerate(words[max_i:], start=max_i):
        url = URL_SIGN.format(word=word)
        response = requests.get(url)
        tree = etree.HTML(response.text)
        videos = tree.xpath("//div[@itemprop='video']")
        for video in videos:
            video = etree.HTML(etree.tostring(video))
            name = video.xpath("//div/i")
            assert len(name) == 1
            name = name[0].text
            src = video.xpath("//video/source")
            if src:
                assert len(src) == 1
                src = src[0].attrib['src']
            else:
                src = video.xpath("//iframe")[0].attrib['src']
                idxs = src.split('?')[0].split('/')[-2:]
                assert idxs[0] == 'embed'
                src = f'https://www.youtube.com/watch?v={idxs[1]}'

            orig_src = src
            # Create a unique id based on the video source
            video_id = base64.urlsafe_b64encode(hashlib.md5(src.encode()).digest()).decode('ascii').replace('=', '')
            if video_id in added_ids:
                continue

            video_data = download_video(src)
            if video_data is None:
                print('Could not get video from YouTube:', name, orig_src)
                continue
            with open(db_path, 'a') as f:
                f.write(f'\n{i},{name},{video_id},{orig_src}')
            added_ids.add(video_id)

            video_path = os.path.join(video_folder, video_id) + '.mp4'
            with open(video_path, 'wb') as f:
                f.write(video_data)

            # estimate total videos based on running mean value
            tk0.total = int(round(len(words) * len(added_ids) / (i + 1)))
            tk0.update(1)

    return


URL_LIST = 'https://www.signasl.org/dictionary/{letter}/{page}'
URL_SIGN = 'https://www.signasl.org/sign/{word}'

DST_PATH = os.path.join(constants.DATASET_PATH, 'external', 'signasl')
videos_folder = os.path.join(DST_PATH, 'videos')

if __name__ == '__main__':
    parse_signasl()
