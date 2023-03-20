import base64
import hashlib
import time
from typing import Optional

import requests
from pytube import YouTube
from pytube.exceptions import RegexMatchError


def hash_url(url) -> str:
    return base64.urlsafe_b64encode(hashlib.md5(url.encode()).digest()).decode('ascii').replace('=', '')


def download_video(src) -> Optional[bytes]:
    if 'www.youtube.com' in src:
        try:
            yt = YouTube(src)
        except RegexMatchError:
            return None
        src = None
        for retry in range(3):
            try:
                src = yt.streaming_data['formats'][-1]['url']
            except KeyError:
                time.sleep(1 << retry)
            else:
                break
        if src is None:
            return None

    for retry in range(6):
        try:
            return requests.get(src).content
        except requests.exceptions.SSLError:
            time.sleep(1 << retry)
