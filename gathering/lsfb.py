# Download data from jefidev.github.io/lsfb-dataset
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import constants
from lsfb_dataset.utils.download.dataset_downloader import DatasetDownloader

destination_folder = os.path.join(constants.DATASET_PATH, 'external', "lsfb")

ds = DatasetDownloader(
    destination_folder,
    dataset="isol",
    landmarks=[],
    include_video=True,
)
ds.download()
