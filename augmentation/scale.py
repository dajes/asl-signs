import numpy as np

from augmentation.base import Augmentation


class ScaleAugmentation(Augmentation):
    def __init__(self, scale: float = np.e / 2):
        self.scale = scale

    def on_points(self, points: np.ndarray):
        if self.scale == 0:
            return points
        scale = np.random.uniform(1 / self.scale, self.scale)
        return points * scale
