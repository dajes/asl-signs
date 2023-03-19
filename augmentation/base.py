import numpy as np


class Augmentation:
    def on_points(self, points: np.ndarray):
        raise NotImplementedError()

    def __call__(self, data):
        data = data.copy()
        flattened = data[..., :2].reshape((-1, 2))
        augmented = self.on_points(flattened)
        data[..., :2] = augmented.reshape([*data.shape[:-1], 2])
        return data
