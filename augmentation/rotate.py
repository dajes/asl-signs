import numpy as np

from augmentation.base import Augmentation


class RotateAugmentation(Augmentation):
    def on_points(self, points: np.ndarray):
        """Not used"""

    def __init__(self, angle=30.):
        self.angle = angle / 180 * np.pi

    def __call__(self, points: np.ndarray):
        angles = np.random.uniform(-self.angle, self.angle, 3).astype(np.float32)
        angles[1:] /= 2

        rotation_z = np.array([
            [np.cos(angles[0]), -np.sin(angles[0]), 0],
            [np.sin(angles[0]), np.cos(angles[0]), 0],
            [0, 0, 1]
        ], dtype=np.float32)

        rotation_y = np.array([
            [np.cos(angles[1]), 0, np.sin(angles[1])],
            [0, 1, 0],
            [-np.sin(angles[1]), 0, np.cos(angles[1])]
        ], dtype=np.float32)

        rotation_x = np.array([
            [1, 0, 0],
            [0, np.cos(angles[2]), -np.sin(angles[2])],
            [0, np.sin(angles[2]), np.cos(angles[2])]
        ], dtype=np.float32)

        rotation_matrix = rotation_y @ rotation_x @ rotation_z

        return points @ rotation_matrix
