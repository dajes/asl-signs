import numba
import numpy as np

from augmentation.base import Augmentation


@numba.jit(nopython=True)
def random_gradient(x, y):
    a = np.uint32(x)
    b = np.uint32(y)
    a *= 3284157443
    b ^= (a << 16) | (a >> 16)
    b *= 1911520717
    a ^= (b << 16) | (b >> 16)
    random = a * np.float32(3.14159265 / 0x80000000)
    result = np.array([np.cos(random), np.sin(random)])
    return result


@numba.jit(nopython=True)
def apply_gradients(points, scale, gamma, seed):
    unscaled = points
    if scale != 1:
        points = points * scale
    result = np.zeros_like(points)
    int_points = points.astype(np.int32)
    x_pad = np.uint32(np.uint64(seed) & 0xFFFF)
    y_pad = np.uint32(np.uint64(seed) >> 16)
    for i in range(points.shape[0]):
        gradient = random_gradient(int_points[i, 0] + x_pad, int_points[i, 1] + y_pad)
        dx = points[i, 0] - int_points[i, 0]
        dy = points[i, 1] - int_points[i, 1]
        result[i, 0] = unscaled[i, 0] + gradient[0] * dx * gamma
        result[i, 1] = unscaled[i, 1] + gradient[1] * dy * gamma
    return result


class DistortionAugmentation(Augmentation):
    def __init__(self, scale=(10, 1000), gamma=(0.001, .01)):
        self.scale = scale
        self.gamma = gamma

    def on_points(self, points):
        augmented = apply_gradients(
            points.astype(np.float32),
            np.float32(np.exp(np.random.uniform(np.log(self.scale[0]), np.log(self.scale[1])))),
            np.float32(np.exp(np.random.uniform(np.log(self.gamma[0]), np.log(self.gamma[1])))),
            np.uint64(np.random.randint(0, 2 ** 32))
        )
        return augmented
