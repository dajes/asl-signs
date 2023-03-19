import numba
import numpy as np

from augmentation.base import Augmentation


def multi_interp(x, xp, fp):
    j = np.searchsorted(xp, x, 'right').clip(0, xp.shape[0] - 1) - 1
    d = ((x - xp[j]) / (xp[j + 1] - xp[j]))[(Ellipsis,) + (None,) * (fp.ndim - 1)]

    fpj = fp[j]
    fpn = fp[j + 1]

    for i in range(10):
        m = np.isnan(fpj)
        if not np.any(m):
            break
        fpj = np.where(m, fp[(j - i).clip(0)], fpj)

    fpn = np.where(np.isnan(fpn), fpj, fpn)

    return (1 - d) * fpj + fpn * d


class TemporalAugmentation(Augmentation):
    def __init__(self, max_len: int, scale: float = 2., dropout: float = .125):
        self.max_len = max_len
        self.scale = scale
        self.dropout = dropout

    def interpolate(self, x, points):
        return multi_interp(x, np.arange(len(points)), points)

    def __call__(self, points: np.ndarray):
        n_frames = len(points)
        speed = np.random.uniform(1 / self.scale, self.scale)

        start = np.random.uniform(0, .5)
        end = n_frames - 1 - np.random.uniform(0, .5)
        m = end - start

        # make linspace not linear but raised to a slightly deviating power from approximately 1/1.2 to 1.2
        power = np.exp(np.random.uniform(-.2, .2))

        x = (np.linspace(
            0, 1,
            int(np.ceil(n_frames * speed)),
            dtype=np.float32
        ) ** power) * m + start
        if len(x) > self.max_len:
            start = np.random.randint(0, len(x) - self.max_len)
            x = x[start:start + self.max_len]

        if self.dropout > 0:
            keep_points = int(max(2, np.random.normal(len(points) * (1 - self.dropout), .45 / len(points) ** .5)))
            drop_points = len(points) - keep_points
        else:
            drop_points = 0
            # keep_points = len(points)
        if drop_points > 0:
            drops = np.random.uniform(0, 1, points.shape[0])
            threshold = np.partition(drops, drop_points - 1, 0)[drop_points - 1:drop_points]
            drops = drops > threshold
            points = points[drops]
            x = x * ((len(points) - 1) / (n_frames - 1))

        return self.interpolate(x, points)


@numba.njit()
def get_non_nans(points, fill_nans):
    non_nans = np.zeros((len(points) + 2, points.shape[1]), np.uint8)
    for i in range(points.shape[0]):
        for j in range(points.shape[1]):
            non_nan = True
            for k in range(points.shape[2]):
                if np.isnan(points[i, j, k]) and fill_nans:
                    non_nan = False
                    break
            non_nans[i, j] = non_nan
    return non_nans


@numba.njit()
def get_idxes(points, non_nans):
    idxes = np.zeros((len(points) + 3, points.shape[1]), np.int32)

    for j in range(idxes.shape[1]):
        last_i = 0
        for i in range(idxes.shape[0] - 1):
            if non_nans[i, j]:
                last_i = i
            idxes[i, j] = last_i
    return idxes


@numba.jit(nopython=True, nogil=True, fastmath=True)
def get_interpolations(points, idxes, x, fill_nans):
    result = np.empty((len(x),) + points.shape[1:], points.dtype)

    for i, t in enumerate(x):
        p1 = int(t)
        # instead of boundary checks we are sure that the "idxes" mapping contains all possible values
        # for p0, p1, p2, p3 and maps them to the nearest previous non-nan point
        p2 = idxes[p1 + 1]
        p3 = idxes[p1 + 2]
        p0 = idxes[p1 - 1]
        t = t - p1
        if fill_nans:
            p1 = idxes[p1]
        else:
            p1 = np.full(points.shape[1], np.minimum(p1, len(points) - 1), np.int32)

        tt = t * t
        ttt = tt * t

        q1 = .5 * (-ttt + 2 * tt - t)
        q2 = .5 * (3 * ttt - 5 * tt + 2)
        q3 = .5 * (-3 * ttt + 4 * tt + t)
        q4 = .5 * (ttt - tt)

        for j in range(points.shape[1]):
            result[i, j] = (points[p1[j], j] * q2 +
                            points[p2[j], j] * q3 +
                            points[p0[j], j] * q1 +
                            points[p3[j], j] * q4)
    return result


def get_interpolations2(points, idxes, x, fill_nans, momentum=1.):
    """
    Vectorized version of get_interpolations. Can not be compiled with numba, but is of similar speed anyway.

    Args:
        points: (n, d, 3) array of points
        idxes: mapping from x to the nearest previous non-nan point
        x: (m,) array of floats in [0, n-1]
        fill_nans: if True, fill nans with the nearest non-nan point
        momentum: how much the previous frame and frame after the next frame should be taken into account

    Returns:
        (m, d, 3) array of interpolated points

    """
    p1s = x.astype(np.int32)

    p2s = idxes[p1s + 1].copy()
    p3s = idxes[p1s + 2].copy()
    p0s = idxes[p1s - 1].copy()
    ts = x - np.floor(x)
    if fill_nans:
        p1s = idxes[p1s].copy()
    else:
        p1s = np.minimum(p1s, np.int32(len(points) - 1)).repeat(points.shape[1]).reshape(-1, points.shape[1])

    tts = ts * ts
    ttts = np.multiply(tts, ts)

    qs = np.empty((4,) + tts.shape, points.dtype)
    qs[0] = np.multiply(-ttts + np.multiply(tts, 2) - ts, .5)
    qs[1] = np.multiply(np.multiply(ttts, 3) - np.multiply(tts, 5) + 2, .5)
    qs[2] = np.multiply(np.multiply(ttts, -3) + np.multiply(tts, 4) + ts, .5)
    qs[3] = np.multiply(ttts - tts, .5)

    if momentum != 1:
        delta = qs[[0, 3]] * (momentum - 1)
        qs[[0, 3]] += delta
        qs[1:3] *= (1 - qs[[0, 3]].sum(0)) / qs[1:3].sum(0)

    ps = np.concatenate((p0s, p1s, p2s, p3s), 0)
    iss = ps.reshape(-1)
    jss = np.arange(ps.shape[1]).repeat(ps.shape[0]).reshape(-1, ps.shape[0]).T.reshape(-1)

    points_ = points[iss, jss].copy()
    result = np.sum(np.multiply(
        points_.reshape((4,) + p1s.shape + (points.shape[-1],)),
        qs.reshape(4, -1, 1, 1)
    ), 0)

    return result


def spline_interpolate(x, points, fill_nans, momentum=1.):
    """
    Catmull-Rom splines
    """

    # format mapping from any point id which can occur in the method to the nearest previous
    # non-nan point id
    non_nans = get_non_nans(points, fill_nans)
    idxes = get_idxes(points, non_nans)
    result = get_interpolations2(points, idxes, x, fill_nans, momentum)
    return result


class TemporalInterpolation(TemporalAugmentation):
    def interpolate(self, x, points, fill_nans=None, momentum=None):
        if fill_nans is None:
            fill_nans = np.random.rand() < .8
        if momentum is None:
            momentum = np.random.uniform(0, 3)
        return spline_interpolate(x, points, fill_nans, momentum)


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    aug = TemporalInterpolation(10)
    np.random.seed(0)
    points = np.random.randn(5 * 2 * 3).reshape(5, 2, 3)
    points[2, 0, 1:3] = np.nan

    n_frames = len(points)
    x = (np.linspace(0, 1, 100, dtype=np.float32) ** (1 / 1.5)) * (n_frames - 1)
    kwargs = dict(fill_nans=True, momentum=3)
    augmented = aug.interpolate(x, points, **kwargs)
    scattered = aug.interpolate(x[::10], points, **kwargs)

    colors = ['r', 'g', 'b']

    for i in range(points.shape[1]):
        plt.scatter(points[:, i, 0], points[:, i, 1], c=colors[i], s=10)
        plt.plot(points[:, i, 0], points[:, i, 1], c=colors[i], alpha=.15)

    for i in range(augmented.shape[1]):
        plt.plot(augmented[:, i, 0], augmented[:, i, 1], c=colors[i], alpha=.25)
        plt.scatter(scattered[:, i, 0], scattered[:, i, 1], c=colors[i], marker='x')
    plt.show()
