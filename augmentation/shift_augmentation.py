from typing import Tuple

import numpy as np


class ShiftAugmentation:
    def __init__(self, shift_limits: Tuple[float, float, float] = (.5, .5, .5)):
        self.shift_limits = np.array(shift_limits, np.float32).reshape((1, 1, -1))

    def __call__(self, sample):
        shift = np.random.uniform(-self.shift_limits, self.shift_limits).astype(np.float32)
        return sample + shift
