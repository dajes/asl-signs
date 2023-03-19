import numpy as np
from matplotlib import pyplot as plt

from augmentation.compose import ComposeAugmentations
from augmentation.distortion import DistortionAugmentation
from augmentation.rotate import RotateAugmentation
from augmentation.scale import ScaleAugmentation

aug = ComposeAugmentations(
    DistortionAugmentation(),
    RotateAugmentation(),
    ScaleAugmentation(),
)

x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
xx, yy = np.meshgrid(x, y)
grid = np.stack([np.zeros((x.shape[0], x.shape[0])), xx, yy], axis=2)

xs = xx.reshape(-1)
ys = yy.reshape(-1)
colors = grid.reshape(-1, 3)
coords = np.stack([xs, ys], axis=1)

plt.scatter(coords[:, 0], -coords[:, 1], c=colors / 2 + .5)
plt.tight_layout()
plt.show()

for _ in range(3):
    augmented = aug(coords)
    plt.scatter(augmented[:, 0], -augmented[:, 1], c=colors / 2 + .5)
    plt.tight_layout()
    plt.show()
