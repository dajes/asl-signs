from augmentation.base import Augmentation


class ComposeAugmentations(Augmentation):
    def __init__(self, *augmentations: Augmentation):
        self.augmentations = augmentations

    def on_points(self, points):
        augmented = points
        for augmentation in self.augmentations:
            augmented = augmentation(augmented)
        return augmented
