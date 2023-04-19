from typing import Tuple

import numpy as np


class ShiftAugmentation:
    def __init__(self, ds, shift_limits: Tuple[float, float, float] = (0.05, 0.05, 0.05)):
        idx_range_face = ds.idx_range_face
        idx_range_pose = ds.idx_range_pose
        idx_range_hand_left = ds.idx_range_hand_left
        idx_range_hand_right = ds.idx_range_hand_right

        face_ids = (
                set(range(idx_range_face[0], idx_range_face[1])) |
                {idx_range_pose[0] + e for e in range(11)}
        )
        left_hand_ids = (
            set(range(idx_range_hand_left[0], idx_range_hand_left[1])) |
            {idx_range_pose[0] + e for e in range(13, 23, 2)}
        )
        right_hand_ids = (
            set(range(idx_range_hand_right[0], idx_range_hand_right[1])) |
            {idx_range_pose[0] + e for e in range(14, 23, 2)}
        )
        body_ids = set(range(idx_range_pose[0], idx_range_pose[1])) - face_ids - left_hand_ids - right_hand_ids

        clusters = [face_ids, body_ids, left_hand_ids, right_hand_ids]
        relevant_ids = ds.relevant_ids

        clusters = [[relevant_ids.index(e) for e in cluster if e in relevant_ids] for cluster in clusters]
        clusters = [cluster for cluster in clusters if cluster]
        assert len({e for c in clusters for e in c}) == sum(map(len, clusters)), 'Clusters are not disjoint'
        self.clusters = [np.array(cluster, np.int32) for cluster in clusters]
        self.shift_limits = np.array(shift_limits, np.float32).reshape((1, 1, -1))

    def __call__(self, sample):
        sample = sample.copy()
        for cluster in self.clusters:
            shift = np.random.normal(0, self.shift_limits / 3).astype(np.float32)
            sample[:, cluster] += shift
        return sample
