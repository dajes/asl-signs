from typing import Union

import cv2
import mediapipe as mp
import numpy as np

from dataset.basic import BasicDataset

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic


def process_video(cap: Union[str, cv2.VideoCapture]):
    if isinstance(cap, str):
        cap = cv2.VideoCapture(cap)
    range_face = BasicDataset.idx_range_face
    range_pose = BasicDataset.idx_range_pose
    range_left_hand = BasicDataset.idx_range_hand_left
    range_right_hand = BasicDataset.idx_range_hand_right
    total = max(range_face[1], range_pose[1], range_left_hand[1], range_right_hand[1])

    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=2,
    ) as holistic:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            if results.face_landmarks is None:
                face_landmarks = np.full((range_face[1] - range_face[0], 3), np.nan)
            else:
                face_landmarks = np.array([[e.x, e.y, e.z] for e in results.face_landmarks.landmark])
                assert len(face_landmarks) == range_face[1] - range_face[0]

            if results.pose_landmarks is None:
                pose_landmarks = np.full((range_pose[1] - range_pose[0], 3), np.nan)
            else:
                pose_landmarks = np.array([[e.x, e.y, e.z] for e in results.pose_landmarks.landmark])
                assert len(pose_landmarks) == range_pose[1] - range_pose[0]

            if results.left_hand_landmarks is None:
                left_hand_landmarks = np.full((range_left_hand[1] - range_left_hand[0], 3), np.nan)
            else:
                left_hand_landmarks = np.array([[e.x, e.y, e.z] for e in results.left_hand_landmarks.landmark])
                assert len(left_hand_landmarks) == range_left_hand[1] - range_left_hand[0]

            if results.right_hand_landmarks is None:
                right_hand_landmarks = np.full((range_right_hand[1] - range_right_hand[0], 3), np.nan)
            else:
                right_hand_landmarks = np.array([[e.x, e.y, e.z] for e in results.right_hand_landmarks.landmark])
                assert len(right_hand_landmarks) == range_right_hand[1] - range_right_hand[0]

            result = np.empty((total, 3), dtype=np.float64)
            result[range_face[0]:range_face[1]] = face_landmarks
            result[range_pose[0]:range_pose[1]] = pose_landmarks
            result[range_left_hand[0]:range_left_hand[1]] = left_hand_landmarks
            result[range_right_hand[0]:range_right_hand[1]] = right_hand_landmarks
            yield result
    cap.release()
