from pathlib import Path
from typing import Any, Dict, List

import cv2
import mediapipe as mp
import numpy as np


FEATURE_DIMENSION = 225

POSE_POINTS = 33
HAND_POINTS = 21


def _landmark_vector(
    landmarks,
    expected_points: int,
) -> List[float]:
    if landmarks is None:
        return [
            0.0
        ] * (
            expected_points * 3
        )

    values = []

    for landmark in landmarks.landmark:
        values.extend(
            [
                float(landmark.x),
                float(landmark.y),
                float(landmark.z),
            ]
        )

    expected_length = (
        expected_points * 3
    )

    if len(values) < expected_length:
        values.extend(
            [0.0]
            * (
                expected_length
                - len(values)
            )
        )

    return values[
        :expected_length
    ]


def extract_ksl_landmarks(
    video_path: str,
) -> Dict[str, Any]:
    path = Path(video_path)

    if not path.exists():
        raise ValueError(
            f"Video file does not exist: {path}"
        )

    capture = cv2.VideoCapture(
        str(path)
    )

    if not capture.isOpened():
        raise ValueError(
            f"Unable to open video: {path}"
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    if not fps or fps <= 0:
        fps = 30.0

    frames = []

    frames_read = 0
    frames_detected = 0

    mp_holistic = (
        mp.solutions.holistic
    )

    try:
        with mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as holistic:

            while True:
                success, frame = (
                    capture.read()
                )

                if not success:
                    break

                frame_index = (
                    frames_read
                )

                frames_read += 1

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                rgb.flags.writeable = (
                    False
                )

                result = (
                    holistic.process(
                        rgb
                    )
                )

                pose = (
                    _landmark_vector(
                        result.pose_landmarks,
                        POSE_POINTS,
                    )
                )

                left_hand = (
                    _landmark_vector(
                        result.left_hand_landmarks,
                        HAND_POINTS,
                    )
                )

                right_hand = (
                    _landmark_vector(
                        result.right_hand_landmarks,
                        HAND_POINTS,
                    )
                )

                vector = (
                    pose
                    + left_hand
                    + right_hand
                )

                if (
                    len(vector)
                    != FEATURE_DIMENSION
                ):
                    raise RuntimeError(
                        "Unexpected landmark "
                        f"dimension: {len(vector)}"
                    )

                detected = any(
                    abs(value) > 0.0
                    for value in (
                        left_hand
                        + right_hand
                    )
                )

                if detected:
                    frames_detected += 1

                timestamp_seconds = (
                    frame_index
                    / fps
                )

                frames.append(
                    {
                        "frame_index": (
                            frame_index
                        ),
                        "timestamp_seconds": (
                            round(
                                timestamp_seconds,
                                6,
                            )
                        ),
                        "landmarks": (
                            vector
                        ),
                    }
                )

    finally:
        capture.release()

    duration_seconds = (
        frames_read / fps
        if fps > 0
        else 0.0
    )

    detection_rate = (
        frames_detected
        / frames_read
        if frames_read
        else 0.0
    )

    return {
        "filename": path.name,
        "frames_read": (
            frames_read
        ),
        "frames_detected": (
            frames_detected
        ),
        "feature_dimension": (
            FEATURE_DIMENSION
        ),
        "duration_seconds": round(
            duration_seconds,
            6,
        ),
        "detection_rate": round(
            detection_rate,
            6,
        ),
        "frames": frames,
        "note": (
            "MediaPipe landmark representation "
            "using 33 pose landmarks, 21 left-hand "
            "landmarks and 21 right-hand landmarks. "
            "Missing landmarks are represented by "
            "zeros. Landmark extraction is a "
            "preprocessing operation and does not "
            "constitute KSL recognition."
        ),
    }


def frames_to_array(
    result: Dict[str, Any],
) -> np.ndarray:
    if not result["frames"]:
        return np.empty(
            (
                0,
                FEATURE_DIMENSION,
            ),
            dtype=np.float32,
        )

    return np.asarray(
        [
            frame["landmarks"]
            for frame
            in result["frames"]
        ],
        dtype=np.float32,
    )