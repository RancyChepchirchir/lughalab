from typing import Any, Dict, Tuple

import numpy as np


SEQUENCE_LENGTH = 64

POSE_POINTS = 33
HAND_POINTS = 21

TOTAL_POINTS = (
    POSE_POINTS
    + HAND_POINTS
    + HAND_POINTS
)

COORDINATES = 3

FEATURE_DIMENSION = (
    TOTAL_POINTS
    * COORDINATES
)

LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12


def _reshape(
    sequence: np.ndarray,
) -> np.ndarray:
    """
    Convert:

        (T, 225)

    into:

        (T, 75, 3)
    """

    if sequence.ndim != 2:
        raise ValueError(
            "Expected landmark sequence "
            "with shape (T, 225)."
        )

    if (
        sequence.shape[1]
        != FEATURE_DIMENSION
    ):
        raise ValueError(
            "Expected feature dimension "
            f"{FEATURE_DIMENSION}, got "
            f"{sequence.shape[1]}."
        )

    return sequence.reshape(
        sequence.shape[0],
        TOTAL_POINTS,
        COORDINATES,
    )


def _valid_point(
    point: np.ndarray,
) -> bool:
    return bool(
        np.any(
            np.abs(point) > 1e-8
        )
    )


def normalize_geometry(
    sequence: np.ndarray,
) -> Tuple[
    np.ndarray,
    Dict[str, Any],
]:
    """
    Centre each frame around the shoulder
    midpoint and scale by shoulder width.

    Missing landmarks remain zero.
    """

    points = _reshape(
        sequence
    ).copy()

    normalized = np.zeros_like(
        points,
        dtype=np.float32,
    )

    valid_frames = 0
    fallback_frames = 0

    scales = []

    for t in range(
        points.shape[0]
    ):
        frame = points[t]

        left = frame[
            LEFT_SHOULDER
        ]

        right = frame[
            RIGHT_SHOULDER
        ]

        left_valid = _valid_point(
            left
        )

        right_valid = _valid_point(
            right
        )

        if (
            left_valid
            and right_valid
        ):
            centre = (
                left + right
            ) / 2.0

            scale = float(
                np.linalg.norm(
                    left[:2]
                    - right[:2]
                )
            )

            if scale < 1e-6:
                scale = 1.0
                fallback_frames += 1
            else:
                valid_frames += 1

        else:
            centre = np.zeros(
                3,
                dtype=np.float32,
            )

            scale = 1.0

            fallback_frames += 1

        scales.append(
            scale
        )

        valid_mask = np.any(
            np.abs(frame) > 1e-8,
            axis=1,
        )

        transformed = (
            frame - centre
        ) / scale

        transformed[
            ~valid_mask
        ] = 0.0

        normalized[t] = (
            transformed
        )

    flattened = (
        normalized.reshape(
            normalized.shape[0],
            FEATURE_DIMENSION,
        )
    )

    diagnostics = {
        "frames": int(
            sequence.shape[0]
        ),

        "geometry_normalised_frames": (
            valid_frames
        ),

        "geometry_fallback_frames": (
            fallback_frames
        ),

        "mean_scale": (
            float(
                np.mean(scales)
            )
            if scales
            else None
        ),
    }

    return (
        flattened,
        diagnostics,
    )


def temporal_interpolate(
    sequence: np.ndarray,
    target_length: int = (
        SEQUENCE_LENGTH
    ),
) -> np.ndarray:
    """
    Linearly interpolate an arbitrary
    T x D sequence to target_length x D.
    """

    if sequence.ndim != 2:
        raise ValueError(
            "Expected a 2D sequence."
        )

    source_length = (
        sequence.shape[0]
    )

    if source_length == 0:
        raise ValueError(
            "Cannot interpolate an "
            "empty landmark sequence."
        )

    if source_length == 1:
        return np.repeat(
            sequence,
            target_length,
            axis=0,
        ).astype(
            np.float32
        )

    source_positions = (
        np.linspace(
            0.0,
            1.0,
            source_length,
        )
    )

    target_positions = (
        np.linspace(
            0.0,
            1.0,
            target_length,
        )
    )

    output = np.empty(
        (
            target_length,
            sequence.shape[1],
        ),
        dtype=np.float32,
    )

    for feature in range(
        sequence.shape[1]
    ):
        output[
            :,
            feature
        ] = np.interp(
            target_positions,
            source_positions,
            sequence[
                :,
                feature
            ],
        )

    return output


def calculate_missingness(
    sequence: np.ndarray,
) -> Dict[str, float]:
    points = _reshape(
        sequence
    )

    pose = points[
        :,
        :POSE_POINTS,
        :
    ]

    left = points[
        :,
        POSE_POINTS:
        POSE_POINTS + HAND_POINTS,
        :
    ]

    right = points[
        :,
        POSE_POINTS + HAND_POINTS:,
        :
    ]

    def missing_rate(
        block: np.ndarray,
    ) -> float:
        missing = np.all(
            np.abs(block) <= 1e-8,
            axis=2,
        )

        return float(
            np.mean(missing)
        )

    return {
        "pose_missing_rate": (
            missing_rate(
                pose
            )
        ),

        "left_hand_missing_rate": (
            missing_rate(
                left
            )
        ),

        "right_hand_missing_rate": (
            missing_rate(
                right
            )
        ),
    }


def preprocess_landmarks(
    sequence: np.ndarray,
    target_length: int = (
        SEQUENCE_LENGTH
    ),
) -> Dict[str, Any]:
    sequence = np.asarray(
        sequence,
        dtype=np.float32,
    )

    original_shape = list(
        sequence.shape
    )

    missing_before = (
        calculate_missingness(
            sequence
        )
    )

    normalized, geometry = (
        normalize_geometry(
            sequence
        )
    )

    interpolated = (
        temporal_interpolate(
            normalized,
            target_length=(
                target_length
            ),
        )
    )

    missing_after = (
        calculate_missingness(
            interpolated
        )
    )

    return {
        "sequence": (
            interpolated
        ),

        "original_shape": (
            original_shape
        ),

        "output_shape": list(
            interpolated.shape
        ),

        "target_length": (
            target_length
        ),

        "missingness_before": (
            missing_before
        ),

        "missingness_after": (
            missing_after
        ),

        "geometry": (
            geometry
        ),
    }