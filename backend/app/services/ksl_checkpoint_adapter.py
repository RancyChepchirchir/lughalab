from typing import Any, Dict

import numpy as np
import torch


POSE_DIM = 33 * 3
HAND_DIM = 21 * 3

LEFT_HAND_START = POSE_DIM
LEFT_HAND_END = (
    LEFT_HAND_START
    + HAND_DIM
)

RIGHT_HAND_START = (
    LEFT_HAND_END
)

RIGHT_HAND_END = (
    RIGHT_HAND_START
    + HAND_DIM
)

NATIVE_DIMENSION = 225
CHECKPOINT_HAND_DIMENSION = 126
CHECKPOINT_DIMENSION = 225

TARGET_LENGTH = 64


def _validate_native(
    sequence: np.ndarray,
) -> np.ndarray:
    sequence = np.asarray(
        sequence,
        dtype=np.float32,
    )

    if sequence.ndim != 2:
        raise ValueError(
            "Expected native LughaLab "
            "landmarks with shape (T, 225)."
        )

    if (
        sequence.shape[1]
        != NATIVE_DIMENSION
    ):
        raise ValueError(
            "Expected native feature "
            f"dimension {NATIVE_DIMENSION}, "
            f"got {sequence.shape[1]}."
        )

    if sequence.shape[0] == 0:
        raise ValueError(
            "Landmark sequence is empty."
        )

    return sequence


def extract_hands(
    native_sequence: np.ndarray,
) -> np.ndarray:
    """
    Convert LughaLab native representation

        [pose | left hand | right hand]

    into

        [left hand | right hand]

    producing T x 126.
    """

    sequence = _validate_native(
        native_sequence
    )

    left = sequence[
        :,
        LEFT_HAND_START:
        LEFT_HAND_END,
    ]

    right = sequence[
        :,
        RIGHT_HAND_START:
        RIGHT_HAND_END,
    ]

    hands = np.concatenate(
        [
            left,
            right,
        ],
        axis=1,
    )

    if (
        hands.shape[1]
        != CHECKPOINT_HAND_DIMENSION
    ):
        raise RuntimeError(
            "Unexpected hand "
            f"dimension: {hands.shape}"
        )

    return hands.astype(
        np.float32
    )


def pad_checkpoint_features(
    hands: np.ndarray,
) -> np.ndarray:
    """
    Match the historical KSL checkpoint layout:

        [126 hand features | 99 zeros]
    """

    hands = np.asarray(
        hands,
        dtype=np.float32,
    )

    if (
        hands.ndim != 2
        or hands.shape[1]
        != CHECKPOINT_HAND_DIMENSION
    ):
        raise ValueError(
            "Expected hand sequence "
            "with shape (T, 126)."
        )

    output = np.zeros(
        (
            hands.shape[0],
            CHECKPOINT_DIMENSION,
        ),
        dtype=np.float32,
    )

    output[
        :,
        :CHECKPOINT_HAND_DIMENSION,
    ] = hands

    return output


def temporal_resample(
    sequence: np.ndarray,
    target_length: int = TARGET_LENGTH,
) -> np.ndarray:
    """
    Diagnostic linear-resampling baseline.

    This reproduces the required sequence
    shape but is NOT claimed to reproduce
    the original dataset's temporal
    preprocessing exactly.
    """

    source_length = (
        sequence.shape[0]
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


def apply_checkpoint_zscore(
    sequence: np.ndarray,
    mean,
    std,
) -> np.ndarray:
    """
    Apply checkpoint training statistics.

    Supports either PyTorch tensors
    or NumPy arrays.
    """

    def to_numpy(
        value,
    ) -> np.ndarray:
        if isinstance(
            value,
            torch.Tensor,
        ):
            return (
                value
                .detach()
                .cpu()
                .numpy()
                .astype(
                    np.float32
                )
            )

        return np.asarray(
            value,
            dtype=np.float32,
        )

    mean_np = to_numpy(
        mean
    )

    std_np = to_numpy(
        std
    )

    if mean_np.shape != (
        CHECKPOINT_DIMENSION,
    ):
        raise ValueError(
            "Unexpected checkpoint "
            f"mean shape: {mean_np.shape}"
        )

    if std_np.shape != (
        CHECKPOINT_DIMENSION,
    ):
        raise ValueError(
            "Unexpected checkpoint "
            f"std shape: {std_np.shape}"
        )

    safe_std = np.where(
        np.abs(std_np) < 1e-8,
        1.0,
        std_np,
    )

    normalized = (
        sequence
        - mean_np
    ) / safe_std

    return normalized.astype(
        np.float32
    )


def prepare_for_legacy_checkpoint(
    native_sequence: np.ndarray,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> Dict[str, Any]:
    """
    Construct a representation compatible
    in dimensional layout with the public
    4-class KSL checkpoint.

    IMPORTANT:
    This is not proof of exact preprocessing
    equivalence with its original dataset.
    """

    native = _validate_native(
        native_sequence
    )

    hands = extract_hands(
        native
    )

    padded = (
        pad_checkpoint_features(
            hands
        )
    )

    resampled = (
        temporal_resample(
            padded,
            target_length=TARGET_LENGTH,
        )
    )

    standardized = (
        apply_checkpoint_zscore(
            resampled,
            mean=mean,
            std=std,
        )
    )

    nonzero_tail = int(
        np.count_nonzero(
            padded[
                :,
                CHECKPOINT_HAND_DIMENSION:
            ]
        )
    )

    return {
        "sequence": (
            standardized
        ),
        "native_shape": (
            list(
                native.shape
            )
        ),
        "hands_shape": (
            list(
                hands.shape
            )
        ),
        "checkpoint_shape": (
            list(
                standardized.shape
            )
        ),
        "zero_padding_verified": (
            nonzero_tail == 0
        ),
        "zero_padding_nonzero_count": (
            nonzero_tail
        ),
        "note": (
            "Compatibility representation "
            "uses left + right hand landmarks "
            "in the first 126 dimensions and "
            "99 trailing zeros before applying "
            "checkpoint z-score statistics. "
            "Temporal linear resampling is a "
            "diagnostic approximation because "
            "the public model card does not "
            "establish exact equivalence with "
            "the original temporal preprocessing."
        ),
    }