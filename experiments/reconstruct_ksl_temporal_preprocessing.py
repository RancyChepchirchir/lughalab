import json
import sys
from collections import defaultdict
from pathlib import Path

import kagglehub
import numpy as np


ROOT = Path(
    __file__
).resolve().parents[1]

BACKEND = (
    ROOT
    / "backend"
)

sys.path.insert(
    0,
    str(BACKEND),
)


from app.services.ksl_classifier import (
    classify_checkpoint_sequence,
    load_ksl_classifier,
)


KAGGLE_DATASET = (
    "joanwachuka/"
    "ksl-hand-landmarks"
)

LABELS = [
    "father",
    "hello",
    "is",
    "my",
]

TARGET_LENGTH = 64
HAND_DIM = 126
MODEL_DIM = 225

RESULTS_DIR = (
    ROOT
    / "experiments"
    / "results"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "ksl_temporal_preprocessing_reconstruction.json"
)


def linear_interpolation(
    sequence: np.ndarray,
) -> np.ndarray:
    """
    Resample a T x 126 sequence to
    64 frames using linear interpolation
    independently for every feature.
    """

    frames = sequence.shape[0]

    if frames == TARGET_LENGTH:
        return sequence.astype(
            np.float32
        )

    old_positions = np.linspace(
        0.0,
        1.0,
        frames,
        dtype=np.float32,
    )

    new_positions = np.linspace(
        0.0,
        1.0,
        TARGET_LENGTH,
        dtype=np.float32,
    )

    result = np.empty(
        (
            TARGET_LENGTH,
            HAND_DIM,
        ),
        dtype=np.float32,
    )

    for feature in range(
        HAND_DIM
    ):
        result[
            :,
            feature
        ] = np.interp(
            new_positions,
            old_positions,
            sequence[
                :,
                feature
            ],
        )

    return result


def nearest_resample(
    sequence: np.ndarray,
) -> np.ndarray:
    """
    Uniform nearest-neighbour temporal
    resampling to 64 frames.
    """

    frames = sequence.shape[0]

    indices = np.linspace(
        0,
        frames - 1,
        TARGET_LENGTH,
    )

    indices = np.rint(
        indices
    ).astype(
        np.int64
    )

    return sequence[
        indices
    ].astype(
        np.float32
    )


def floor_resample(
    sequence: np.ndarray,
) -> np.ndarray:
    """
    Uniform temporal sampling using
    floor-index selection.
    """

    frames = sequence.shape[0]

    indices = np.linspace(
        0,
        frames - 1,
        TARGET_LENGTH,
    )

    indices = np.floor(
        indices
    ).astype(
        np.int64
    )

    return sequence[
        indices
    ].astype(
        np.float32
    )


def repeat_last_padding(
    sequence: np.ndarray,
) -> np.ndarray:
    """
    Keep all original frames and repeat
    the final frame until length 64.

    No truncation is currently needed
    because the audited source clips are
    all shorter than 64 frames.
    """

    frames = sequence.shape[0]

    if frames >= TARGET_LENGTH:
        return sequence[
            :TARGET_LENGTH
        ].astype(
            np.float32
        )

    missing = (
        TARGET_LENGTH
        - frames
    )

    tail = np.repeat(
        sequence[
            -1:
        ],
        missing,
        axis=0,
    )

    return np.concatenate(
        [
            sequence,
            tail,
        ],
        axis=0,
    ).astype(
        np.float32
    )


def zero_padding(
    sequence: np.ndarray,
) -> np.ndarray:
    """
    Keep original frames and append
    zero frames until length 64.
    """

    frames = sequence.shape[0]

    if frames >= TARGET_LENGTH:
        return sequence[
            :TARGET_LENGTH
        ].astype(
            np.float32
        )

    result = np.zeros(
        (
            TARGET_LENGTH,
            HAND_DIM,
        ),
        dtype=np.float32,
    )

    result[
        :frames
    ] = sequence

    return result


METHODS = {
    "linear_interpolation": (
        linear_interpolation
    ),
    "nearest_resample": (
        nearest_resample
    ),
    "floor_resample": (
        floor_resample
    ),
    "repeat_last_padding": (
        repeat_last_padding
    ),
    "zero_padding": (
        zero_padding
    ),
}


def pad_features(
    sequence: np.ndarray,
) -> np.ndarray:
    """
    Convert 64 x 126 hand landmarks
    into the checkpoint's 64 x 225
    representation.
    """

    if sequence.shape != (
        TARGET_LENGTH,
        HAND_DIM,
    ):
        raise ValueError(
            "Expected hand sequence "
            f"(64, 126), got "
            f"{sequence.shape}."
        )

    result = np.zeros(
        (
            TARGET_LENGTH,
            MODEL_DIM,
        ),
        dtype=np.float32,
    )

    result[
        :,
        :HAND_DIM
    ] = sequence

    return result


def standardize(
    sequence: np.ndarray,
    mean,
    std,
) -> np.ndarray:
    mean = np.asarray(
        mean,
        dtype=np.float32,
    )

    std = np.asarray(
        std,
        dtype=np.float32,
    )

    safe_std = np.where(
        np.abs(
            std
        ) < 1e-8,
        1.0,
        std,
    )

    return (
        (
            sequence
            - mean
        )
        / safe_std
    ).astype(
        np.float32
    )


def get_test_directory() -> Path:
    dataset_root = Path(
        kagglehub.dataset_download(
            KAGGLE_DATASET
        )
    )

    path = (
        dataset_root
        / "data_split"
        / "data_split"
        / "test"
    )

    if not path.exists():
        raise RuntimeError(
            "Could not locate test split: "
            f"{path}"
        )

    return path


def load_test_samples(
    test_dir: Path,
):
    samples = []

    for label in LABELS:
        label_dir = (
            test_dir
            / label
        )

        for path in sorted(
            label_dir.glob(
                "*.npy"
            )
        ):
            array = np.load(
                path,
                allow_pickle=False,
            )

            array = np.asarray(
                array,
                dtype=np.float32,
            )

            if array.ndim != 2:
                raise ValueError(
                    f"Unexpected ndim in "
                    f"{path}: {array.ndim}"
                )

            if array.shape[1] != (
                HAND_DIM
            ):
                raise ValueError(
                    f"Unexpected feature "
                    f"dimension in {path}: "
                    f"{array.shape}"
                )

            if not np.isfinite(
                array
            ).all():
                raise ValueError(
                    f"Non-finite data: "
                    f"{path}"
                )

            samples.append(
                {
                    "label": label,
                    "path": path,
                    "sequence": array,
                }
            )

    return samples


def evaluate_method(
    method_name,
    method,
    samples,
    mean,
    std,
):
    correct = 0

    class_correct = defaultdict(
        int
    )

    class_total = defaultdict(
        int
    )

    confidence_values = []

    errors = []

    for sample in samples:
        expected = sample[
            "label"
        ]

        raw = sample[
            "sequence"
        ]

        temporal = method(
            raw
        )

        padded = pad_features(
            temporal
        )

        checkpoint_input = (
            standardize(
                padded,
                mean=mean,
                std=std,
            )
        )

        result = (
            classify_checkpoint_sequence(
                checkpoint_input
            )
        )

        predicted = result[
            "predicted_label"
        ]

        probability = result[
            "predicted_probability"
        ]

        class_total[
            expected
        ] += 1

        confidence_values.append(
            probability
        )

        is_correct = (
            predicted
            == expected
        )

        if is_correct:
            correct += 1
            class_correct[
                expected
            ] += 1
        else:
            errors.append(
                {
                    "file": str(
                        sample[
                            "path"
                        ].name
                    ),
                    "expected": (
                        expected
                    ),
                    "predicted": (
                        predicted
                    ),
                    "probability": float(
                        probability
                    ),
                    "raw_frames": int(
                        raw.shape[0]
                    ),
                }
            )

    total = len(
        samples
    )

    accuracy = (
        correct
        / total
        if total
        else 0.0
    )

    per_class = {}

    for label in LABELS:
        total_label = (
            class_total[
                label
            ]
        )

        correct_label = (
            class_correct[
                label
            ]
        )

        per_class[
            label
        ] = {
            "correct": int(
                correct_label
            ),
            "total": int(
                total_label
            ),
            "accuracy": float(
                correct_label
                / total_label
                if total_label
                else 0.0
            ),
        }

    return {
        "method": (
            method_name
        ),
        "correct": int(
            correct
        ),
        "total": int(
            total
        ),
        "accuracy": float(
            accuracy
        ),
        "mean_max_probability": float(
            np.mean(
                confidence_values
            )
        ),
        "per_class": (
            per_class
        ),
        "error_count": len(
            errors
        ),
        "errors": (
            errors
        ),
    }


def main():
    print()
    print("=" * 78)
    print(
        "LughaLab KSL Temporal "
        "Preprocessing Reconstruction"
    )
    print("=" * 78)
    print()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    test_dir = (
        get_test_directory()
    )

    samples = (
        load_test_samples(
            test_dir
        )
    )

    print(
        "Test samples:",
        len(
            samples
        ),
    )

    print(
        "Raw frame range:",
        min(
            sample[
                "sequence"
            ].shape[0]
            for sample
            in samples
        ),
        "to",
        max(
            sample[
                "sequence"
            ].shape[0]
            for sample
            in samples
        ),
    )

    print()

    bundle = (
        load_ksl_classifier()
    )

    mean = bundle[
        "mean"
    ]

    std = bundle[
        "std"
    ]

    results = []

    for (
        method_name,
        method,
    ) in METHODS.items():

        print(
            f"Testing "
            f"{method_name}..."
        )

        result = (
            evaluate_method(
                method_name=(
                    method_name
                ),
                method=method,
                samples=samples,
                mean=mean,
                std=std,
            )
        )

        results.append(
            result
        )

        print(
            f"  accuracy: "
            f"{result['correct']}/"
            f"{result['total']} "
            f"="
            f"{result['accuracy']:.4f}"
        )

        print(
            "  mean max probability:",
            f"{result['mean_max_probability']:.4f}",
        )

        print()

    results = sorted(
        results,
        key=lambda item: (
            item[
                "accuracy"
            ],
            item[
                "mean_max_probability"
            ],
        ),
        reverse=True,
    )

    print(
        "RANKING"
    )
    print("-" * 78)

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"{rank}. "
            f"{result['method']:<22} "
            f"accuracy="
            f"{result['accuracy']:.4f} "
            f"("
            f"{result['correct']}/"
            f"{result['total']}"
            f") "
            f"confidence="
            f"{result['mean_max_probability']:.4f}"
        )

    best = results[
        0
    ]

    payload = {
        "experiment": (
            "LughaLab KSL Temporal "
            "Preprocessing Reconstruction"
        ),
        "dataset": (
            KAGGLE_DATASET
        ),
        "split": (
            "data_split/data_split/test"
        ),
        "sample_count": len(
            samples
        ),
        "target_frames": (
            TARGET_LENGTH
        ),
        "raw_feature_dimension": (
            HAND_DIM
        ),
        "checkpoint_feature_dimension": (
            MODEL_DIM
        ),
        "candidate_results": (
            results
        ),
        "best_candidate": (
            best[
                "method"
            ]
        ),
        "best_accuracy": (
            best[
                "accuracy"
            ]
        ),
        "interpretation": (
            "Candidate temporal preprocessing "
            "methods are compared by their "
            "ability to reproduce checkpoint "
            "predictions on the clean original "
            "KSL test split. Matching the "
            "published accuracy would provide "
            "strong evidence, but not absolute "
            "proof, that the candidate matches "
            "the original training pipeline."
        ),
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        "Best candidate:",
        best[
            "method"
        ],
    )

    print(
        "Best accuracy:",
        f"{best['accuracy']:.4f}",
    )

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )
    print()


if __name__ == "__main__":
    main()