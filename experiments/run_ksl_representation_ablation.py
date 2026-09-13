import json
import sys
from collections import defaultdict
from pathlib import Path

import kagglehub
import numpy as np


ROOT = Path(
    __file__
).resolve().parents[1]

BACKEND = ROOT / "backend"

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
BLOCK_DIM = 63
MODEL_DIM = 225

RESULTS_DIR = (
    ROOT
    / "experiments"
    / "results"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "ksl_representation_ablation.json"
)


# ============================================================
# PREPROCESSING
# ============================================================


def linear_interpolation(
    sequence: np.ndarray,
) -> np.ndarray:

    frames = sequence.shape[0]

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
        result[:, feature] = np.interp(
            new_positions,
            old_positions,
            sequence[:, feature],
        )

    return result


def pad_features(
    sequence: np.ndarray,
) -> np.ndarray:

    result = np.zeros(
        (
            TARGET_LENGTH,
            MODEL_DIM,
        ),
        dtype=np.float32,
    )

    result[:, :HAND_DIM] = sequence

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
        np.abs(std) < 1e-8,
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


# ============================================================
# DATA
# ============================================================


def load_test_samples():

    dataset_root = Path(
        kagglehub.dataset_download(
            KAGGLE_DATASET
        )
    )

    test_dir = (
        dataset_root
        / "data_split"
        / "data_split"
        / "test"
    )

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

            sequence = np.load(
                path,
                allow_pickle=False,
            ).astype(
                np.float32
            )

            if (
                sequence.ndim != 2
                or sequence.shape[1]
                != HAND_DIM
            ):
                raise ValueError(
                    f"Unexpected shape "
                    f"{sequence.shape} "
                    f"in {path}"
                )

            samples.append(
                {
                    "label": label,
                    "path": path,
                    "sequence": sequence,
                }
            )

    return samples


# ============================================================
# ABLATIONS
# ============================================================


def clean_condition(
    sequence,
    raw_mean,
):
    return sequence.copy()


def block1_removed(
    sequence,
    raw_mean,
):
    """
    Replace dimensions [0:63] with their
    training means.

    After checkpoint z-scoring these dimensions
    become approximately zero.
    """

    result = sequence.copy()

    result[
        :,
        :BLOCK_DIM
    ] = raw_mean[
        :BLOCK_DIM
    ]

    return result


def block2_removed(
    sequence,
    raw_mean,
):
    """
    Replace dimensions [63:126] with their
    training means.
    """

    result = sequence.copy()

    result[
        :,
        BLOCK_DIM:HAND_DIM
    ] = raw_mean[
        BLOCK_DIM:HAND_DIM
    ]

    return result


def block1_only(
    sequence,
    raw_mean,
):
    """
    Preserve block 1 and neutralise block 2.
    """

    result = sequence.copy()

    result[
        :,
        BLOCK_DIM:HAND_DIM
    ] = raw_mean[
        BLOCK_DIM:HAND_DIM
    ]

    return result


def block2_only(
    sequence,
    raw_mean,
):
    """
    Preserve block 2 and neutralise block 1.
    """

    result = sequence.copy()

    result[
        :,
        :BLOCK_DIM
    ] = raw_mean[
        :BLOCK_DIM
    ]

    return result


def temporal_mean_freeze(
    sequence,
    raw_mean,
):
    """
    Remove temporal dynamics while preserving
    each feature's average spatial configuration.
    """

    temporal_mean = np.mean(
        sequence,
        axis=0,
        keepdims=True,
    )

    return np.repeat(
        temporal_mean,
        TARGET_LENGTH,
        axis=0,
    ).astype(
        np.float32
    )


def first_half_only(
    sequence,
    raw_mean,
):
    """
    Preserve frames 0:32.

    Replace frames 32:64 with the training
    feature mean so the removed region becomes
    approximately zero after z-scoring.
    """

    result = sequence.copy()

    result[
        TARGET_LENGTH // 2:
    ] = raw_mean[
        :HAND_DIM
    ]

    return result


def second_half_only(
    sequence,
    raw_mean,
):
    """
    Preserve frames 32:64 and neutralise
    frames 0:32.
    """

    result = sequence.copy()

    result[
        :TARGET_LENGTH // 2
    ] = raw_mean[
        :HAND_DIM
    ]

    return result


CONDITIONS = {
    "clean": (
        clean_condition
    ),
    "block_1_removed": (
        block1_removed
    ),
    "block_2_removed": (
        block2_removed
    ),
    "block_1_only": (
        block1_only
    ),
    "block_2_only": (
        block2_only
    ),
    "temporal_mean_freeze": (
        temporal_mean_freeze
    ),
    "first_half_only": (
        first_half_only
    ),
    "second_half_only": (
        second_half_only
    ),
}


# ============================================================
# PREDICTION HELPERS
# ============================================================


def probability_for_label(
    prediction,
    label,
):

    for item in prediction[
        "predictions"
    ]:

        if item[
            "label"
        ] == label:

            return float(
                item[
                    "probability"
                ]
            )

    raise KeyError(
        f"Label {label} "
        "not found in predictions."
    )


def max_probability(
    prediction,
):

    return float(
        max(
            item[
                "probability"
            ]
            for item
            in prediction[
                "predictions"
            ]
        )
    )


# ============================================================
# EVALUATION
# ============================================================


def evaluate_condition(
    condition_name,
    condition_function,
    samples,
    checkpoint_mean,
    checkpoint_std,
    clean_predictions,
):

    raw_mean = np.asarray(
        checkpoint_mean,
        dtype=np.float32,
    )[
        :HAND_DIM
    ]

    correct = 0
    retained = 0

    true_probabilities = []
    max_probabilities = []
    true_probability_drops = []

    class_correct = defaultdict(
        int
    )

    class_total = defaultdict(
        int
    )

    records = []

    for sample in samples:

        label = sample[
            "label"
        ]

        path = sample[
            "path"
        ]

        # --------------------------------
        # temporal normalization
        # --------------------------------

        temporal = (
            linear_interpolation(
                sample[
                    "sequence"
                ]
            )
        )

        # --------------------------------
        # ablation
        # --------------------------------

        ablated = (
            condition_function(
                temporal,
                raw_mean,
            )
        )

        # --------------------------------
        # 126 -> 225
        # --------------------------------

        padded = (
            pad_features(
                ablated
            )
        )

        # --------------------------------
        # checkpoint normalization
        # --------------------------------

        prepared = (
            standardize(
                padded,
                checkpoint_mean,
                checkpoint_std,
            )
        )

        prediction = (
            classify_checkpoint_sequence(
                prepared
            )
        )

        predicted_label = (
            prediction[
                "predicted_label"
            ]
        )

        true_probability = (
            probability_for_label(
                prediction,
                label,
            )
        )

        pmax = (
            max_probability(
                prediction
            )
        )

        clean_prediction = (
            clean_predictions[
                path.name
            ]
        )

        clean_label = (
            clean_prediction[
                "predicted_label"
            ]
        )

        clean_true_probability = (
            clean_prediction[
                "true_probability"
            ]
        )

        probability_drop = (
            clean_true_probability
            - true_probability
        )

        is_correct = (
            predicted_label
            == label
        )

        prediction_retained = (
            predicted_label
            == clean_label
        )

        if is_correct:
            correct += 1
            class_correct[
                label
            ] += 1

        if prediction_retained:
            retained += 1

        class_total[
            label
        ] += 1

        true_probabilities.append(
            true_probability
        )

        max_probabilities.append(
            pmax
        )

        true_probability_drops.append(
            probability_drop
        )

        records.append(
            {
                "file": (
                    path.name
                ),
                "expected": (
                    label
                ),
                "predicted": (
                    predicted_label
                ),
                "clean_prediction": (
                    clean_label
                ),
                "correct": bool(
                    is_correct
                ),
                "prediction_retained": bool(
                    prediction_retained
                ),
                "true_class_probability": float(
                    true_probability
                ),
                "clean_true_class_probability": float(
                    clean_true_probability
                ),
                "true_probability_drop": float(
                    probability_drop
                ),
                "max_probability": float(
                    pmax
                ),
            }
        )

    total = len(
        samples
    )

    per_class = {}

    for label in LABELS:

        n = (
            class_total[
                label
            ]
        )

        c = (
            class_correct[
                label
            ]
        )

        per_class[
            label
        ] = {
            "correct": int(
                c
            ),
            "total": int(
                n
            ),
            "accuracy": float(
                c / n
                if n
                else 0.0
            ),
        }

    return {
        "condition": (
            condition_name
        ),
        "correct": int(
            correct
        ),
        "total": int(
            total
        ),
        "accuracy": float(
            correct
            / total
        ),
        "prediction_retention": float(
            retained
            / total
        ),
        "mean_true_class_probability": float(
            np.mean(
                true_probabilities
            )
        ),
        "mean_max_probability": float(
            np.mean(
                max_probabilities
            )
        ),
        "mean_true_probability_drop": float(
            np.mean(
                true_probability_drops
            )
        ),
        "median_true_probability_drop": float(
            np.median(
                true_probability_drops
            )
        ),
        "per_class": (
            per_class
        ),
        "records": (
            records
        ),
    }


# ============================================================
# CLEAN BASELINE
# ============================================================


def build_clean_predictions(
    samples,
    checkpoint_mean,
    checkpoint_std,
):

    clean_predictions = {}

    for sample in samples:

        temporal = (
            linear_interpolation(
                sample[
                    "sequence"
                ]
            )
        )

        padded = (
            pad_features(
                temporal
            )
        )

        prepared = (
            standardize(
                padded,
                checkpoint_mean,
                checkpoint_std,
            )
        )

        prediction = (
            classify_checkpoint_sequence(
                prepared
            )
        )

        true_probability = (
            probability_for_label(
                prediction,
                sample[
                    "label"
                ],
            )
        )

        clean_predictions[
            sample[
                "path"
            ].name
        ] = {
            "predicted_label": (
                prediction[
                    "predicted_label"
                ]
            ),
            "true_probability": float(
                true_probability
            ),
        }

    return clean_predictions


# ============================================================
# MAIN
# ============================================================


def main():

    print()
    print("=" * 78)
    print(
        "LughaLab KSL Representation "
        "Ablation Experiment"
    )
    print("=" * 78)
    print()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    bundle = (
        load_ksl_classifier()
    )

    checkpoint_mean = (
        bundle[
            "mean"
        ]
    )

    checkpoint_std = (
        bundle[
            "std"
        ]
    )

    samples = (
        load_test_samples()
    )

    print(
        "Clean KSL test samples:",
        len(
            samples
        ),
    )

    print()
    print(
        "Building clean reference "
        "predictions..."
    )

    clean_predictions = (
        build_clean_predictions(
            samples,
            checkpoint_mean,
            checkpoint_std,
        )
    )

    print()

    results = []

    for (
        condition_name,
        condition_function,
    ) in CONDITIONS.items():

        print(
            f"Evaluating "
            f"{condition_name}..."
        )

        result = (
            evaluate_condition(
                condition_name=(
                    condition_name
                ),
                condition_function=(
                    condition_function
                ),
                samples=(
                    samples
                ),
                checkpoint_mean=(
                    checkpoint_mean
                ),
                checkpoint_std=(
                    checkpoint_std
                ),
                clean_predictions=(
                    clean_predictions
                ),
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
            "  prediction retention:",
            f"{result['prediction_retention']:.4f}",
        )

        print(
            "  mean true-class probability:",
            f"{result['mean_true_class_probability']:.4f}",
        )

        print(
            "  mean true-probability drop:",
            f"{result['mean_true_probability_drop']:.4f}",
        )

        print()

    print()
    print(
        "SUMMARY"
    )
    print("-" * 78)

    for result in results:

        print(
            f"{result['condition']:<22} "
            f"acc="
            f"{result['accuracy']:.4f} "
            f"retain="
            f"{result['prediction_retention']:.4f} "
            f"p(true)="
            f"{result['mean_true_class_probability']:.4f} "
            f"drop="
            f"{result['mean_true_probability_drop']:.4f}"
        )

    clean_result = next(
        result
        for result
        in results
        if result[
            "condition"
        ] == "clean"
    )

    payload = {
        "experiment": (
            "LughaLab KSL "
            "Representation Ablation"
        ),
        "sample_count": int(
            len(
                samples
            )
        ),
        "baseline_accuracy": (
            clean_result[
                "accuracy"
            ]
        ),
        "feature_interpretation": {
            "block_1": (
                "dimensions 0:63 of "
                "the 126-dimensional "
                "hand representation"
            ),
            "block_2": (
                "dimensions 63:126 of "
                "the 126-dimensional "
                "hand representation"
            ),
            "warning": (
                "Blocks are intentionally "
                "not labelled left/right "
                "until ordering is verified "
                "from source preprocessing."
            ),
        },
        "results": (
            results
        ),
        "interpretation_warning": (
            "Ablations probe model dependence, "
            "not causal linguistic importance. "
            "Mean replacement creates neutral "
            "features under checkpoint z-score "
            "normalisation but may still create "
            "inputs outside the original data "
            "distribution."
        ),
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )
    print()


if __name__ == "__main__":
    main()