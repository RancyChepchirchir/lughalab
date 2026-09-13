import json
import math
import sys
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

from app.services.ksl_checkpoint_adapter import (
    prepare_for_legacy_checkpoint,
)

from app.services.ksl_landmarks import (
    extract_ksl_landmarks,
    frames_to_array,
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

EXTERNAL_VIDEO = (
    ROOT
    / "data"
    / "ksl"
    / "videos"
    / "landmark_test_01.mp4"
)

RESULTS_DIR = (
    ROOT
    / "experiments"
    / "results"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "ksl_unknown_rejection.json"
)


def entropy(
    probabilities,
):
    value = 0.0

    for probability in probabilities:
        if probability > 0.0:
            value -= (
                probability
                * math.log(
                    probability
                )
            )

    return float(
        value
    )


def prediction_metrics(
    result,
):
    probabilities = [
        item[
            "probability"
        ]
        for item
        in result[
            "predictions"
        ]
    ]

    ordered = sorted(
        probabilities,
        reverse=True,
    )

    hn = (
        entropy(
            probabilities
        )
        / math.log(
            len(
                probabilities
            )
        )
    )

    return {
        "predicted_label": (
            result[
                "predicted_label"
            ]
        ),
        "max_probability": float(
            ordered[0]
        ),
        "margin": float(
            ordered[0]
            - ordered[1]
        ),
        "normalized_entropy": float(
            hn
        ),
        "probabilities": (
            result[
                "predictions"
            ]
        ),
    }


def linear_interpolation(
    sequence,
):
    frames = (
        sequence.shape[0]
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


def pad_features(
    sequence,
):
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
    sequence,
    mean,
    std,
):
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


def prepare_raw_126(
    sequence,
    mean,
    std,
):
    temporal = (
        linear_interpolation(
            sequence
        )
    )

    padded = (
        pad_features(
            temporal
        )
    )

    return standardize(
        padded,
        mean,
        std,
    )


def load_test_samples():
    root = Path(
        kagglehub.dataset_download(
            KAGGLE_DATASET
        )
    )

    test_dir = (
        root
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
                    f"Unexpected sample shape "
                    f"{sequence.shape} in {path}"
                )

            samples.append(
                {
                    "label": label,
                    "path": path,
                    "sequence": sequence,
                }
            )

    return samples


def reverse_sequence(
    sequence,
):
    return np.flip(
        sequence,
        axis=0,
    ).copy()


def shuffle_sequence(
    sequence,
    rng,
):
    indices = np.arange(
        sequence.shape[0]
    )

    rng.shuffle(
        indices
    )

    return sequence[
        indices
    ].copy()


def freeze_sequence(
    sequence,
):
    middle = (
        sequence.shape[0]
        // 2
    )

    frame = sequence[
        middle
        : middle + 1
    ]

    return np.repeat(
        frame,
        sequence.shape[0],
        axis=0,
    ).astype(
        np.float32
    )


def add_noise(
    sequence,
    rng,
    scale=0.05,
):
    noise = rng.normal(
        loc=0.0,
        scale=scale,
        size=sequence.shape,
    ).astype(
        np.float32
    )

    return (
        sequence
        + noise
    ).astype(
        np.float32
    )


def summarize(
    records,
):
    probabilities = np.asarray(
        [
            item[
                "max_probability"
            ]
            for item
            in records
        ],
        dtype=np.float32,
    )

    margins = np.asarray(
        [
            item[
                "margin"
            ]
            for item
            in records
        ],
        dtype=np.float32,
    )

    entropies = np.asarray(
        [
            item[
                "normalized_entropy"
            ]
            for item
            in records
        ],
        dtype=np.float32,
    )

    return {
        "count": int(
            len(
                records
            )
        ),
        "mean_max_probability": float(
            np.mean(
                probabilities
            )
        ),
        "median_max_probability": float(
            np.median(
                probabilities
            )
        ),
        "p05_max_probability": float(
            np.percentile(
                probabilities,
                5,
            )
        ),
        "mean_margin": float(
            np.mean(
                margins
            )
        ),
        "p05_margin": float(
            np.percentile(
                margins,
                5,
            )
        ),
        "mean_normalized_entropy": float(
            np.mean(
                entropies
            )
        ),
        "p95_normalized_entropy": float(
            np.percentile(
                entropies,
                95,
            )
        ),
    }


def evaluate_sequences(
    samples,
    transform_name,
    transform,
    mean,
    std,
):
    records = []

    for index, sample in enumerate(
        samples
    ):
        shifted = transform(
            sample[
                "sequence"
            ],
            index,
        )

        prepared = prepare_raw_126(
            shifted,
            mean,
            std,
        )

        prediction = (
            classify_checkpoint_sequence(
                prepared
            )
        )

        metrics = prediction_metrics(
            prediction
        )

        records.append(
            {
                "source_file": (
                    sample[
                        "path"
                    ].name
                ),
                "source_label": (
                    sample[
                        "label"
                    ]
                ),
                "condition": (
                    transform_name
                ),
                **metrics,
            }
        )

    return records


def main():
    print()
    print("=" * 78)
    print(
        "LughaLab KSL "
        "Unknown-Sign Rejection Diagnostic"
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

    mean = bundle[
        "mean"
    ]

    std = bundle[
        "std"
    ]

    samples = (
        load_test_samples()
    )

    print(
        "Clean ID samples:",
        len(
            samples
        ),
    )

    rng = np.random.default_rng(
        42
    )

    clean_records = []

    correct = 0

    for sample in samples:
        prepared = prepare_raw_126(
            sample[
                "sequence"
            ],
            mean,
            std,
        )

        result = (
            classify_checkpoint_sequence(
                prepared
            )
        )

        metrics = prediction_metrics(
            result
        )

        is_correct = (
            metrics[
                "predicted_label"
            ]
            == sample[
                "label"
            ]
        )

        if is_correct:
            correct += 1

        clean_records.append(
            {
                "source_file": (
                    sample[
                        "path"
                    ].name
                ),
                "expected_label": (
                    sample[
                        "label"
                    ]
                ),
                "correct": bool(
                    is_correct
                ),
                "condition": "clean_id",
                **metrics,
            }
        )

    print(
        "Clean accuracy:",
        f"{correct}/{len(samples)} "
        f"="
        f"{correct / len(samples):.4f}",
    )

    reverse_records = (
        evaluate_sequences(
            samples=samples,
            transform_name=(
                "temporal_reverse"
            ),
            transform=(
                lambda sequence, _: (
                    reverse_sequence(
                        sequence
                    )
                )
            ),
            mean=mean,
            std=std,
        )
    )

    shuffled_records = (
        evaluate_sequences(
            samples=samples,
            transform_name=(
                "temporal_shuffle"
            ),
            transform=(
                lambda sequence, index: (
                    shuffle_sequence(
                        sequence,
                        np.random.default_rng(
                            1000
                            + index
                        ),
                    )
                )
            ),
            mean=mean,
            std=std,
        )
    )

    frozen_records = (
        evaluate_sequences(
            samples=samples,
            transform_name=(
                "static_freeze"
            ),
            transform=(
                lambda sequence, _: (
                    freeze_sequence(
                        sequence
                    )
                )
            ),
            mean=mean,
            std=std,
        )
    )

    noise_records = (
        evaluate_sequences(
            samples=samples,
            transform_name=(
                "coordinate_noise"
            ),
            transform=(
                lambda sequence, index: (
                    add_noise(
                        sequence,
                        np.random.default_rng(
                            2000
                            + index
                        ),
                        scale=0.05,
                    )
                )
            ),
            mean=mean,
            std=std,
        )
    )

    summaries = {
        "clean_id": summarize(
            clean_records
        ),
        "temporal_reverse": summarize(
            reverse_records
        ),
        "temporal_shuffle": summarize(
            shuffled_records
        ),
        "static_freeze": summarize(
            frozen_records
        ),
        "coordinate_noise": summarize(
            noise_records
        ),
    }

    external_record = None

    if EXTERNAL_VIDEO.exists():
        extraction = (
            extract_ksl_landmarks(
                str(
                    EXTERNAL_VIDEO
                )
            )
        )

        native = (
            frames_to_array(
                extraction
            )
        )

        adapted = (
            prepare_for_legacy_checkpoint(
                native,
                mean=mean,
                std=std,
            )
        )

        result = (
            classify_checkpoint_sequence(
                adapted[
                    "sequence"
                ]
            )
        )

        external_record = {
            "condition": (
                "external_video"
            ),
            "source": (
                str(
                    EXTERNAL_VIDEO
                    .relative_to(
                        ROOT
                    )
                )
            ),
            "hand_detection_rate": float(
                extraction[
                    "detection_rate"
                ]
            ),
            **prediction_metrics(
                result
            ),
            "note": (
                "External pipeline-control "
                "video. It is not assigned a "
                "true KSL label."
            ),
        }

    id_summary = (
        summaries[
            "clean_id"
        ]
    )

    thresholds = {
        "max_probability_lower_5pct": (
            id_summary[
                "p05_max_probability"
            ]
        ),
        "margin_lower_5pct": (
            id_summary[
                "p05_margin"
            ]
        ),
        "entropy_upper_95pct": (
            id_summary[
                "p95_normalized_entropy"
            ]
        ),
    }

    def rejection_rate(
        records,
    ):
        rejected = 0

        for record in records:
            reject = (
                record[
                    "max_probability"
                ]
                <
                thresholds[
                    "max_probability_lower_5pct"
                ]
                or
                record[
                    "margin"
                ]
                <
                thresholds[
                    "margin_lower_5pct"
                ]
                or
                record[
                    "normalized_entropy"
                ]
                >
                thresholds[
                    "entropy_upper_95pct"
                ]
            )

            if reject:
                rejected += 1

        return float(
            rejected
            / len(
                records
            )
        )

    rejection = {
        "clean_id": rejection_rate(
            clean_records
        ),
        "temporal_reverse": rejection_rate(
            reverse_records
        ),
        "temporal_shuffle": rejection_rate(
            shuffled_records
        ),
        "static_freeze": rejection_rate(
            frozen_records
        ),
        "coordinate_noise": rejection_rate(
            noise_records
        ),
    }

    if external_record is not None:
        external_reject = (
            external_record[
                "max_probability"
            ]
            <
            thresholds[
                "max_probability_lower_5pct"
            ]
            or
            external_record[
                "margin"
            ]
            <
            thresholds[
                "margin_lower_5pct"
            ]
            or
            external_record[
                "normalized_entropy"
            ]
            >
            thresholds[
                "entropy_upper_95pct"
            ]
        )

        rejection[
            "external_video"
        ] = bool(
            external_reject
        )

    payload = {
        "experiment": (
            "LughaLab KSL "
            "Unknown-Sign Rejection "
            "Diagnostic v1"
        ),
        "clean_accuracy": float(
            correct
            / len(
                samples
            )
        ),
        "temporal_preprocessing": (
            "linear interpolation to 64 "
            "frames; one of several "
            "near-equivalent candidates "
            "under the reconstruction audit"
        ),
        "summaries": (
            summaries
        ),
        "id_derived_thresholds": (
            thresholds
        ),
        "rejection_rates": (
            rejection
        ),
        "external_video": (
            external_record
        ),
        "methodological_warning": (
            "The shifted sequences are "
            "synthetic distribution-shift "
            "controls, not genuine unknown "
            "KSL lexical classes. Thresholds "
            "are exploratory and must not be "
            "interpreted as a validated "
            "open-set recognition system."
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
        "CONFIDENCE SUMMARY"
    )
    print("-" * 78)

    for condition, summary in (
        summaries.items()
    ):
        print(
            f"{condition:<20} "
            f"pmax="
            f"{summary['mean_max_probability']:.4f} "
            f"margin="
            f"{summary['mean_margin']:.4f} "
            f"Hn="
            f"{summary['mean_normalized_entropy']:.4f}"
        )

    print()
    print(
        "ID-DERIVED THRESHOLDS"
    )
    print("-" * 78)

    print(
        "Minimum-ish pmax:",
        f"{thresholds['max_probability_lower_5pct']:.4f}",
    )

    print(
        "Minimum-ish margin:",
        f"{thresholds['margin_lower_5pct']:.4f}",
    )

    print(
        "Maximum-ish entropy:",
        f"{thresholds['entropy_upper_95pct']:.4f}",
    )

    print()
    print(
        "REJECTION RATES"
    )
    print("-" * 78)

    for condition, value in (
        rejection.items()
    ):
        if isinstance(
            value,
            bool,
        ):
            print(
                f"{condition:<20} "
                f"{'REJECT' if value else 'ACCEPT'}"
            )

        else:
            print(
                f"{condition:<20} "
                f"{value:.4f}"
            )

    if external_record is not None:
        print()
        print(
            "EXTERNAL VIDEO"
        )
        print("-" * 78)

        print(
            "Forced class:",
            external_record[
                "predicted_label"
            ],
        )

        print(
            "Max probability:",
            f"{external_record['max_probability']:.4f}",
        )

        print(
            "Margin:",
            f"{external_record['margin']:.4f}",
        )

        print(
            "Normalized entropy:",
            f"{external_record['normalized_entropy']:.4f}",
        )

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )
    print()


if __name__ == "__main__":
    main()