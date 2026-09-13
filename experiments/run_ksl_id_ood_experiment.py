import json
import math
import sys
from pathlib import Path
from typing import Dict, List

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

VIDEO_PATH = (
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
    / "ksl_id_ood_experiment.json"
)


def entropy(
    probabilities: List[float],
) -> float:
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


def normalized_entropy(
    probabilities: List[float],
) -> float:
    if len(
        probabilities
    ) <= 1:
        return 0.0

    return float(
        entropy(
            probabilities
        )
        / math.log(
            len(
                probabilities
            )
        )
    )


def prediction_metrics(
    result: Dict,
) -> Dict:
    probabilities = [
        item[
            "probability"
        ]
        for item
        in result[
            "predictions"
        ]
    ]

    sorted_probs = sorted(
        probabilities,
        reverse=True,
    )

    max_probability = (
        sorted_probs[0]
    )

    margin = (
        sorted_probs[0]
        - sorted_probs[1]
        if len(
            sorted_probs
        ) > 1
        else sorted_probs[0]
    )

    return {
        "max_probability": float(
            max_probability
        ),
        "prediction_margin": float(
            margin
        ),
        "entropy": (
            entropy(
                probabilities
            )
        ),
        "normalized_entropy": (
            normalized_entropy(
                probabilities
            )
        ),
    }


def get_test_directory() -> Path:
    dataset_root = Path(
        kagglehub.dataset_download(
            KAGGLE_DATASET
        )
    )

    candidate = (
        dataset_root
        / "data_split"
        / "data_split"
        / "test"
    )

    if not candidate.exists():
        raise RuntimeError(
            "Expected KSL test directory "
            "was not found:\n"
            f"{candidate}"
        )

    return candidate


def inspect_array(
    path: Path,
) -> np.ndarray:
    array = np.load(
        path,
        allow_pickle=False,
    )

    array = np.asarray(
        array,
        dtype=np.float32,
    )

    if not np.isfinite(
        array
    ).all():
        raise ValueError(
            f"Non-finite values in {path}"
        )

    return array


def prepare_dataset_sample(
    array: np.ndarray,
) -> np.ndarray:
    """
    The published dataset is expected to
    contain checkpoint-compatible landmark
    arrays.

    We deliberately validate their shape
    rather than silently reshaping them.
    """

    if array.shape == (
        64,
        225,
    ):
        return array

    raise ValueError(
        "Unexpected KSL dataset "
        f"sample shape: {array.shape}. "
        "Expected (64, 225)."
    )


def apply_checkpoint_statistics(
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

    result = (
        sequence
        - mean
    ) / safe_std

    return result.astype(
        np.float32
    )


def select_id_samples(
    test_dir: Path,
) -> List[Dict]:

    samples = []

    for label in LABELS:
        label_dir = (
            test_dir
            / label
        )

        if not label_dir.exists():
            raise RuntimeError(
                "Missing class directory: "
                f"{label_dir}"
            )

        files = sorted(
            label_dir.glob(
                "*.npy"
            )
        )

        if not files:
            raise RuntimeError(
                "No .npy files found for "
                f"class '{label}'."
            )

        samples.append(
            {
                "label": label,
                "path": files[0],
            }
        )

    return samples


def main():
    print()
    print("=" * 78)
    print(
        "LughaLab KSL "
        "ID vs OOD Experiment"
    )
    print("=" * 78)
    print()

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not VIDEO_PATH.exists():
        raise RuntimeError(
            "OOD test video not found: "
            f"{VIDEO_PATH}"
        )

    test_dir = (
        get_test_directory()
    )

    print(
        "KSL test directory:"
    )
    print(
        test_dir
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

    id_samples = (
        select_id_samples(
            test_dir
        )
    )

    id_results = []

    print(
        "IN-DISTRIBUTION SAMPLES"
    )
    print("-" * 78)

    for item in id_samples:
        expected_label = (
            item[
                "label"
            ]
        )

        path = item[
            "path"
        ]

        raw = inspect_array(
            path
        )

        print(
            f"{expected_label:<8} "
            f"raw shape={raw.shape}"
        )

        checkpoint_layout = (
            prepare_dataset_sample(
                raw
            )
        )

        checkpoint_input = (
            apply_checkpoint_statistics(
                checkpoint_layout,
                mean=mean,
                std=std,
            )
        )

        result = (
            classify_checkpoint_sequence(
                checkpoint_input
            )
        )

        metrics = (
            prediction_metrics(
                result
            )
        )

        record = {
            "source": (
                str(
                    path
                    .relative_to(
                        test_dir
                    )
                )
            ),
            "expected_label": (
                expected_label
            ),
            "predicted_label": (
                result[
                    "predicted_label"
                ]
            ),
            "correct": (
                result[
                    "predicted_label"
                ]
                == expected_label
            ),
            "probabilities": (
                result[
                    "predictions"
                ]
            ),
            **metrics,
        }

        id_results.append(
            record
        )

        print(
            f"         → "
            f"{result['predicted_label']:<8} "
            f"p={metrics['max_probability']:.4f} "
            f"margin={metrics['prediction_margin']:.4f} "
            f"Hn={metrics['normalized_entropy']:.4f}"
        )

    print()
    print(
        "OUT-OF-DISTRIBUTION CONTROL"
    )
    print("-" * 78)

    extraction = (
        extract_ksl_landmarks(
            str(
                VIDEO_PATH
            )
        )
    )

    native_sequence = (
        frames_to_array(
            extraction
        )
    )

    adapted = (
        prepare_for_legacy_checkpoint(
            native_sequence,
            mean=mean,
            std=std,
        )
    )

    ood_result = (
        classify_checkpoint_sequence(
            adapted[
                "sequence"
            ]
        )
    )

    ood_metrics = (
        prediction_metrics(
            ood_result
        )
    )

    ood_record = {
        "source": (
            str(
                VIDEO_PATH
                .relative_to(
                    ROOT
                )
            )
        ),
        "known_ksl_label": None,
        "ood_control": True,
        "frames_read": (
            extraction[
                "frames_read"
            ]
        ),
        "hand_detection_rate": (
            extraction[
                "detection_rate"
            ]
        ),
        "predicted_label": (
            ood_result[
                "predicted_label"
            ]
        ),
        "probabilities": (
            ood_result[
                "predictions"
            ]
        ),
        **ood_metrics,
        "interpretation": (
            "This external clip is used only "
            "as an out-of-distribution control. "
            "Its predicted class must not be "
            "interpreted as the true KSL gloss."
        ),
    }

    print(
        f"OOD clip → "
        f"{ood_result['predicted_label']} "
        f"p={ood_metrics['max_probability']:.4f} "
        f"margin={ood_metrics['prediction_margin']:.4f} "
        f"Hn={ood_metrics['normalized_entropy']:.4f}"
    )

    id_max_probabilities = [
        item[
            "max_probability"
        ]
        for item
        in id_results
    ]

    id_entropies = [
        item[
            "normalized_entropy"
        ]
        for item
        in id_results
    ]

    id_margins = [
        item[
            "prediction_margin"
        ]
        for item
        in id_results
    ]

    id_accuracy = float(
        np.mean(
            [
                item[
                    "correct"
                ]
                for item
                in id_results
            ]
        )
    )

    summary = {
        "id_sample_count": (
            len(
                id_results
            )
        ),
        "id_accuracy": (
            id_accuracy
        ),
        "mean_id_max_probability": float(
            np.mean(
                id_max_probabilities
            )
        ),
        "mean_id_prediction_margin": float(
            np.mean(
                id_margins
            )
        ),
        "mean_id_normalized_entropy": float(
            np.mean(
                id_entropies
            )
        ),
        "ood_max_probability": (
            ood_metrics[
                "max_probability"
            ]
        ),
        "ood_prediction_margin": (
            ood_metrics[
                "prediction_margin"
            ]
        ),
        "ood_normalized_entropy": (
            ood_metrics[
                "normalized_entropy"
            ]
        ),
        "confidence_gap": float(
            np.mean(
                id_max_probabilities
            )
            - ood_metrics[
                "max_probability"
            ]
        ),
        "margin_gap": float(
            np.mean(
                id_margins
            )
            - ood_metrics[
                "prediction_margin"
            ]
        ),
        "entropy_gap": float(
            ood_metrics[
                "normalized_entropy"
            ]
            - np.mean(
                id_entropies
            )
        ),
    }

    payload = {
        "experiment": (
            "LughaLab KSL "
            "ID vs OOD Diagnostic v1"
        ),
        "model": (
            "luciayen/"
            "afrisign-exp1-ksl-baseline"
        ),
        "dataset": (
            KAGGLE_DATASET
        ),
        "evaluation_split": (
            "data_split/data_split/test"
        ),
        "excluded_from_evaluation": (
            "dataset3"
        ),
        "supported_labels": (
            LABELS
        ),
        "id_results": (
            id_results
        ),
        "ood_result": (
            ood_record
        ),
        "summary": (
            summary
        ),
        "research_note": (
            "This experiment uses one genuine "
            "test sample from each supported "
            "KSL class and one external OOD "
            "control video. It is an exploratory "
            "closed-set reliability diagnostic, "
            "not a validated OOD benchmark."
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
        "SUMMARY"
    )
    print("-" * 78)

    print(
        "ID accuracy:",
        f"{summary['id_accuracy']:.3f}",
    )

    print(
        "Mean ID max probability:",
        f"{summary['mean_id_max_probability']:.4f}",
    )

    print(
        "OOD max probability:",
        f"{summary['ood_max_probability']:.4f}",
    )

    print(
        "Mean ID margin:",
        f"{summary['mean_id_prediction_margin']:.4f}",
    )

    print(
        "OOD margin:",
        f"{summary['ood_prediction_margin']:.4f}",
    )

    print(
        "Mean ID normalized entropy:",
        f"{summary['mean_id_normalized_entropy']:.4f}",
    )

    print(
        "OOD normalized entropy:",
        f"{summary['ood_normalized_entropy']:.4f}",
    )

    print(
        "Confidence gap:",
        f"{summary['confidence_gap']:.4f}",
    )

    print(
        "Margin gap:",
        f"{summary['margin_gap']:.4f}",
    )

    print(
        "Entropy gap:",
        f"{summary['entropy_gap']:.4f}",
    )

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )
    print()


if __name__ == "__main__":
    main()