from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import numpy as np


K = 5

# Provisional operating point:
# 99th percentile of held-out ID 5-NN scores.
OOD_THRESHOLD = 5.4896

ROOT = Path(__file__).resolve().parents[3]

TRAIN_BANK_PATH = (
    ROOT
    / "experiments"
    / "results"
    / "ksl_train_embedding_bank.npz"
)


@lru_cache(maxsize=1)
def load_open_set_reference() -> Dict[str, Any]:
    if not TRAIN_BANK_PATH.exists():
        raise FileNotFoundError(
            "KSL training embedding bank was not found at "
            f"{TRAIN_BANK_PATH}."
        )

    bank = np.load(
        TRAIN_BANK_PATH,
        allow_pickle=True,
    )

    embeddings = np.asarray(
        bank["embeddings"],
        dtype=np.float32,
    )

    labels = np.asarray(
        bank["labels"]
    )

    if embeddings.ndim != 2:
        raise ValueError(
            "Expected training embeddings to have "
            "shape (samples, features)."
        )

    if len(embeddings) != len(labels):
        raise ValueError(
            "Training embedding and label counts "
            "do not match."
        )

    return {
        "embeddings": embeddings,
        "labels": labels,
    }


def evaluate_open_set(
    embedding: np.ndarray,
) -> Dict[str, Any]:
    reference = load_open_set_reference()

    train_embeddings = reference[
        "embeddings"
    ]

    train_labels = reference[
        "labels"
    ]

    embedding = np.asarray(
        embedding,
        dtype=np.float32,
    ).reshape(-1)

    if embedding.shape[0] != (
        train_embeddings.shape[1]
    ):
        raise ValueError(
            "Embedding dimension mismatch: "
            f"expected "
            f"{train_embeddings.shape[1]}, "
            f"got {embedding.shape[0]}."
        )

    if not np.isfinite(
        embedding
    ).all():
        raise ValueError(
            "Embedding contains NaN or "
            "infinite values."
        )

    distances = np.linalg.norm(
        train_embeddings
        - embedding[None, :],
        axis=1,
    )

    nearest_indices = np.argsort(
        distances
    )[:K]

    nearest_distances = distances[
        nearest_indices
    ]

    nearest_labels = train_labels[
        nearest_indices
    ]

    score = float(
        nearest_distances.mean()
    )

    is_ood = (
        score > OOD_THRESHOLD
    )

    unique_labels, counts = np.unique(
        nearest_labels,
        return_counts=True,
    )

    nearest_majority_label = str(
        unique_labels[
            np.argmax(counts)
        ]
    )

    return {
        "is_ood": bool(is_ood),
        "open_set_label": (
            "unknown"
            if is_ood
            else nearest_majority_label
        ),
        "ood_score": score,
        "ood_threshold": OOD_THRESHOLD,
        "ood_detector": (
            f"{K}-NN mean embedding L2 distance"
        ),
        "nearest_known_label": (
            nearest_majority_label
        ),
        "nearest_labels": [
            str(label)
            for label in nearest_labels
        ],
        "nearest_distances": [
            float(distance)
            for distance
            in nearest_distances
        ],
        "calibration": (
            "Provisional 99th-percentile "
            "held-out ID threshold."
        ),
    }