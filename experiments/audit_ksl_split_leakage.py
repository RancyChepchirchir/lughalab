from __future__ import annotations

"""
KSL train/test leakage and redundancy audit.

Purpose
-------
The four-class public KSL baseline achieves near-perfect performance on a very
small vocabulary. Before treating that performance as evidence of robust sign
recognition, we should independently audit the clean train/test split for:

1. exact duplicated arrays across train and test;
2. duplicated samples after temporal normalization;
3. unusually high train-test nearest-neighbour similarity;
4. whether each test sample's nearest training neighbour usually shares its
   class label.

This experiment deliberately excludes any path containing "dataset3", because
the public baseline documentation reports duplicate leakage in that branch.

The audit does NOT prove signer leakage, because signer identity metadata is not
established here. It measures sample-level duplication/redundancy only.

Source representation:
    X_raw in R^(T x 126)

For similarity analysis:
    X_raw -> 64 x 126 by linear interpolation
    -> flatten to R^8064
    -> per-sample L2 normalization
    -> cosine similarity

For test sample q and training samples r_i:

    s_i = cos(q, r_i)

and

    s_max(q) = max_i s_i

High nearest-neighbour similarity is a diagnostic of possible redundancy, not
by itself proof of leakage.
"""

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import kagglehub
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

KAGGLE_DATASET = "joanwachuka/ksl-hand-landmarks"
TARGET_FRAMES = 64
RAW_FEATURES = 126
EXPECTED_LABELS = ("father", "hello", "is", "my")

RESULTS_DIR = ROOT / "experiments" / "results"

JSON_PATH = RESULTS_DIR / "ksl_split_leakage_audit.json"
SIMILARITY_PLOT_PATH = RESULTS_DIR / "ksl_train_test_nn_similarity.png"
CLASS_PLOT_PATH = RESULTS_DIR / "ksl_train_test_nn_similarity_by_class.png"
MATCH_PLOT_PATH = RESULTS_DIR / "ksl_train_test_nn_label_match.png"

# Diagnostic thresholds only. They are not universal leakage criteria.
SIMILARITY_THRESHOLDS = (0.90, 0.95, 0.98, 0.99, 0.995, 0.999)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def safe_median(values: Sequence[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def array_hash(array: np.ndarray) -> str:
    """
    Hash shape + dtype + contiguous bytes so identical arrays map identically.
    """
    arr = np.ascontiguousarray(array)
    payload = (
        str(arr.shape).encode("utf-8")
        + b"|"
        + str(arr.dtype).encode("utf-8")
        + b"|"
        + arr.tobytes()
    )
    return sha256_bytes(payload)


def rounded_array_hash(array: np.ndarray, decimals: int = 6) -> str:
    """
    Hash a rounded float32 representation to catch numerically identical
    normalized samples despite tiny floating-point differences.
    """
    arr = np.round(
        np.asarray(array, dtype=np.float32),
        decimals=decimals,
    )
    return array_hash(arr)


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------


def find_split_dirs(dataset_root: Path) -> Tuple[Path, Path]:
    """
    Locate clean train/test directories while rejecting dataset3.
    """
    candidate_pairs = [
        (
            dataset_root / "data_split" / "data_split" / "train",
            dataset_root / "data_split" / "data_split" / "test",
        ),
        (
            dataset_root / "data_split" / "train",
            dataset_root / "data_split" / "test",
        ),
        (
            dataset_root / "train",
            dataset_root / "test",
        ),
    ]

    for train_dir, test_dir in candidate_pairs:
        if _valid_split_dir(train_dir) and _valid_split_dir(test_dir):
            return train_dir, test_dir

    # Fallback recursive discovery.
    train_candidates = []
    test_candidates = []

    for path in dataset_root.rglob("*"):
        if not path.is_dir():
            continue

        lower_parts = {part.lower() for part in path.parts}
        if "dataset3" in lower_parts:
            continue

        if path.name.lower() == "train" and _valid_split_dir(path):
            train_candidates.append(path)

        if path.name.lower() == "test" and _valid_split_dir(path):
            test_candidates.append(path)

    for train_dir in train_candidates:
        for test_dir in test_candidates:
            # Prefer sibling split folders.
            if train_dir.parent == test_dir.parent:
                return train_dir, test_dir

    raise FileNotFoundError(
        f"Could not locate clean train/test split beneath {dataset_root}"
    )


def _valid_split_dir(path: Path) -> bool:
    if not path.is_dir():
        return False

    labels = {p.name for p in path.iterdir() if p.is_dir()}
    return set(EXPECTED_LABELS).issubset(labels)


def load_split(
    split_dir: Path,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []

    for label in EXPECTED_LABELS:
        class_dir = split_dir / label

        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        for path in sorted(class_dir.rglob("*.npy")):
            array = np.load(path).astype(np.float32)

            if array.ndim != 2 or array.shape[1] != RAW_FEATURES:
                raise ValueError(
                    f"{path} has shape {array.shape}; "
                    f"expected (T,{RAW_FEATURES})"
                )

            records.append(
                {
                    "label": label,
                    "path": path,
                    "raw": array,
                    "raw_hash": array_hash(array),
                }
            )

    if not records:
        raise RuntimeError(f"No .npy samples found in {split_dir}")

    return records


# ---------------------------------------------------------------------------
# Temporal normalization
# ---------------------------------------------------------------------------


def linear_resample(
    sequence: np.ndarray,
    target_frames: int = TARGET_FRAMES,
) -> np.ndarray:
    t, d = sequence.shape

    if t == target_frames:
        return sequence.astype(np.float32, copy=True)

    if t == 1:
        return np.repeat(
            sequence,
            target_frames,
            axis=0,
        ).astype(np.float32)

    old_x = np.linspace(0.0, 1.0, t)
    new_x = np.linspace(0.0, 1.0, target_frames)

    out = np.empty(
        (target_frames, d),
        dtype=np.float32,
    )

    for feature_idx in range(d):
        out[:, feature_idx] = np.interp(
            new_x,
            old_x,
            sequence[:, feature_idx],
        ).astype(np.float32)

    return out


def prepare_similarity_vector(
    raw: np.ndarray,
) -> np.ndarray:
    """
    Linear-resample -> flatten -> L2 normalize.
    """
    sequence = linear_resample(raw)
    vector = sequence.reshape(-1).astype(np.float32)

    norm = np.linalg.norm(vector)

    if norm > 0:
        vector = vector / norm

    return vector


# ---------------------------------------------------------------------------
# Exact duplicate audit
# ---------------------------------------------------------------------------


def exact_duplicate_matches(
    train_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    train_by_hash: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for record in train_records:
        train_by_hash[record["raw_hash"]].append(record)

    matches = []

    for test in test_records:
        for train in train_by_hash.get(test["raw_hash"], []):
            matches.append(
                {
                    "train_file": train["path"].name,
                    "train_label": train["label"],
                    "test_file": test["path"].name,
                    "test_label": test["label"],
                    "same_label": train["label"] == test["label"],
                    "hash": test["raw_hash"],
                }
            )

    return matches


def normalized_duplicate_matches(
    train_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Detect duplicates after deterministic 64-frame interpolation.

    Hashes are rounded to 6 decimals to avoid meaningless floating-point noise.
    """
    train_by_hash: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for record in train_records:
        normalized = linear_resample(record["raw"])
        h = rounded_array_hash(normalized, decimals=6)
        train_by_hash[h].append(record)

    matches = []

    for test in test_records:
        normalized = linear_resample(test["raw"])
        h = rounded_array_hash(normalized, decimals=6)

        for train in train_by_hash.get(h, []):
            matches.append(
                {
                    "train_file": train["path"].name,
                    "train_label": train["label"],
                    "test_file": test["path"].name,
                    "test_label": test["label"],
                    "same_label": train["label"] == test["label"],
                    "normalized_hash": h,
                }
            )

    return matches


# ---------------------------------------------------------------------------
# Nearest-neighbour redundancy audit
# ---------------------------------------------------------------------------


def nearest_neighbour_audit(
    train_records: List[Dict[str, Any]],
    test_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    print("Preparing normalized similarity vectors...")

    train_matrix = np.stack(
        [
            prepare_similarity_vector(record["raw"])
            for record in train_records
        ],
        axis=0,
    )

    test_matrix = np.stack(
        [
            prepare_similarity_vector(record["raw"])
            for record in test_records
        ],
        axis=0,
    )

    # Rows are L2 normalized, so dot product = cosine similarity.
    similarity = test_matrix @ train_matrix.T

    results = []

    for test_idx, test in enumerate(test_records):
        row = similarity[test_idx]

        nearest_idx = int(np.argmax(row))
        nearest_similarity = float(row[nearest_idx])
        nearest_train = train_records[nearest_idx]

        # Highest same-class and different-class similarities.
        same_mask = np.array(
            [
                train["label"] == test["label"]
                for train in train_records
            ],
            dtype=bool,
        )
        diff_mask = ~same_mask

        best_same = (
            float(np.max(row[same_mask]))
            if np.any(same_mask)
            else float("nan")
        )

        best_diff = (
            float(np.max(row[diff_mask]))
            if np.any(diff_mask)
            else float("nan")
        )

        margin = best_same - best_diff

        results.append(
            {
                "test_file": test["path"].name,
                "test_label": test["label"],
                "nearest_train_file": nearest_train["path"].name,
                "nearest_train_label": nearest_train["label"],
                "nearest_similarity": nearest_similarity,
                "nearest_label_match": (
                    nearest_train["label"] == test["label"]
                ),
                "best_same_class_similarity": best_same,
                "best_different_class_similarity": best_diff,
                "same_minus_different_similarity_margin": margin,
            }
        )

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def threshold_summary(
    nearest_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    similarities = np.asarray(
        [
            record["nearest_similarity"]
            for record in nearest_results
        ],
        dtype=np.float32,
    )

    summary = {}

    for threshold in SIMILARITY_THRESHOLDS:
        mask = similarities >= threshold

        summary[str(threshold)] = {
            "count": int(np.sum(mask)),
            "fraction": float(np.mean(mask)),
        }

    return summary


def main() -> None:
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Downloading/locating KSL Kaggle dataset...")
    dataset_root = Path(
        kagglehub.dataset_download(
            KAGGLE_DATASET
        )
    )

    train_dir, test_dir = find_split_dirs(
        dataset_root
    )

    print(f"Dataset root: {dataset_root}")
    print(f"Train split: {train_dir}")
    print(f"Test split:  {test_dir}")

    train_records = load_split(train_dir)
    test_records = load_split(test_dir)

    print(
        f"Loaded train={len(train_records)}, "
        f"test={len(test_records)}"
    )

    print("\nChecking exact raw-array duplicates...")
    raw_duplicates = exact_duplicate_matches(
        train_records,
        test_records,
    )

    print(
        f"Raw train/test duplicate pairs: "
        f"{len(raw_duplicates)}"
    )

    print("\nChecking duplicates after 64-frame interpolation...")
    normalized_duplicates = normalized_duplicate_matches(
        train_records,
        test_records,
    )

    print(
        f"Normalized train/test duplicate pairs: "
        f"{len(normalized_duplicates)}"
    )

    print("\nComputing train-test nearest-neighbour similarities...")
    nearest_results = nearest_neighbour_audit(
        train_records,
        test_records,
    )

    similarities = [
        record["nearest_similarity"]
        for record in nearest_results
    ]

    margins = [
        record["same_minus_different_similarity_margin"]
        for record in nearest_results
    ]

    label_matches = [
        bool(record["nearest_label_match"])
        for record in nearest_results
    ]

    per_class: Dict[str, Any] = {}

    for label in EXPECTED_LABELS:
        class_records = [
            record
            for record in nearest_results
            if record["test_label"] == label
        ]

        class_similarities = [
            record["nearest_similarity"]
            for record in class_records
        ]

        class_margins = [
            record[
                "same_minus_different_similarity_margin"
            ]
            for record in class_records
        ]

        class_matches = [
            bool(record["nearest_label_match"])
            for record in class_records
        ]

        per_class[label] = {
            "n_test": len(class_records),
            "nearest_similarity_mean": safe_mean(
                class_similarities
            ),
            "nearest_similarity_median": safe_median(
                class_similarities
            ),
            "nearest_similarity_p05": percentile(
                class_similarities,
                5,
            ),
            "nearest_similarity_p95": percentile(
                class_similarities,
                95,
            ),
            "nearest_label_match_fraction": (
                float(np.mean(class_matches))
                if class_matches
                else float("nan")
            ),
            "same_minus_different_margin_mean": safe_mean(
                class_margins
            ),
        }

    train_counts = Counter(
        record["label"]
        for record in train_records
    )

    test_counts = Counter(
        record["label"]
        for record in test_records
    )

    payload = {
        "experiment": (
            "KSL clean train/test leakage and redundancy audit"
        ),
        "dataset": KAGGLE_DATASET,
        "split_policy": (
            "Any path containing dataset3 is excluded. "
            "This follows the public baseline documentation's "
            "reported duplicate-leakage concern."
        ),
        "counts": {
            "train_total": len(train_records),
            "test_total": len(test_records),
            "train_by_class": dict(train_counts),
            "test_by_class": dict(test_counts),
        },
        "representation": {
            "raw": "(T,126)",
            "similarity_transform": (
                "(T,126) -> linear interpolation to (64,126) "
                "-> flatten -> per-sample L2 normalization"
            ),
            "similarity": "cosine similarity",
        },
        "exact_raw_duplicates": {
            "pair_count": len(raw_duplicates),
            "matches": raw_duplicates,
        },
        "normalized_duplicates": {
            "pair_count": len(normalized_duplicates),
            "rounding_decimals": 6,
            "matches": normalized_duplicates,
        },
        "nearest_neighbour": {
            "mean_similarity": safe_mean(similarities),
            "median_similarity": safe_median(similarities),
            "p01_similarity": percentile(
                similarities,
                1,
            ),
            "p05_similarity": percentile(
                similarities,
                5,
            ),
            "p95_similarity": percentile(
                similarities,
                95,
            ),
            "p99_similarity": percentile(
                similarities,
                99,
            ),
            "nearest_label_match_fraction": float(
                np.mean(label_matches)
            ),
            "same_minus_different_margin_mean": safe_mean(
                margins
            ),
            "threshold_counts": threshold_summary(
                nearest_results
            ),
            "by_class": per_class,
            "samples": nearest_results,
        },
        "interpretation_warning": (
            "Exact or near-duplicate evidence is relevant to sample-level "
            "split integrity. High cosine similarity alone is not proof of "
            "data leakage, signer leakage, or invalid evaluation. Signer "
            "identity is not established by this audit."
        ),
    }

    JSON_PATH.write_text(
        json.dumps(
            payload,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------

    plt.figure(
        figsize=(9, 5)
    )
    plt.hist(
        similarities,
        bins=20,
    )
    plt.xlabel(
        "Cosine similarity to nearest training sample"
    )
    plt.ylabel(
        "Number of test samples"
    )
    plt.title(
        "KSL Clean Test Split — Nearest Training Similarity"
    )
    plt.tight_layout()
    plt.savefig(
        SIMILARITY_PLOT_PATH,
        dpi=180,
    )
    plt.close()

    class_values = [
        [
            record["nearest_similarity"]
            for record in nearest_results
            if record["test_label"] == label
        ]
        for label in EXPECTED_LABELS
    ]

    plt.figure(
        figsize=(9, 5)
    )
    plt.boxplot(
        class_values,
        labels=EXPECTED_LABELS,
    )
    plt.ylabel(
        "Cosine similarity to nearest training sample"
    )
    plt.xlabel(
        "True class"
    )
    plt.title(
        "KSL Train-Test Nearest-Neighbour Similarity by Class"
    )
    plt.tight_layout()
    plt.savefig(
        CLASS_PLOT_PATH,
        dpi=180,
    )
    plt.close()

    match_fractions = [
        per_class[label][
            "nearest_label_match_fraction"
        ]
        for label in EXPECTED_LABELS
    ]

    plt.figure(
        figsize=(8, 5)
    )
    plt.bar(
        EXPECTED_LABELS,
        match_fractions,
    )
    plt.ylim(
        0.0,
        1.05,
    )
    plt.xlabel(
        "True class"
    )
    plt.ylabel(
        "Fraction whose nearest train sample has same label"
    )
    plt.title(
        "KSL Nearest-Training-Label Agreement"
    )
    plt.tight_layout()
    plt.savefig(
        MATCH_PLOT_PATH,
        dpi=180,
    )
    plt.close()

    # -------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("KSL SPLIT LEAKAGE / REDUNDANCY AUDIT COMPLETE")
    print("=" * 72)

    print(
        f"Train samples: {len(train_records)}"
    )
    print(
        f"Test samples:  {len(test_records)}"
    )

    print(
        f"Exact raw train/test duplicate pairs: "
        f"{len(raw_duplicates)}"
    )

    print(
        f"Duplicates after temporal normalization: "
        f"{len(normalized_duplicates)}"
    )

    print(
        f"Nearest-train cosine similarity: "
        f"mean={safe_mean(similarities):.6f}, "
        f"median={safe_median(similarities):.6f}, "
        f"p95={percentile(similarities, 95):.6f}, "
        f"p99={percentile(similarities, 99):.6f}"
    )

    print(
        f"Nearest-train label match: "
        f"{sum(label_matches)}/{len(label_matches)} "
        f"= {float(np.mean(label_matches)):.4f}"
    )

    print(
        f"Mean same-class minus different-class "
        f"nearest-similarity margin: "
        f"{safe_mean(margins):.6f}"
    )

    print("\nThreshold counts:")

    thresholds = threshold_summary(
        nearest_results
    )

    for threshold in SIMILARITY_THRESHOLDS:
        stats = thresholds[str(threshold)]

        print(
            f"  similarity >= {threshold:.3f}: "
            f"{stats['count']}/{len(test_records)} "
            f"= {stats['fraction']:.4f}"
        )

    print("\nPer-class summary:")

    for label in EXPECTED_LABELS:
        stats = per_class[label]

        print(
            f"  {label:<6} "
            f"mean_nn={stats['nearest_similarity_mean']:.6f}, "
            f"label_match="
            f"{stats['nearest_label_match_fraction']:.4f}, "
            f"margin="
            f"{stats['same_minus_different_margin_mean']:.6f}"
        )

    if raw_duplicates:
        print(
            "\nWARNING: exact train/test raw duplicates were found."
        )

    elif normalized_duplicates:
        print(
            "\nWARNING: train/test duplicates emerged after temporal "
            "normalization."
        )

    else:
        print(
            "\nNo exact train/test duplicates were found by the two "
            "tested equality checks."
        )

    print("\nOutputs:")

    for path in (
        JSON_PATH,
        SIMILARITY_PLOT_PATH,
        CLASS_PLOT_PATH,
        MATCH_PLOT_PATH,
    ):
        print(
            f"  {path.relative_to(ROOT)}"
        )

    print(
        "\nInterpretation: this audit addresses sample-level duplication "
        "and redundancy. It cannot establish signer independence without "
        "verified signer metadata."
    )


if __name__ == "__main__":
    main()