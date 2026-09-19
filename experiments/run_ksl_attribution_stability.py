from __future__ import annotations

"""
KSL attribution stability under plausible temporal preprocessing choices.

Why this experiment exists
--------------------------
Earlier LughaLab reconstruction showed that several non-zero temporal transforms
reproduce the clean four-class KSL test accuracy almost identically:

- linear interpolation
- nearest-neighbour resampling
- floor-index resampling
- repeat-last padding

That means downstream attribution should not be interpreted from only one
preprocessing choice without testing robustness.

This experiment computes joint-level occlusion attribution under each candidate
temporal transform and measures how stable the resulting 42-joint importance
profiles are.

For true class y and joint j:

    A_j = p_theta(y | X) - p_theta(y | X^(-j))

A joint is neutralized by replacing its xyz coordinates at every frame with
the checkpoint training mean. After checkpoint z-score normalization, those
features become approximately zero.

Stability metrics:
- Pearson correlation of attribution magnitudes
- Spearman rank correlation
- Top-k Jaccard overlap
- Top-k intersection counts

This remains a model-sensitivity analysis. Stable attribution does not establish
linguistic causality, anatomical importance, or human-grounded KSL saliency.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Repository imports
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"

for path in (ROOT, BACKEND):
    p = str(path)
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.app.services.ksl_classifier import (  # noqa: E402
    classify_checkpoint_sequence,
    load_ksl_classifier,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

KAGGLE_DATASET = "joanwachuka/ksl-hand-landmarks"

TARGET_FRAMES = 64
RAW_FEATURES = 126
MODEL_FEATURES = 225
N_JOINTS = 42
COORDS_PER_JOINT = 3

EXPECTED_LABELS = ("father", "hello", "is", "my")

METHODS = (
    "linear_interpolation",
    "nearest_resample",
    "floor_resample",
    "repeat_last_padding",
)

TOP_K_VALUES = (5, 10, 15)

RESULTS_DIR = ROOT / "experiments" / "results"
JSON_PATH = RESULTS_DIR / "ksl_attribution_stability.json"
CSV_PATH = RESULTS_DIR / "ksl_attribution_stability.csv"
CORR_PLOT_PATH = RESULTS_DIR / "ksl_attribution_stability_correlation.png"
PROFILE_PLOT_PATH = RESULTS_DIR / "ksl_attribution_stability_profiles.png"
TOPK_PLOT_PATH = RESULTS_DIR / "ksl_attribution_stability_topk.png"


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)


def safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def top_indices(values: np.ndarray, k: int) -> List[int]:
    k = min(k, len(values))
    return np.argsort(values)[::-1][:k].astype(int).tolist()


def average_ranks(values: np.ndarray) -> np.ndarray:
    """
    Average ranks for ties, ascending convention.
    Implemented locally to avoid adding scipy as a dependency.
    """
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]

    ranks = np.empty(len(values), dtype=np.float64)

    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1

        # Ranks are 1-based; ties receive average rank.
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j

    return ranks


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")

    return float(np.corrcoef(a, b)[0, 1])


def spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    return pearson_corr(average_ranks(a), average_ranks(b))


def jaccard_topk(a: np.ndarray, b: np.ndarray, k: int) -> Tuple[float, int]:
    set_a = set(top_indices(a, k))
    set_b = set(top_indices(b, k))

    union = set_a | set_b
    intersection = set_a & set_b

    score = len(intersection) / len(union) if union else 1.0
    return float(score), len(intersection)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def find_clean_test_dir(dataset_root: Path) -> Path:
    candidates = [
        dataset_root / "data_split" / "data_split" / "test",
        dataset_root / "data_split" / "test",
        dataset_root / "test",
    ]

    for candidate in candidates:
        if candidate.is_dir():
            labels = {p.name for p in candidate.iterdir() if p.is_dir()}
            if set(EXPECTED_LABELS).issubset(labels):
                return candidate

    for candidate in dataset_root.rglob("test"):
        if not candidate.is_dir():
            continue

        if "dataset3" in {part.lower() for part in candidate.parts}:
            continue

        labels = {p.name for p in candidate.iterdir() if p.is_dir()}
        if set(EXPECTED_LABELS).issubset(labels):
            return candidate

    raise FileNotFoundError(
        f"Could not locate clean KSL test split beneath {dataset_root}"
    )


def load_clean_test_samples(test_dir: Path) -> List[Tuple[str, Path, np.ndarray]]:
    samples: List[Tuple[str, Path, np.ndarray]] = []

    for label in EXPECTED_LABELS:
        class_dir = test_dir / label

        for path in sorted(class_dir.rglob("*.npy")):
            arr = np.load(path).astype(np.float32)

            if arr.ndim != 2 or arr.shape[1] != RAW_FEATURES:
                raise ValueError(
                    f"{path} has shape {arr.shape}; expected (T,{RAW_FEATURES})"
                )

            samples.append((label, path, arr))

    if not samples:
        raise RuntimeError("No clean KSL test samples found.")

    return samples


# ---------------------------------------------------------------------------
# Temporal reconstruction candidates
# ---------------------------------------------------------------------------


def linear_interpolation(sequence: np.ndarray) -> np.ndarray:
    t, d = sequence.shape

    if t == TARGET_FRAMES:
        return sequence.astype(np.float32, copy=True)

    if t == 1:
        return np.repeat(sequence, TARGET_FRAMES, axis=0).astype(np.float32)

    old_x = np.linspace(0.0, 1.0, t)
    new_x = np.linspace(0.0, 1.0, TARGET_FRAMES)

    out = np.empty((TARGET_FRAMES, d), dtype=np.float32)

    for j in range(d):
        out[:, j] = np.interp(
            new_x,
            old_x,
            sequence[:, j],
        ).astype(np.float32)

    return out


def nearest_resample(sequence: np.ndarray) -> np.ndarray:
    t = sequence.shape[0]

    if t == TARGET_FRAMES:
        return sequence.astype(np.float32, copy=True)

    positions = np.linspace(0, t - 1, TARGET_FRAMES)
    indices = np.rint(positions).astype(int)

    return sequence[indices].astype(np.float32, copy=True)


def floor_resample(sequence: np.ndarray) -> np.ndarray:
    t = sequence.shape[0]

    if t == TARGET_FRAMES:
        return sequence.astype(np.float32, copy=True)

    positions = np.linspace(0, t - 1, TARGET_FRAMES)
    indices = np.floor(positions).astype(int)
    indices = np.clip(indices, 0, t - 1)

    return sequence[indices].astype(np.float32, copy=True)


def repeat_last_padding(sequence: np.ndarray) -> np.ndarray:
    """
    If shorter than 64, keep observed frames in order and repeat the final
    frame. If longer than 64, truncate.

    In the audited source test split T <= 43, so padding is the relevant case.
    """
    t, d = sequence.shape

    if t >= TARGET_FRAMES:
        return sequence[:TARGET_FRAMES].astype(np.float32, copy=True)

    out = np.empty((TARGET_FRAMES, d), dtype=np.float32)
    out[:t] = sequence

    if t == 0:
        raise ValueError("Cannot repeat-pad an empty sequence.")

    out[t:] = sequence[-1]

    return out


TEMPORAL_FUNCTIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "linear_interpolation": linear_interpolation,
    "nearest_resample": nearest_resample,
    "floor_resample": floor_resample,
    "repeat_last_padding": repeat_last_padding,
}


def prepare_sequence(raw: np.ndarray, method: str) -> np.ndarray:
    hands = TEMPORAL_FUNCTIONS[method](raw)

    if hands.shape != (TARGET_FRAMES, RAW_FEATURES):
        raise ValueError(
            f"{method} produced {hands.shape}; "
            f"expected ({TARGET_FRAMES},{RAW_FEATURES})"
        )

    out = np.zeros((TARGET_FRAMES, MODEL_FEATURES), dtype=np.float32)
    out[:, :RAW_FEATURES] = hands

    return out


# ---------------------------------------------------------------------------
# Checkpoint / classifier helpers
# ---------------------------------------------------------------------------


def extract_checkpoint_mean() -> np.ndarray:
    bundle = load_ksl_classifier()

    checkpoint = None

    if isinstance(bundle, dict):
        checkpoint = bundle.get("checkpoint", bundle)

    elif isinstance(bundle, tuple):
        for item in bundle:
            if isinstance(item, dict) and "mean" in item:
                checkpoint = item
                break

    else:
        checkpoint = getattr(bundle, "checkpoint", None)

    if checkpoint is not None and "mean" in checkpoint:
        mean = to_numpy(checkpoint["mean"])
    else:
        mean = getattr(bundle, "mean", None)

        if mean is None:
            raise RuntimeError(
                "Could not recover checkpoint mean from load_ksl_classifier()."
            )

        mean = to_numpy(mean)

    if mean.shape != (MODEL_FEATURES,):
        raise ValueError(
            f"Checkpoint mean has shape {mean.shape}; expected (225,)"
        )

    return mean


def _parse_prediction_container(container: Any) -> Dict[str, float]:
    if container is None:
        return {}

    if isinstance(container, dict):
        direct = {
            str(k): float(v)
            for k, v in container.items()
            if isinstance(v, (int, float, np.floating))
        }

        if direct:
            return direct

        out: Dict[str, float] = {}

        for value in container.values():
            if isinstance(value, dict):
                label = (
                    value.get("label")
                    or value.get("class")
                    or value.get("gloss")
                )
                prob = value.get(
                    "probability",
                    value.get("score", value.get("confidence")),
                )

                if label is not None and prob is not None:
                    out[str(label)] = float(prob)

        return out

    if isinstance(container, (list, tuple)):
        out: Dict[str, float] = {}

        for item in container:
            if isinstance(item, dict):
                label = (
                    item.get("label")
                    or item.get("class")
                    or item.get("gloss")
                )
                prob = item.get(
                    "probability",
                    item.get("score", item.get("confidence")),
                )
            else:
                label = (
                    getattr(item, "label", None)
                    or getattr(item, "class_name", None)
                    or getattr(item, "gloss", None)
                )

                prob = getattr(item, "probability", None)

                if prob is None:
                    prob = getattr(item, "score", None)

                if prob is None:
                    prob = getattr(item, "confidence", None)

            if label is not None and prob is not None:
                out[str(label)] = float(prob)

        return out

    return {}


def prediction_probabilities(sequence: np.ndarray) -> Dict[str, float]:
    result = classify_checkpoint_sequence(sequence)

    if isinstance(result, dict):
        for key in (
            "probabilities",
            "scores",
            "predictions",
            "ranking",
            "ranked_predictions",
        ):
            if key in result:
                parsed = _parse_prediction_container(result[key])

                if parsed:
                    return parsed

        direct = {
            label: float(result[label])
            for label in EXPECTED_LABELS
            if label in result
            and isinstance(result[label], (int, float, np.floating))
        }

        if direct:
            return direct

    for attr in (
        "probabilities",
        "scores",
        "predictions",
        "ranking",
        "ranked_predictions",
    ):
        if hasattr(result, attr):
            parsed = _parse_prediction_container(getattr(result, attr))

            if parsed:
                return parsed

    parsed = _parse_prediction_container(result)

    if parsed:
        return parsed

    raise RuntimeError(f"Could not parse classifier output: {result!r}")


def neutralize_joint(
    sequence: np.ndarray,
    joint_index: int,
    checkpoint_mean: np.ndarray,
) -> np.ndarray:
    start = joint_index * COORDS_PER_JOINT
    end = start + COORDS_PER_JOINT

    out = sequence.copy()
    out[:, start:end] = checkpoint_mean[start:end][None, :]

    return out


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading/locating KSL Kaggle dataset...")
    dataset_root = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    test_dir = find_clean_test_dir(dataset_root)
    samples = load_clean_test_samples(test_dir)

    print(f"Dataset root: {dataset_root}")
    print(f"Clean test split: {test_dir}")
    print(f"Samples: {len(samples)}")

    _ = load_ksl_classifier()
    checkpoint_mean = extract_checkpoint_mean()

    method_joint_values: Dict[str, List[List[float]]] = {
        method: [[] for _ in range(N_JOINTS)]
        for method in METHODS
    }

    method_class_joint_values: Dict[str, Dict[str, List[List[float]]]] = {
        method: {
            label: [[] for _ in range(N_JOINTS)]
            for label in EXPECTED_LABELS
        }
        for method in METHODS
    }

    method_clean_correct = {method: 0 for method in METHODS}
    csv_rows: List[Dict[str, Any]] = []

    for method_index, method in enumerate(METHODS, start=1):
        print("\n" + "=" * 72)
        print(f"[{method_index}/{len(METHODS)}] {method}")
        print("=" * 72)

        for sample_index, (true_label, path, raw) in enumerate(
            samples,
            start=1,
        ):
            sequence = prepare_sequence(raw, method)
            clean_probs = prediction_probabilities(sequence)

            clean_true_prob = float(clean_probs[true_label])
            clean_prediction = max(
                clean_probs.items(),
                key=lambda pair: pair[1],
            )[0]

            method_clean_correct[method] += int(
                clean_prediction == true_label
            )

            joint_drops = np.zeros(N_JOINTS, dtype=np.float32)

            for joint_index in range(N_JOINTS):
                modified = neutralize_joint(
                    sequence,
                    joint_index,
                    checkpoint_mean,
                )

                modified_probs = prediction_probabilities(modified)
                modified_true_prob = float(
                    modified_probs[true_label]
                )

                drop = clean_true_prob - modified_true_prob
                joint_drops[joint_index] = drop

                method_joint_values[method][joint_index].append(drop)
                method_class_joint_values[
                    method
                ][
                    true_label
                ][
                    joint_index
                ].append(drop)

                csv_rows.append(
                    {
                        "method": method,
                        "sample": path.name,
                        "true_label": true_label,
                        "clean_prediction": clean_prediction,
                        "clean_true_probability": clean_true_prob,
                        "joint_index": joint_index,
                        "true_probability_drop": drop,
                    }
                )

            if sample_index % 10 == 0 or sample_index == len(samples):
                print(
                    f"  [{sample_index:03d}/{len(samples)}] "
                    f"last={true_label}/{clean_prediction}"
                )

    # -------------------------------------------------------------------
    # Aggregate attribution profiles
    # -------------------------------------------------------------------

    global_profiles: Dict[str, np.ndarray] = {}

    class_profiles: Dict[str, Dict[str, np.ndarray]] = {
        label: {}
        for label in EXPECTED_LABELS
    }

    for method in METHODS:
        global_profiles[method] = np.asarray(
            [
                safe_mean(values)
                for values in method_joint_values[method]
            ],
            dtype=np.float32,
        )

        for label in EXPECTED_LABELS:
            class_profiles[label][method] = np.asarray(
                [
                    safe_mean(values)
                    for values in method_class_joint_values[
                        method
                    ][
                        label
                    ]
                ],
                dtype=np.float32,
            )

    # -------------------------------------------------------------------
    # Pairwise stability
    # -------------------------------------------------------------------

    pairwise: List[Dict[str, Any]] = []

    for i, method_a in enumerate(METHODS):
        for method_b in METHODS[i + 1:]:
            profile_a = global_profiles[method_a]
            profile_b = global_profiles[method_b]

            record: Dict[str, Any] = {
                "method_a": method_a,
                "method_b": method_b,
                "pearson": pearson_corr(
                    profile_a,
                    profile_b,
                ),
                "spearman": spearman_corr(
                    profile_a,
                    profile_b,
                ),
                "top_k": {},
            }

            for k in TOP_K_VALUES:
                jaccard, intersection = jaccard_topk(
                    profile_a,
                    profile_b,
                    k,
                )

                record["top_k"][str(k)] = {
                    "jaccard": jaccard,
                    "intersection_count": intersection,
                }

            pairwise.append(record)

    # Per-class pairwise Spearman, useful because global averaging can hide
    # class-specific instability.
    class_pairwise: Dict[str, List[Dict[str, Any]]] = {}

    for label in EXPECTED_LABELS:
        class_pairwise[label] = []

        for i, method_a in enumerate(METHODS):
            for method_b in METHODS[i + 1:]:
                profile_a = class_profiles[label][method_a]
                profile_b = class_profiles[label][method_b]

                class_pairwise[label].append(
                    {
                        "method_a": method_a,
                        "method_b": method_b,
                        "pearson": pearson_corr(
                            profile_a,
                            profile_b,
                        ),
                        "spearman": spearman_corr(
                            profile_a,
                            profile_b,
                        ),
                        "top_10_jaccard": jaccard_topk(
                            profile_a,
                            profile_b,
                            10,
                        )[0],
                    }
                )

    # -------------------------------------------------------------------
    # Consensus rankings
    # -------------------------------------------------------------------

    # Mean normalized rank across methods. We rank larger probability drops
    # as more important, hence negate values before ascending rank assignment.
    rank_matrix = []

    for method in METHODS:
        ranks = average_ranks(
            -global_profiles[method]
        )
        rank_matrix.append(ranks)

    rank_matrix = np.asarray(rank_matrix, dtype=np.float64)
    mean_rank = np.mean(rank_matrix, axis=0)

    consensus_order = np.argsort(mean_rank).astype(int)

    consensus = [
        {
            "joint_index": int(joint_index),
            "mean_rank": float(mean_rank[joint_index]),
            "mean_probability_drop_across_methods": float(
                np.mean(
                    [
                        global_profiles[method][joint_index]
                        for method in METHODS
                    ]
                )
            ),
            "method_ranks": {
                method: float(
                    rank_matrix[m_idx, joint_index]
                )
                for m_idx, method in enumerate(METHODS)
            },
        }
        for joint_index in consensus_order
    ]

    # -------------------------------------------------------------------
    # Structured output
    # -------------------------------------------------------------------

    method_summaries: Dict[str, Any] = {}

    for method in METHODS:
        profile = global_profiles[method]

        method_summaries[method] = {
            "clean_accuracy": (
                method_clean_correct[method] / len(samples)
            ),
            "clean_correct": method_clean_correct[method],
            "n_samples": len(samples),
            "global_joint_mean_probability_drop": profile.tolist(),
            "top_15_joints": top_indices(
                profile,
                15,
            ),
            "by_class": {
                label: {
                    "joint_mean_probability_drop": (
                        class_profiles[label][method].tolist()
                    ),
                    "top_10_joints": top_indices(
                        class_profiles[label][method],
                        10,
                    ),
                }
                for label in EXPECTED_LABELS
            },
        }

    payload = {
        "experiment": (
            "KSL landmark attribution stability across temporal preprocessing"
        ),
        "dataset": KAGGLE_DATASET,
        "n_samples": len(samples),
        "methods": list(METHODS),
        "representation": {
            "source": "(T,126) = 42 joints x 3 coordinates",
            "checkpoint_input": "(64,225)",
            "padding": "99 structural zeros",
            "joint_labels": (
                "joint_00 ... joint_41; anatomical left/right ordering "
                "is not assumed."
            ),
        },
        "attribution": {
            "metric": (
                "p_true(clean) - p_true(joint-neutralized)"
            ),
            "neutralization": (
                "xyz coordinates replaced by checkpoint training mean"
            ),
        },
        "method_summaries": method_summaries,
        "global_pairwise_stability": pairwise,
        "class_pairwise_stability": class_pairwise,
        "consensus_joint_ranking": consensus,
        "interpretation": {
            "purpose": (
                "Determine whether joint attribution conclusions survive "
                "reasonable uncertainty about temporal reconstruction."
            ),
            "warning": (
                "Stable attribution means model sensitivity is robust to "
                "these tested preprocessing choices. It does not establish "
                "linguistic causality, anatomical importance, or a validated "
                "explanation of KSL."
            ),
        },
    }

    JSON_PATH.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fieldnames = [
            "method",
            "sample",
            "true_label",
            "clean_prediction",
            "clean_true_probability",
            "joint_index",
            "true_probability_drop",
        ]

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(csv_rows)

    # -------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------

    joints = np.arange(N_JOINTS)

    plt.figure(figsize=(12, 6))

    for method in METHODS:
        plt.plot(
            joints,
            global_profiles[method],
            label=method,
        )

    plt.axhline(0.0, linewidth=1)
    plt.axvline(
        20.5,
        linestyle="--",
        linewidth=1,
    )
    plt.xlabel("Source joint index")
    plt.ylabel("Mean drop in true-class probability")
    plt.title(
        "KSL Joint Attribution Across Temporal Preprocessing Candidates"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        PROFILE_PLOT_PATH,
        dpi=180,
    )
    plt.close()

    # Pairwise Spearman matrix
    spearman_matrix = np.eye(
        len(METHODS),
        dtype=np.float32,
    )

    for record in pairwise:
        i = METHODS.index(record["method_a"])
        j = METHODS.index(record["method_b"])

        value = record["spearman"]

        spearman_matrix[i, j] = value
        spearman_matrix[j, i] = value

    plt.figure(figsize=(8, 7))
    image = plt.imshow(
        spearman_matrix,
        vmin=-1.0,
        vmax=1.0,
    )
    plt.xticks(
        np.arange(len(METHODS)),
        METHODS,
        rotation=35,
        ha="right",
    )
    plt.yticks(
        np.arange(len(METHODS)),
        METHODS,
    )
    plt.colorbar(
        image,
        label="Spearman rank correlation",
    )
    plt.title(
        "KSL Attribution Stability — Pairwise Rank Correlation"
    )
    plt.tight_layout()
    plt.savefig(
        CORR_PLOT_PATH,
        dpi=180,
    )
    plt.close()

    # Top-10 Jaccard matrix
    topk_matrix = np.eye(
        len(METHODS),
        dtype=np.float32,
    )

    for record in pairwise:
        i = METHODS.index(record["method_a"])
        j = METHODS.index(record["method_b"])

        value = record["top_k"]["10"]["jaccard"]

        topk_matrix[i, j] = value
        topk_matrix[j, i] = value

    plt.figure(figsize=(8, 7))
    image = plt.imshow(
        topk_matrix,
        vmin=0.0,
        vmax=1.0,
    )
    plt.xticks(
        np.arange(len(METHODS)),
        METHODS,
        rotation=35,
        ha="right",
    )
    plt.yticks(
        np.arange(len(METHODS)),
        METHODS,
    )
    plt.colorbar(
        image,
        label="Top-10 Jaccard overlap",
    )
    plt.title(
        "KSL Attribution Stability — Top-10 Joint Overlap"
    )
    plt.tight_layout()
    plt.savefig(
        TOPK_PLOT_PATH,
        dpi=180,
    )
    plt.close()

    # -------------------------------------------------------------------
    # Console summary
    # -------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("KSL ATTRIBUTION STABILITY COMPLETE")
    print("=" * 72)

    print("\nClean classification by temporal method:")

    for method in METHODS:
        correct = method_clean_correct[method]
        accuracy = correct / len(samples)

        print(
            f"  {method:<24} "
            f"{correct}/{len(samples)} "
            f"= {accuracy:.4f}"
        )

    print("\nGlobal top-10 joints:")

    for method in METHODS:
        print(
            f"  {method:<24} "
            f"{top_indices(global_profiles[method], 10)}"
        )

    print("\nPairwise global stability:")

    for record in pairwise:
        print(
            f"  {record['method_a']} vs {record['method_b']}: "
            f"Spearman={record['spearman']:.4f}, "
            f"Pearson={record['pearson']:.4f}, "
            f"Top10-Jaccard="
            f"{record['top_k']['10']['jaccard']:.4f}"
        )

    print("\nConsensus top-15 joints:")
    print(
        "  "
        + str(
            [
                item["joint_index"]
                for item in consensus[:15]
            ]
        )
    )

    print("\nOutputs:")

    for path in (
        JSON_PATH,
        CSV_PATH,
        PROFILE_PLOT_PATH,
        CORR_PLOT_PATH,
        TOPK_PLOT_PATH,
    ):
        print(
            f"  {path.relative_to(ROOT)}"
        )

    print(
        "\nInterpretation: high agreement means the model-sensitivity "
        "ranking is robust to the tested temporal preprocessing uncertainty. "
        "It still does not establish linguistic or anatomical causality."
    )


if __name__ == "__main__":
    main()