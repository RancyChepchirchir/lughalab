from __future__ import annotations

"""
KSL landmark-level occlusion attribution for the four-class Landmark Transformer.

This experiment asks which of the 42 source hand joints the trained classifier
depends on most strongly.

The public source representation is:

    42 joints x 3 coordinates = 126 features

The checkpoint then uses:

    [126 source hand features, 99 structural zeros] -> 225 features

Important:
- The public documentation establishes 42 hand joints in the first 126 features.
- It does NOT establish the internal anatomical ordering strongly enough for us
  to label the first 21 joints "left" and the second 21 "right".
- Therefore this script uses neutral labels joint_00 ... joint_41 and, where
  useful, block_1 (joints 0-20) and block_2 (joints 21-41).

For true class y and joint j:

    A_j = p_theta(y | X) - p_theta(y | X^(-j))

where all xyz coordinates for joint j, across all 64 frames, are replaced by
the checkpoint training mean. After checkpoint z-score normalization these
neutralized features become approximately zero.

This measures model sensitivity. It is NOT causal linguistic importance,
anatomical importance, or a validated explanation of Kenyan Sign Language.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

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
AXIS_NAMES = ("x", "y", "z")

EXPECTED_LABELS = ("father", "hello", "is", "my")

RESULTS_DIR = ROOT / "experiments" / "results"
JSON_PATH = RESULTS_DIR / "ksl_landmark_attribution.json"
CSV_PATH = RESULTS_DIR / "ksl_landmark_attribution.csv"
GLOBAL_PLOT_PATH = RESULTS_DIR / "ksl_landmark_attribution_global.png"
CLASS_PLOT_PATH = RESULTS_DIR / "ksl_landmark_attribution_by_class.png"
AXIS_PLOT_PATH = RESULTS_DIR / "ksl_coordinate_axis_ablation.png"

# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)


def safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def safe_std(values: Sequence[float]) -> float:
    return float(np.std(values)) if values else float("nan")


def top_indices(values: np.ndarray, k: int = 10) -> List[int]:
    k = min(k, len(values))
    return np.argsort(values)[::-1][:k].astype(int).tolist()


# ---------------------------------------------------------------------------
# Dataset loading
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

        # Explicitly avoid the duplicate-containing dataset3 branch.
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

        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        for path in sorted(class_dir.rglob("*.npy")):
            array = np.load(path).astype(np.float32)

            if array.ndim != 2 or array.shape[1] != RAW_FEATURES:
                raise ValueError(
                    f"{path} has shape {array.shape}; expected (T,{RAW_FEATURES})"
                )

            samples.append((label, path, array))

    if not samples:
        raise RuntimeError(f"No .npy samples found beneath {test_dir}")

    return samples


# ---------------------------------------------------------------------------
# Reconstructed temporal preprocessing
# ---------------------------------------------------------------------------


def linear_resample(
    sequence: np.ndarray,
    target_frames: int = TARGET_FRAMES,
) -> np.ndarray:
    """
    Deterministically resample (T, D) -> (64, D) with linear interpolation.

    Earlier LughaLab temporal reconstruction showed several non-zero resampling
    candidates tied on classification accuracy. Linear interpolation is used
    here as a reproducible candidate, not as a claim about the original
    training pipeline.
    """
    if sequence.ndim != 2:
        raise ValueError(f"Expected a 2D sequence, got {sequence.shape}")

    t, d = sequence.shape

    if t == target_frames:
        return sequence.astype(np.float32, copy=True)

    if t == 1:
        return np.repeat(sequence, target_frames, axis=0).astype(np.float32)

    old_x = np.linspace(0.0, 1.0, t)
    new_x = np.linspace(0.0, 1.0, target_frames)

    out = np.empty((target_frames, d), dtype=np.float32)

    for feature_idx in range(d):
        out[:, feature_idx] = np.interp(
            new_x,
            old_x,
            sequence[:, feature_idx],
        ).astype(np.float32)

    return out


def prepare_sequence(raw: np.ndarray) -> np.ndarray:
    """
    (T,126) -> (64,126) -> append 99 structural zeros -> (64,225)
    """
    hands = linear_resample(raw)

    out = np.zeros((TARGET_FRAMES, MODEL_FEATURES), dtype=np.float32)
    out[:, :RAW_FEATURES] = hands

    return out


# ---------------------------------------------------------------------------
# Checkpoint / classifier compatibility
# ---------------------------------------------------------------------------


def extract_checkpoint_mean() -> np.ndarray:
    """
    Recover the checkpoint training mean from the existing classifier service.

    Supports the bundle structures used during LughaLab development.
    """
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
            f"Checkpoint mean has shape {mean.shape}; expected ({MODEL_FEATURES},)"
        )

    return mean


def _parse_prediction_container(container: Any) -> Dict[str, float]:
    if container is None:
        return {}

    if isinstance(container, dict):
        direct = {
            str(key): float(value)
            for key, value in container.items()
            if isinstance(value, (int, float, np.floating))
        }

        if direct:
            return direct

        out: Dict[str, float] = {}

        for value in container.values():
            if not isinstance(value, dict):
                continue

            label = (
                value.get("label")
                or value.get("class")
                or value.get("gloss")
            )
            probability = value.get(
                "probability",
                value.get("score", value.get("confidence")),
            )

            if label is not None and probability is not None:
                out[str(label)] = float(probability)

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
                probability = item.get(
                    "probability",
                    item.get("score", item.get("confidence")),
                )
            else:
                label = (
                    getattr(item, "label", None)
                    or getattr(item, "class_name", None)
                    or getattr(item, "gloss", None)
                )

                probability = getattr(item, "probability", None)

                if probability is None:
                    probability = getattr(item, "score", None)

                if probability is None:
                    probability = getattr(item, "confidence", None)

            if label is not None and probability is not None:
                out[str(label)] = float(probability)

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


# ---------------------------------------------------------------------------
# Attribution perturbations
# ---------------------------------------------------------------------------


def joint_feature_slice(joint_index: int) -> slice:
    if not 0 <= joint_index < N_JOINTS:
        raise IndexError(f"Joint index out of range: {joint_index}")

    start = joint_index * COORDS_PER_JOINT
    end = start + COORDS_PER_JOINT

    return slice(start, end)


def neutralize_joint(
    sequence: np.ndarray,
    joint_index: int,
    checkpoint_mean: np.ndarray,
) -> np.ndarray:
    """
    Replace all xyz coordinates for one source joint across all 64 frames with
    the checkpoint mean.
    """
    out = sequence.copy()
    feature_slice = joint_feature_slice(joint_index)

    out[:, feature_slice] = checkpoint_mean[feature_slice][None, :]

    return out


def neutralize_axis(
    sequence: np.ndarray,
    axis_index: int,
    checkpoint_mean: np.ndarray,
) -> np.ndarray:
    """
    Neutralize one coordinate axis (x, y, or z) across all 42 source joints
    and all 64 frames.
    """
    if axis_index not in (0, 1, 2):
        raise IndexError(f"Axis index out of range: {axis_index}")

    out = sequence.copy()

    feature_indices = np.arange(
        axis_index,
        RAW_FEATURES,
        COORDS_PER_JOINT,
    )

    out[:, feature_indices] = checkpoint_mean[feature_indices][None, :]

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

    print("Loading KSL classifier checkpoint...")
    _ = load_ksl_classifier()
    checkpoint_mean = extract_checkpoint_mean()

    global_joint_drops: List[List[float]] = [
        [] for _ in range(N_JOINTS)
    ]

    class_joint_drops: Dict[str, List[List[float]]] = {
        label: [[] for _ in range(N_JOINTS)]
        for label in EXPECTED_LABELS
    }

    axis_drops: Dict[str, List[float]] = {
        axis_name: [] for axis_name in AXIS_NAMES
    }

    class_axis_drops: Dict[str, Dict[str, List[float]]] = {
        label: {
            axis_name: [] for axis_name in AXIS_NAMES
        }
        for label in EXPECTED_LABELS
    }

    csv_rows: List[Dict[str, Any]] = []
    sample_records: List[Dict[str, Any]] = []

    clean_correct = 0

    for sample_index, (true_label, path, raw) in enumerate(
        samples,
        start=1,
    ):
        sequence = prepare_sequence(raw)
        clean_probs = prediction_probabilities(sequence)

        if true_label not in clean_probs:
            raise RuntimeError(
                f"Classifier output does not contain true label "
                f"{true_label!r}. Available: {sorted(clean_probs)}"
            )

        clean_true_prob = float(clean_probs[true_label])
        clean_prediction = max(
            clean_probs.items(),
            key=lambda pair: pair[1],
        )[0]

        clean_correct += int(clean_prediction == true_label)

        joint_drops = np.zeros(N_JOINTS, dtype=np.float32)

        # ---------------------------------------------------------------
        # Joint-level occlusion
        # ---------------------------------------------------------------

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
            global_joint_drops[joint_index].append(drop)
            class_joint_drops[true_label][joint_index].append(drop)

            csv_rows.append(
                {
                    "sample": path.name,
                    "true_label": true_label,
                    "clean_prediction": clean_prediction,
                    "clean_true_probability": clean_true_prob,
                    "ablation_type": "joint",
                    "feature_name": f"joint_{joint_index:02d}",
                    "feature_index": joint_index,
                    "true_probability_drop": drop,
                }
            )

        # ---------------------------------------------------------------
        # Coordinate-axis ablation
        # ---------------------------------------------------------------

        sample_axis_drops: Dict[str, float] = {}

        for axis_index, axis_name in enumerate(AXIS_NAMES):
            modified = neutralize_axis(
                sequence,
                axis_index,
                checkpoint_mean,
            )

            modified_probs = prediction_probabilities(modified)
            modified_true_prob = float(
                modified_probs[true_label]
            )

            drop = clean_true_prob - modified_true_prob

            sample_axis_drops[axis_name] = drop
            axis_drops[axis_name].append(drop)
            class_axis_drops[true_label][axis_name].append(drop)

            csv_rows.append(
                {
                    "sample": path.name,
                    "true_label": true_label,
                    "clean_prediction": clean_prediction,
                    "clean_true_probability": clean_true_prob,
                    "ablation_type": "coordinate_axis",
                    "feature_name": axis_name,
                    "feature_index": axis_index,
                    "true_probability_drop": drop,
                }
            )

        record = {
            "sample": path.name,
            "source_path": str(path),
            "true_label": true_label,
            "raw_shape": list(raw.shape),
            "clean_prediction": clean_prediction,
            "clean_true_probability": clean_true_prob,
            "clean_probabilities": clean_probs,
            "joint_probability_drop": joint_drops.tolist(),
            "top_10_joints": top_indices(
                joint_drops,
                k=10,
            ),
            "coordinate_axis_probability_drop": sample_axis_drops,
        }

        sample_records.append(record)

        print(
            f"[{sample_index:03d}/{len(samples)}] "
            f"{true_label:<6} "
            f"pred={clean_prediction:<6} "
            f"p(y)={clean_true_prob:.4f} "
            f"top_joints={record['top_10_joints'][:5]}"
        )

    # -------------------------------------------------------------------
    # Aggregate joint profiles
    # -------------------------------------------------------------------

    global_joint_mean = np.asarray(
        [
            safe_mean(values)
            for values in global_joint_drops
        ],
        dtype=np.float32,
    )

    global_joint_std = np.asarray(
        [
            safe_std(values)
            for values in global_joint_drops
        ],
        dtype=np.float32,
    )

    by_class: Dict[str, Any] = {}

    for label in EXPECTED_LABELS:
        mean_profile = np.asarray(
            [
                safe_mean(values)
                for values in class_joint_drops[label]
            ],
            dtype=np.float32,
        )

        std_profile = np.asarray(
            [
                safe_std(values)
                for values in class_joint_drops[label]
            ],
            dtype=np.float32,
        )

        by_class[label] = {
            "n_samples": sum(
                record["true_label"] == label
                for record in sample_records
            ),
            "joint_mean_probability_drop": mean_profile.tolist(),
            "joint_std_probability_drop": std_profile.tolist(),
            "top_10_joints": top_indices(mean_profile, k=10),
        }

    # -------------------------------------------------------------------
    # Aggregate axis profiles
    # -------------------------------------------------------------------

    axis_summary: Dict[str, Any] = {}

    for axis_name in AXIS_NAMES:
        axis_summary[axis_name] = {
            "global_mean_probability_drop": safe_mean(
                axis_drops[axis_name]
            ),
            "global_std_probability_drop": safe_std(
                axis_drops[axis_name]
            ),
            "by_class": {
                label: {
                    "mean_probability_drop": safe_mean(
                        class_axis_drops[label][axis_name]
                    ),
                    "std_probability_drop": safe_std(
                        class_axis_drops[label][axis_name]
                    ),
                }
                for label in EXPECTED_LABELS
            },
        }

    clean_accuracy = clean_correct / len(samples)

    # Two abstract 21-joint source blocks. Do not assign anatomical names.
    block_1_mean = float(np.mean(global_joint_mean[:21]))
    block_2_mean = float(np.mean(global_joint_mean[21:]))

    payload = {
        "experiment": "KSL landmark-level occlusion attribution",
        "dataset": KAGGLE_DATASET,
        "n_samples": len(samples),
        "labels": list(EXPECTED_LABELS),
        "clean_accuracy": clean_accuracy,
        "representation": {
            "raw_source": "(T,126) = 42 joints x 3 coordinates",
            "model_input": "(64,225)",
            "first_126_features": "source hand landmarks",
            "last_99_features": "structural zero padding",
            "joint_naming": (
                "joint_00 ... joint_41. Anatomical left/right labels are "
                "deliberately not assigned because source ordering has not "
                "been verified."
            ),
            "abstract_blocks": {
                "block_1": "joints 0-20 / features 0-62",
                "block_2": "joints 21-41 / features 63-125",
            },
        },
        "preprocessing": {
            "temporal_method": "linear interpolation",
            "target_frames": TARGET_FRAMES,
            "claim": (
                "Deterministic reconstructed candidate only; not claimed "
                "to be the exact original training transform."
            ),
        },
        "occlusion": {
            "neutral_value": "checkpoint training mean",
            "joint_metric": (
                "p_true(clean) - p_true(joint-neutralized)"
            ),
            "axis_metric": (
                "p_true(clean) - p_true(axis-neutralized)"
            ),
        },
        "global": {
            "joint_mean_probability_drop": global_joint_mean.tolist(),
            "joint_std_probability_drop": global_joint_std.tolist(),
            "top_15_joints": top_indices(global_joint_mean, k=15),
            "abstract_block_mean_drop": {
                "block_1": block_1_mean,
                "block_2": block_2_mean,
            },
        },
        "by_class": by_class,
        "coordinate_axis_ablation": axis_summary,
        "samples": sample_records,
        "interpretation_warning": (
            "The reported quantities measure classifier sensitivity to "
            "checkpoint-mean neutralization. They must not be interpreted "
            "as causal linguistic importance, anatomical importance, or "
            "human-grounded KSL saliency. Perturbed inputs may also be "
            "partially out-of-distribution."
        ),
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
            "sample",
            "true_label",
            "clean_prediction",
            "clean_true_probability",
            "ablation_type",
            "feature_name",
            "feature_index",
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

    plt.figure(figsize=(12, 5))
    plt.plot(joints, global_joint_mean)
    plt.fill_between(
        joints,
        global_joint_mean - global_joint_std,
        global_joint_mean + global_joint_std,
        alpha=0.2,
    )
    plt.axhline(0.0, linewidth=1)
    plt.axvline(20.5, linestyle="--", linewidth=1)
    plt.xlabel("Source joint index")
    plt.ylabel("Mean drop in true-class probability")
    plt.title("KSL Landmark Attribution — Global Joint Occlusion")
    plt.tight_layout()
    plt.savefig(
        GLOBAL_PLOT_PATH,
        dpi=180,
    )
    plt.close()

    plt.figure(figsize=(12, 6))

    for label in EXPECTED_LABELS:
        profile = np.asarray(
            by_class[label]["joint_mean_probability_drop"],
            dtype=np.float32,
        )
        plt.plot(
            joints,
            profile,
            label=label,
        )

    plt.axhline(0.0, linewidth=1)
    plt.axvline(20.5, linestyle="--", linewidth=1)
    plt.xlabel("Source joint index")
    plt.ylabel("Mean drop in true-class probability")
    plt.title("KSL Landmark Attribution — By Gloss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        CLASS_PLOT_PATH,
        dpi=180,
    )
    plt.close()

    axis_global_means = [
        axis_summary[axis_name][
            "global_mean_probability_drop"
        ]
        for axis_name in AXIS_NAMES
    ]

    plt.figure(figsize=(7, 5))
    plt.bar(
        AXIS_NAMES,
        axis_global_means,
    )
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Coordinate axis")
    plt.ylabel("Mean drop in true-class probability")
    plt.title("KSL Coordinate-Axis Ablation")
    plt.tight_layout()
    plt.savefig(
        AXIS_PLOT_PATH,
        dpi=180,
    )
    plt.close()

    # -------------------------------------------------------------------
    # Console report
    # -------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("KSL LANDMARK ATTRIBUTION COMPLETE")
    print("=" * 72)

    print(
        f"Clean accuracy: "
        f"{clean_correct}/{len(samples)} "
        f"= {clean_accuracy:.4f}"
    )

    print(
        "Global top-15 joints: "
        f"{top_indices(global_joint_mean, k=15)}"
    )

    print(
        "Abstract block mean drop: "
        f"block_1={block_1_mean:.6f}, "
        f"block_2={block_2_mean:.6f}"
    )

    print("\nPer-class top-10 joints:")

    for label in EXPECTED_LABELS:
        print(
            f"  {label:<6}: "
            f"{by_class[label]['top_10_joints']}"
        )

    print("\nCoordinate-axis mean probability drops:")

    for axis_name in AXIS_NAMES:
        value = axis_summary[axis_name][
            "global_mean_probability_drop"
        ]
        print(
            f"  {axis_name}: {value:.6f}"
        )

    print("\nOutputs:")

    for path in (
        JSON_PATH,
        CSV_PATH,
        GLOBAL_PLOT_PATH,
        CLASS_PLOT_PATH,
        AXIS_PLOT_PATH,
    ):
        print(
            f"  {path.relative_to(ROOT)}"
        )

    print(
        "\nInterpretation: positive probability drop means the classifier "
        "was sensitive to neutralizing that source joint or coordinate axis. "
        "This is a model-dependence diagnostic, not linguistic ground truth."
    )


if __name__ == "__main__":
    main()