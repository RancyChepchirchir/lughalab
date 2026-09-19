from __future__ import annotations

"""
Temporal occlusion attribution for the four-class KSL Landmark Transformer.

For clean input X and true class y:

    A_t = p_theta(y | X) - p_theta(y | X^(-t))

where X^(-t) replaces frame t with the checkpoint training mean.

This is a model-sensitivity diagnostic, not linguistic segmentation,
causal evidence, or a human-grounded explanation of KSL.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import kagglehub

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"

# Support both repository-style imports (backend.app...)
# and the backend's existing top-level package imports (app...).
for path in (ROOT, BACKEND):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from backend.app.services.ksl_classifier import (
    classify_checkpoint_sequence,
    load_ksl_classifier,
)

KAGGLE_DATASET = "joanwachuka/ksl-hand-landmarks"
TARGET_FRAMES = 64
RAW_FEATURES = 126
MODEL_FEATURES = 225
WINDOW_SIZES = (3, 5, 9)
EXPECTED_LABELS = ("father", "hello", "is", "my")

RESULTS_DIR = ROOT / "experiments" / "results"
JSON_PATH = RESULTS_DIR / "ksl_temporal_attribution.json"
CSV_PATH = RESULTS_DIR / "ksl_temporal_attribution.csv"
GLOBAL_PLOT_PATH = RESULTS_DIR / "ksl_temporal_attribution_global.png"
CLASS_PLOT_PATH = RESULTS_DIR / "ksl_temporal_attribution_by_class.png"
WINDOW_PLOT_PATH = RESULTS_DIR / "ksl_temporal_window_attribution.png"


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)


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

    raise FileNotFoundError(f"Could not locate clean KSL test split in {dataset_root}")


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
        raise RuntimeError(f"No KSL test samples found in {test_dir}")
    return samples


def linear_resample(sequence: np.ndarray, target_frames: int = TARGET_FRAMES) -> np.ndarray:
    t, d = sequence.shape
    if t == target_frames:
        return sequence.astype(np.float32, copy=True)
    if t == 1:
        return np.repeat(sequence, target_frames, axis=0).astype(np.float32)

    old_x = np.linspace(0.0, 1.0, t)
    new_x = np.linspace(0.0, 1.0, target_frames)
    out = np.empty((target_frames, d), dtype=np.float32)

    for j in range(d):
        out[:, j] = np.interp(new_x, old_x, sequence[:, j]).astype(np.float32)

    return out


def prepare_sequence(raw: np.ndarray) -> np.ndarray:
    hands = linear_resample(raw)
    out = np.zeros((TARGET_FRAMES, MODEL_FEATURES), dtype=np.float32)
    out[:, :RAW_FEATURES] = hands
    return out


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
        raise ValueError(f"Checkpoint mean has shape {mean.shape}; expected (225,)")

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
                label = value.get("label") or value.get("class") or value.get("gloss")
                prob = value.get("probability", value.get("score", value.get("confidence")))
                if label is not None and prob is not None:
                    out[str(label)] = float(prob)
        return out

    if isinstance(container, (list, tuple)):
        out: Dict[str, float] = {}
        for item in container:
            if isinstance(item, dict):
                label = item.get("label") or item.get("class") or item.get("gloss")
                prob = item.get("probability", item.get("score", item.get("confidence")))
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
        for key in ("probabilities", "scores", "predictions", "ranking", "ranked_predictions"):
            if key in result:
                parsed = _parse_prediction_container(result[key])
                if parsed:
                    return parsed

        direct = {
            label: float(result[label])
            for label in EXPECTED_LABELS
            if label in result and isinstance(result[label], (int, float, np.floating))
        }
        if direct:
            return direct

    for attr in ("probabilities", "scores", "predictions", "ranking", "ranked_predictions"):
        if hasattr(result, attr):
            parsed = _parse_prediction_container(getattr(result, attr))
            if parsed:
                return parsed

    parsed = _parse_prediction_container(result)
    if parsed:
        return parsed

    raise RuntimeError(f"Could not parse classifier output: {result!r}")


def neutralize_frames(
    sequence: np.ndarray,
    start: int,
    end_exclusive: int,
    checkpoint_mean: np.ndarray,
) -> np.ndarray:
    out = sequence.copy()
    out[start:end_exclusive, :] = checkpoint_mean[None, :]
    return out


def centered_window_bounds(center: int, width: int) -> Tuple[int, int]:
    half = width // 2
    start = max(0, center - half)
    end = min(TARGET_FRAMES, start + width)
    start = max(0, end - width)
    return start, end


def safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def safe_std(values: Sequence[float]) -> float:
    return float(np.std(values)) if values else float("nan")


def top_indices(values: np.ndarray, k: int) -> List[int]:
    k = min(k, len(values))
    return np.argsort(values)[::-1][:k].astype(int).tolist()


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    test_dir = find_clean_test_dir(dataset_root)
    samples = load_clean_test_samples(test_dir)

    print(f"Dataset: {dataset_root}")
    print(f"Clean test split: {test_dir}")
    print(f"Samples: {len(samples)}")

    _ = load_ksl_classifier()
    checkpoint_mean = extract_checkpoint_mean()

    global_frame_drops = [[] for _ in range(TARGET_FRAMES)]
    class_frame_drops = {
        label: [[] for _ in range(TARGET_FRAMES)]
        for label in EXPECTED_LABELS
    }
    global_window_drops = {
        width: [[] for _ in range(TARGET_FRAMES)]
        for width in WINDOW_SIZES
    }

    sample_records: List[Dict[str, Any]] = []
    csv_rows: List[Dict[str, Any]] = []
    clean_correct = 0

    for sample_idx, (true_label, path, raw) in enumerate(samples, start=1):
        sequence = prepare_sequence(raw)
        clean_probs = prediction_probabilities(sequence)

        clean_true_prob = float(clean_probs[true_label])
        clean_pred = max(clean_probs.items(), key=lambda kv: kv[1])[0]
        clean_correct += int(clean_pred == true_label)

        frame_drops = np.zeros(TARGET_FRAMES, dtype=np.float32)

        for frame_idx in range(TARGET_FRAMES):
            occluded = neutralize_frames(
                sequence, frame_idx, frame_idx + 1, checkpoint_mean
            )
            probs = prediction_probabilities(occluded)
            drop = clean_true_prob - float(probs[true_label])

            frame_drops[frame_idx] = drop
            global_frame_drops[frame_idx].append(drop)
            class_frame_drops[true_label][frame_idx].append(drop)

            csv_rows.append(
                {
                    "sample": path.name,
                    "true_label": true_label,
                    "clean_prediction": clean_pred,
                    "clean_true_probability": clean_true_prob,
                    "attribution_type": "single_frame",
                    "window_size": 1,
                    "center_frame": frame_idx,
                    "start_frame": frame_idx,
                    "end_frame_exclusive": frame_idx + 1,
                    "true_probability_drop": drop,
                }
            )

        window_results: Dict[str, List[float]] = {}

        for width in WINDOW_SIZES:
            width_drops = np.zeros(TARGET_FRAMES, dtype=np.float32)

            for center in range(TARGET_FRAMES):
                start, end = centered_window_bounds(center, width)
                occluded = neutralize_frames(sequence, start, end, checkpoint_mean)
                probs = prediction_probabilities(occluded)
                drop = clean_true_prob - float(probs[true_label])

                width_drops[center] = drop
                global_window_drops[width][center].append(drop)

                csv_rows.append(
                    {
                        "sample": path.name,
                        "true_label": true_label,
                        "clean_prediction": clean_pred,
                        "clean_true_probability": clean_true_prob,
                        "attribution_type": "temporal_window",
                        "window_size": width,
                        "center_frame": center,
                        "start_frame": start,
                        "end_frame_exclusive": end,
                        "true_probability_drop": drop,
                    }
                )

            window_results[str(width)] = width_drops.tolist()

        sample_records.append(
            {
                "sample": path.name,
                "true_label": true_label,
                "raw_shape": list(raw.shape),
                "clean_prediction": clean_pred,
                "clean_true_probability": clean_true_prob,
                "clean_probabilities": clean_probs,
                "single_frame_attribution": frame_drops.tolist(),
                "top_5_single_frames": top_indices(frame_drops, 5),
                "window_attribution": window_results,
            }
        )

        print(
            f"[{sample_idx:03d}/{len(samples)}] "
            f"{true_label:<6} pred={clean_pred:<6} "
            f"p(y)={clean_true_prob:.4f} "
            f"top={top_indices(frame_drops, 5)}"
        )

    global_mean = np.asarray(
        [safe_mean(v) for v in global_frame_drops], dtype=np.float32
    )
    global_std = np.asarray(
        [safe_std(v) for v in global_frame_drops], dtype=np.float32
    )

    by_class: Dict[str, Any] = {}
    for label in EXPECTED_LABELS:
        mean_profile = np.asarray(
            [safe_mean(v) for v in class_frame_drops[label]], dtype=np.float32
        )
        std_profile = np.asarray(
            [safe_std(v) for v in class_frame_drops[label]], dtype=np.float32
        )
        by_class[label] = {
            "n_samples": sum(r["true_label"] == label for r in sample_records),
            "single_frame_mean_drop": mean_profile.tolist(),
            "single_frame_std_drop": std_profile.tolist(),
            "top_5_frames": top_indices(mean_profile, 5),
        }

    window_summary: Dict[str, Any] = {}
    for width in WINDOW_SIZES:
        mean_profile = np.asarray(
            [safe_mean(v) for v in global_window_drops[width]], dtype=np.float32
        )
        std_profile = np.asarray(
            [safe_std(v) for v in global_window_drops[width]], dtype=np.float32
        )
        window_summary[str(width)] = {
            "global_mean_drop": mean_profile.tolist(),
            "global_std_drop": std_profile.tolist(),
            "top_5_centers": top_indices(mean_profile, 5),
        }

    clean_accuracy = clean_correct / len(samples)

    payload = {
        "experiment": "KSL temporal occlusion attribution",
        "dataset": KAGGLE_DATASET,
        "n_samples": len(samples),
        "labels": list(EXPECTED_LABELS),
        "clean_accuracy": clean_accuracy,
        "preprocessing": {
            "source_shape": "(T,126)",
            "temporal_method": "linear interpolation",
            "target_shape": "(64,225)",
            "feature_padding": "99 structural zeros",
            "claim": (
                "This is a deterministic reconstructed candidate. It is not "
                "claimed to be the exact original training preprocessing."
            ),
        },
        "occlusion": {
            "neutral_value": "checkpoint training mean",
            "single_frame_metric": "p_true(clean) - p_true(occluded)",
            "window_sizes": list(WINDOW_SIZES),
        },
        "global": {
            "single_frame_mean_drop": global_mean.tolist(),
            "single_frame_std_drop": global_std.tolist(),
            "top_10_frames": top_indices(global_mean, 10),
        },
        "by_class": by_class,
        "window_summary": window_summary,
        "samples": sample_records,
        "warning": (
            "Occlusion attribution measures model sensitivity. It is not "
            "linguistic ground truth, causal importance, or sign segmentation."
        ),
    }

    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample",
            "true_label",
            "clean_prediction",
            "clean_true_probability",
            "attribution_type",
            "window_size",
            "center_frame",
            "start_frame",
            "end_frame_exclusive",
            "true_probability_drop",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    frames = np.arange(TARGET_FRAMES)

    plt.figure(figsize=(11, 5))
    plt.plot(frames, global_mean)
    plt.fill_between(
        frames,
        global_mean - global_std,
        global_mean + global_std,
        alpha=0.2,
    )
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Resampled frame index")
    plt.ylabel("Mean drop in true-class probability")
    plt.title("KSL Temporal Attribution — Global Single-Frame Occlusion")
    plt.tight_layout()
    plt.savefig(GLOBAL_PLOT_PATH, dpi=180)
    plt.close()

    plt.figure(figsize=(11, 6))
    for label in EXPECTED_LABELS:
        profile = np.asarray(by_class[label]["single_frame_mean_drop"])
        plt.plot(frames, profile, label=label)
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Resampled frame index")
    plt.ylabel("Mean drop in true-class probability")
    plt.title("KSL Temporal Attribution — By Gloss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(CLASS_PLOT_PATH, dpi=180)
    plt.close()

    plt.figure(figsize=(11, 6))
    for width in WINDOW_SIZES:
        profile = np.asarray(window_summary[str(width)]["global_mean_drop"])
        plt.plot(frames, profile, label=f"window={width}")
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Window centre frame")
    plt.ylabel("Mean drop in true-class probability")
    plt.title("KSL Temporal Attribution — Window Occlusion")
    plt.legend()
    plt.tight_layout()
    plt.savefig(WINDOW_PLOT_PATH, dpi=180)
    plt.close()

    print("\nKSL TEMPORAL ATTRIBUTION COMPLETE")
    print(f"Clean accuracy: {clean_correct}/{len(samples)} = {clean_accuracy:.4f}")
    print(f"Global top-10 frames: {top_indices(global_mean, 10)}")
    for label in EXPECTED_LABELS:
        print(f"{label:<6}: {by_class[label]['top_5_frames']}")

    print("\nOutputs:")
    for path in (
        JSON_PATH,
        CSV_PATH,
        GLOBAL_PLOT_PATH,
        CLASS_PLOT_PATH,
        WINDOW_PLOT_PATH,
    ):
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()