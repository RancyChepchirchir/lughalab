"""
Expanded natural-OOD extraction for LughaLab KSL.

Uses the official TEST split of:
    ekitabu/kenyan-sign-language-videos

The experiment evaluates all 450 test videos:
    30 classes x 15 videos

For each video:
    1. Extract MediaPipe landmarks.
    2. Convert to checkpoint-compatible representation.
    3. Extract the reconstructed Transformer's CLS embedding.
    4. Record closed-set probabilities and diagnostics.

No source videos are copied into the LughaLab repository.
"""

from pathlib import Path
import sys
import time

import kagglehub
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT / "backend") not in sys.path:
    sys.path.insert(
        0,
        str(ROOT / "backend"),
    )


from app.services.ksl_landmarks import (
    extract_ksl_landmarks,
    frames_to_array,
)
from app.services.ksl_checkpoint_adapter import (
    prepare_for_legacy_checkpoint,
)
from app.services.ksl_classifier import (
    classify_checkpoint_sequence,
    extract_checkpoint_embedding,
    load_ksl_classifier,
)


RESULTS_DIR = ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "ksl_expanded_natural_ood.npz"
)

DATASET = (
    "ekitabu/"
    "kenyan-sign-language-videos"
)


def main():
    print("=" * 78)
    print(
        "LughaLab KSL Expanded Natural OOD Extraction"
    )
    print("=" * 78)

    dataset_root = Path(
        kagglehub.dataset_download(
            DATASET
        )
    )

    metadata_path = (
        dataset_root
        / "splits_info.csv"
    )

    metadata = pd.read_csv(
        metadata_path
    )

    test = (
        metadata[
            metadata["split"] == "test"
        ]
        .copy()
        .reset_index(drop=True)
    )

    print()
    print("Dataset:", dataset_root)
    print("Test videos:", len(test))
    print(
        "Classes:",
        test["class"].nunique(),
    )

    class_counts = (
        test["class"]
        .value_counts()
        .sort_index()
    )

    print()
    print("TEST CLASS COUNTS")
    print("-" * 78)

    for label, count in class_counts.items():
        print(
            f"{label:<25} "
            f"{count:>5}"
        )

    bundle = load_ksl_classifier()

    embeddings = []
    probabilities = []
    predicted_labels = []
    predicted_probabilities = []

    glosses = []
    filenames = []
    relative_paths = []

    frames_read = []
    frames_detected = []
    detection_rates = []

    failed_paths = []
    failure_messages = []

    start_time = time.time()

    total = len(test)

    for row_index, row in test.iterrows():
        relative_path = str(
            row["relative_path"]
        )

        gloss = str(
            row["class"]
        )

        video_path = (
            dataset_root
            / relative_path
        )

        number = row_index + 1

        print(
            f"[{number:03d}/{total:03d}] "
            f"{gloss:<15} "
            f"{video_path.name}",
            flush=True,
        )

        try:
            extraction = (
                extract_ksl_landmarks(
                    str(video_path)
                )
            )

            native_sequence = (
                frames_to_array(
                    extraction
                )
            )

            if (
                native_sequence.ndim != 2
                or native_sequence.shape[0] == 0
            ):
                raise ValueError(
                    "No valid landmark sequence."
                )

            adapter = (
                prepare_for_legacy_checkpoint(
                    native_sequence,
                    mean=bundle["mean"],
                    std=bundle["std"],
                )
            )

            checkpoint_sequence = (
                adapter["sequence"]
            )

            classification = (
                classify_checkpoint_sequence(
                    checkpoint_sequence
                )
            )

            embedding = (
                extract_checkpoint_embedding(
                    checkpoint_sequence
                )
            )

            probs = np.asarray(
                [
                    item["probability"]
                    for item
                    in classification[
                        "predictions"
                    ]
                ],
                dtype=np.float32,
            )

            # Predictions are returned in
            # probability-ranking order.
            # Reconstruct fixed class-index order.
            fixed_probs = np.zeros(
                len(
                    classification[
                        "supported_labels"
                    ]
                ),
                dtype=np.float32,
            )

            for item in classification[
                "predictions"
            ]:
                fixed_probs[
                    int(item["index"])
                ] = float(
                    item["probability"]
                )

            embeddings.append(
                embedding
            )

            probabilities.append(
                fixed_probs
            )

            predicted_labels.append(
                classification[
                    "predicted_label"
                ]
            )

            predicted_probabilities.append(
                classification[
                    "predicted_probability"
                ]
            )

            glosses.append(
                gloss.lower()
            )

            filenames.append(
                video_path.name
            )

            relative_paths.append(
                relative_path
            )

            frames_read.append(
                extraction[
                    "frames_read"
                ]
            )

            frames_detected.append(
                extraction[
                    "frames_detected"
                ]
            )

            detection_rates.append(
                extraction[
                    "detection_rate"
                ]
            )

        except Exception as exc:
            print(
                "    FAILED:",
                exc,
                flush=True,
            )

            failed_paths.append(
                relative_path
            )

            failure_messages.append(
                str(exc)
            )

    elapsed = time.time() - start_time

    if not embeddings:
        raise RuntimeError(
            "No videos were processed "
            "successfully."
        )

    embeddings = np.asarray(
        embeddings,
        dtype=np.float32,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32,
    )

    predicted_probabilities = np.asarray(
        predicted_probabilities,
        dtype=np.float32,
    )

    frames_read = np.asarray(
        frames_read,
        dtype=np.int32,
    )

    frames_detected = np.asarray(
        frames_detected,
        dtype=np.int32,
    )

    detection_rates = np.asarray(
        detection_rates,
        dtype=np.float32,
    )

    np.savez_compressed(
        OUTPUT_PATH,
        embeddings=embeddings,
        probabilities=probabilities,
        glosses=np.asarray(glosses),
        filenames=np.asarray(filenames),
        relative_paths=np.asarray(
            relative_paths
        ),
        predicted_labels=np.asarray(
            predicted_labels
        ),
        predicted_probabilities=(
            predicted_probabilities
        ),
        frames_read=frames_read,
        frames_detected=frames_detected,
        detection_rates=detection_rates,
        failed_paths=np.asarray(
            failed_paths
        ),
        failure_messages=np.asarray(
            failure_messages
        ),
    )

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)

    print(
        "Successful:",
        len(embeddings),
    )

    print(
        "Failed:",
        len(failed_paths),
    )

    print(
        "Embedding shape:",
        embeddings.shape,
    )

    print(
        "Probability shape:",
        probabilities.shape,
    )

    print(
        "Mean detection rate:",
        f"{detection_rates.mean():.4f}",
    )

    print(
        "Minimum detection rate:",
        f"{detection_rates.min():.4f}",
    )

    print(
        "Elapsed seconds:",
        f"{elapsed:.1f}",
    )

    print(
        "Seconds/video:",
        f"{elapsed / total:.3f}",
    )

    print()
    print(
        "CLOSED-SET PREDICTIONS"
    )
    print("-" * 78)

    labels, counts = np.unique(
        np.asarray(
            predicted_labels
        ),
        return_counts=True,
    )

    for label, count in zip(
        labels,
        counts,
    ):
        print(
            f"{label:<15} "
            f"{count:>5} "
            f"({count / len(embeddings):.4f})"
        )

    print()
    print(
        "Mean max softmax:",
        f"{predicted_probabilities.mean():.4f}",
    )

    print()
    print("Saved:")
    print(
        f"  {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()