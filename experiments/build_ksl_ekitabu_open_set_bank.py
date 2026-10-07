from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.services.ksl_checkpoint_adapter import (
    prepare_for_legacy_checkpoint,
)
from app.services.ksl_classifier import (
    classify_checkpoint_sequence,
    extract_checkpoint_embedding,
    load_ksl_classifier,
)
from app.services.ksl_landmarks import (
    extract_ksl_landmarks,
    frames_to_array,
)


RESULTS = ROOT / "experiments" / "results"

MANIFEST_PATH = (
    RESULTS
    / "ksl_ekitabu_open_set_manifest.csv"
)

OUTPUT_PATH = (
    RESULTS
    / "ksl_ekitabu_open_set_bank.npz"
)

PROGRESS_PATH = (
    RESULTS
    / "ksl_ekitabu_open_set_progress.json"
)

PARTIAL_PATH = (
    RESULTS
    / "ksl_ekitabu_open_set_partial.npz"
)

VALID_ROLES = {
    "known_reference",
    "known_validation",
    "known_test",
    "unknown_test",
}

SAVE_EVERY = 25


def save_bank(
    path: Path,
    records: list[dict],
    failures: list[dict],
) -> None:
    if records:
        embeddings = np.stack(
            [
                r["embedding"]
                for r in records
            ]
        ).astype(np.float32)

        probabilities = np.stack(
            [
                r["probabilities"]
                for r in records
            ]
        ).astype(np.float32)
    else:
        embeddings = np.empty(
            (0, 256),
            dtype=np.float32,
        )

        probabilities = np.empty(
            (0, 4),
            dtype=np.float32,
        )

    np.savez_compressed(
        path,
        embeddings=embeddings,
        probabilities=probabilities,
        classes=np.asarray(
            [r["class"] for r in records]
        ),
        roles=np.asarray(
            [r["role"] for r in records]
        ),
        splits=np.asarray(
            [r["split"] for r in records]
        ),
        signers=np.asarray(
            [r["signer"] for r in records]
        ),
        filenames=np.asarray(
            [r["filename"] for r in records]
        ),
        relative_paths=np.asarray(
            [
                r["relative_path"]
                for r in records
            ]
        ),
        predicted_labels=np.asarray(
            [
                r["predicted_label"]
                for r in records
            ]
        ),
        predicted_probabilities=np.asarray(
            [
                r["predicted_probability"]
                for r in records
            ],
            dtype=np.float32,
        ),
        frames_read=np.asarray(
            [
                r["frames_read"]
                for r in records
            ],
            dtype=np.int32,
        ),
        frames_detected=np.asarray(
            [
                r["frames_detected"]
                for r in records
            ],
            dtype=np.int32,
        ),
        detection_rates=np.asarray(
            [
                r["detection_rate"]
                for r in records
            ],
            dtype=np.float32,
        ),
        failed_paths=np.asarray(
            [
                f["relative_path"]
                for f in failures
            ]
        ),
        failure_messages=np.asarray(
            [
                f["message"]
                for f in failures
            ]
        ),
    )


def load_partial():
    if not PARTIAL_PATH.exists():
        return [], [], set()

    data = np.load(
        PARTIAL_PATH,
        allow_pickle=True,
    )

    records = []

    n = len(data["relative_paths"])

    for i in range(n):
        records.append(
            {
                "embedding": (
                    data["embeddings"][i]
                ),
                "probabilities": (
                    data["probabilities"][i]
                ),
                "class": str(
                    data["classes"][i]
                ),
                "role": str(
                    data["roles"][i]
                ),
                "split": str(
                    data["splits"][i]
                ),
                "signer": str(
                    data["signers"][i]
                ),
                "filename": str(
                    data["filenames"][i]
                ),
                "relative_path": str(
                    data["relative_paths"][i]
                ),
                "predicted_label": str(
                    data[
                        "predicted_labels"
                    ][i]
                ),
                "predicted_probability": (
                    float(
                        data[
                            "predicted_probabilities"
                        ][i]
                    )
                ),
                "frames_read": int(
                    data["frames_read"][i]
                ),
                "frames_detected": int(
                    data[
                        "frames_detected"
                    ][i]
                ),
                "detection_rate": float(
                    data[
                        "detection_rates"
                    ][i]
                ),
            }
        )

    failures = [
        {
            "relative_path": str(path),
            "message": str(message),
        }
        for path, message in zip(
            data["failed_paths"],
            data["failure_messages"],
        )
    ]

    completed = {
        r["relative_path"]
        for r in records
    }

    completed.update(
        f["relative_path"]
        for f in failures
    )

    return records, failures, completed


def save_progress(
    total: int,
    records: list[dict],
    failures: list[dict],
    elapsed: float,
) -> None:
    payload = {
        "total_expected": total,
        "successful": len(records),
        "failed": len(failures),
        "processed": (
            len(records)
            + len(failures)
        ),
        "remaining": (
            total
            - len(records)
            - len(failures)
        ),
        "elapsed_seconds": elapsed,
    }

    with PROGRESS_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
        )


def main() -> None:
    print("=" * 78)
    print(
        "LughaLab eKitabu Open-Set "
        "Embedding Bank"
    )
    print("=" * 78)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            MANIFEST_PATH
        )

    dataset_root = Path(
        kagglehub.dataset_download(
            "ekitabu/"
            "kenyan-sign-language-videos"
        )
    )

    manifest = pd.read_csv(
        MANIFEST_PATH,
        dtype={"signer": str},
    )

    manifest = manifest[
        manifest["open_set_role"].isin(
            VALID_ROLES
        )
    ].copy()

    manifest = manifest.sort_values(
        [
            "open_set_role",
            "class",
            "signer",
            "filename",
        ]
    ).reset_index(drop=True)

    print(
        f"Dataset root: {dataset_root}"
    )
    print(
        f"Relevant videos: {len(manifest)}"
    )

    print("\nROLE COUNTS")
    print("-" * 78)

    print(
        manifest[
            "open_set_role"
        ].value_counts().to_string()
    )

    bundle = load_ksl_classifier()

    records, failures, completed = (
        load_partial()
    )

    if completed:
        print(
            "\nResuming from checkpoint:"
        )
        print(
            f"  completed={len(completed)}"
        )

    start = time.time()

    for row_index, row in (
        manifest.iterrows()
    ):
        relative_path = str(
            row["relative_path"]
        )

        if relative_path in completed:
            continue

        video_path = (
            dataset_root
            / relative_path
        )

        processed_before = (
            len(records)
            + len(failures)
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

            adapter = (
                prepare_for_legacy_checkpoint(
                    native_sequence,
                    mean=bundle["mean"],
                    std=bundle["std"],
                )
            )

            sequence = adapter["sequence"]

            embedding = (
                extract_checkpoint_embedding(
                    sequence
                )
            )

            classification = (
                classify_checkpoint_sequence(
                    sequence
                )
            )

            predictions = (
                classification[
                    "predictions"
                ]
            )

            probability_by_index = {
                int(item["index"]): float(
                    item["probability"]
                )
                for item in predictions
            }

            probabilities = np.asarray(
                [
                    probability_by_index[i]
                    for i in range(4)
                ],
                dtype=np.float32,
            )

            records.append(
                {
                    "embedding": embedding,
                    "probabilities": (
                        probabilities
                    ),
                    "class": str(
                        row["class"]
                    ),
                    "role": str(
                        row[
                            "open_set_role"
                        ]
                    ),
                    "split": str(
                        row["split"]
                    ),
                    "signer": str(
                        row["signer"]
                    ).zfill(2),
                    "filename": str(
                        row["filename"]
                    ),
                    "relative_path": (
                        relative_path
                    ),
                    "predicted_label": str(
                        classification[
                            "predicted_label"
                        ]
                    ),
                    "predicted_probability": (
                        float(
                            classification[
                                "predicted_probability"
                            ]
                        )
                    ),
                    "frames_read": int(
                        extraction[
                            "frames_read"
                        ]
                    ),
                    "frames_detected": int(
                        extraction[
                            "frames_detected"
                        ]
                    ),
                    "detection_rate": float(
                        extraction[
                            "detection_rate"
                        ]
                    ),
                }
            )

        except Exception as exc:
            failures.append(
                {
                    "relative_path": (
                        relative_path
                    ),
                    "message": repr(exc),
                }
            )

            print(
                "\nFAILED:",
                relative_path,
            )
            print(
                " ",
                repr(exc),
            )

        completed.add(
            relative_path
        )

        processed = (
            len(records)
            + len(failures)
        )

        if (
            processed % 10 == 0
            or processed == len(manifest)
        ):
            elapsed = (
                time.time() - start
            )

            session_processed = (
                processed
                - processed_before
            )

            print(
                f"[{processed:4d}/"
                f"{len(manifest)}] "
                f"ok={len(records)} "
                f"failed={len(failures)} "
                f"current={relative_path}"
            )

        if (
            processed % SAVE_EVERY == 0
            or processed == len(manifest)
        ):
            elapsed = (
                time.time() - start
            )

            save_bank(
                PARTIAL_PATH,
                records,
                failures,
            )

            save_progress(
                len(manifest),
                records,
                failures,
                elapsed,
            )

    elapsed = time.time() - start

    save_bank(
        OUTPUT_PATH,
        records,
        failures,
    )

    save_progress(
        len(manifest),
        records,
        failures,
        elapsed,
    )

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    print(
        f"Successful: {len(records)}"
    )
    print(
        f"Failed: {len(failures)}"
    )

    if records:
        embeddings = np.stack(
            [
                r["embedding"]
                for r in records
            ]
        )

        rates = np.asarray(
            [
                r["detection_rate"]
                for r in records
            ]
        )

        print(
            "Embedding shape:",
            embeddings.shape,
        )
        print(
            "Mean detection rate:",
            f"{rates.mean():.4f}",
        )
        print(
            "Minimum detection rate:",
            f"{rates.min():.4f}",
        )

    print(
        "Elapsed seconds:",
        f"{elapsed:.1f}",
    )

    print("\nSaved:")
    print(
        f"  {OUTPUT_PATH.resolve()}"
    )


if __name__ == "__main__":
    main()