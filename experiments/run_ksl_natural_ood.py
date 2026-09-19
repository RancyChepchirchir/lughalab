from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"

DATA_ROOT = (
    ROOT
    / "data"
    / "ksl"
    / "natural_ood"
)

RESULTS = (
    ROOT
    / "experiments"
    / "results"
)

sys.path.insert(
    0,
    str(BACKEND),
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


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".webm",
}


def discover_videos():
    samples = []

    for category in [
        "near",
        "cross_language",
        "far",
    ]:
        directory = (
            DATA_ROOT / category
        )

        if not directory.exists():
            continue

        for path in sorted(
            directory.rglob("*")
        ):
            if (
                path.is_file()
                and path.suffix.lower()
                in VIDEO_EXTENSIONS
            ):
                samples.append(
                    (
                        category,
                        path,
                    )
                )

    return samples


def main():
    print("=" * 78)
    print(
        "LughaLab KSL Natural OOD Extraction"
    )
    print("=" * 78)

    samples = discover_videos()

    print()
    print(
        "Natural OOD root:",
        DATA_ROOT,
    )

    print(
        "Videos discovered:",
        len(samples),
    )

    if not samples:
        print()
        print(
            "No natural OOD videos found."
        )

        print(
            "Add videos under:"
        )

        print(
            "  data/ksl/natural_ood/near/"
        )

        print(
            "  data/ksl/natural_ood/"
            "cross_language/"
        )

        print(
            "  data/ksl/natural_ood/far/"
        )

        return

    bundle = (
        load_ksl_classifier()
    )

    mean = bundle[
        "mean"
    ]

    std = bundle[
        "std"
    ]

    embeddings = []
    logits = []
    probabilities = []

    categories = []
    paths = []

    predicted_labels = []
    predicted_probabilities = []

    frames_read_values = []
    frames_detected_values = []
    detection_rates = []

    failed_paths = []
    failure_messages = []

    print()
    print("PROCESSING")
    print("-" * 78)

    for index, (
        category,
        path,
    ) in enumerate(
        samples,
        start=1,
    ):
        relative_path = (
            path.relative_to(ROOT)
        )

        print(
            f"[{index:03d}/"
            f"{len(samples):03d}] "
            f"{category:<15} "
            f"{relative_path}"
        )

        try:
            extraction = (
                extract_ksl_landmarks(
                    str(path)
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
                    mean=mean,
                    std=std,
                )
            )

            sequence = adapter[
                "sequence"
            ]

            classification = (
                classify_checkpoint_sequence(
                    sequence
                )
            )

            embedding = (
                extract_checkpoint_embedding(
                    sequence
                )
            )

            prediction_vector = np.asarray(
                [
                    item[
                        "probability"
                    ]
                    for item
                    in classification[
                        "predictions"
                    ]
                ],
                dtype=np.float32,
            )

            #
            # IMPORTANT:
            #
            # classification["predictions"]
            # is sorted by probability,
            # not necessarily class index.
            #
            # Reconstruct canonical
            # probability order.
            #
            probability_by_index = (
                np.zeros(
                    len(
                        classification[
                            "supported_labels"
                        ]
                    ),
                    dtype=np.float32,
                )
            )

            for item in classification[
                "predictions"
            ]:
                probability_by_index[
                    item["index"]
                ] = item[
                    "probability"
                ]

            #
            # Recover logits up to an
            # additive constant using
            # log probabilities.
            #
            # For MSP this is irrelevant.
            # For energy, absolute logit
            # offset matters, so we should
            # NOT use these reconstructed
            # values for energy scoring.
            #
            pseudo_logits = np.log(
                np.maximum(
                    probability_by_index,
                    1e-12,
                )
            ).astype(
                np.float32
            )

            embeddings.append(
                embedding
            )

            probabilities.append(
                probability_by_index
            )

            logits.append(
                pseudo_logits
            )

            categories.append(
                category
            )

            paths.append(
                str(path)
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

            frames_read = int(
                extraction[
                    "frames_read"
                ]
            )

            frames_detected = int(
                extraction[
                    "frames_detected"
                ]
            )

            detection_rate = float(
                extraction[
                    "detection_rate"
                ]
            )

            frames_read_values.append(
                frames_read
            )

            frames_detected_values.append(
                frames_detected
            )

            detection_rates.append(
                detection_rate
            )

            print(
                "      prediction="
                f"{classification['predicted_label']} "
                "p="
                f"{classification['predicted_probability']:.4f} "
                "coverage="
                f"{detection_rate:.4f}"
            )

        except Exception as exc:
            print(
                "      FAILED:",
                str(exc),
            )

            failed_paths.append(
                str(path)
            )

            failure_messages.append(
                str(exc)
            )

    if not embeddings:
        raise RuntimeError(
            "All natural OOD videos failed "
            "during preprocessing."
        )

    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        RESULTS
        / "ksl_natural_ood.npz"
    )

    np.savez_compressed(
        output,

        embeddings=np.stack(
            embeddings
        ),

        probabilities=np.stack(
            probabilities
        ),

        pseudo_logits=np.stack(
            logits
        ),

        categories=np.asarray(
            categories
        ),

        paths=np.asarray(
            paths
        ),

        predicted_labels=np.asarray(
            predicted_labels
        ),

        predicted_probabilities=np.asarray(
            predicted_probabilities,
            dtype=np.float32,
        ),

        frames_read=np.asarray(
            frames_read_values,
            dtype=np.int32,
        ),

        frames_detected=np.asarray(
            frames_detected_values,
            dtype=np.int32,
        ),

        detection_rates=np.asarray(
            detection_rates,
            dtype=np.float32,
        ),

        failed_paths=np.asarray(
            failed_paths
        ),

        failure_messages=np.asarray(
            failure_messages
        ),
    )

    print()
    print("RESULT")
    print("-" * 78)

    print(
        "Successful videos:",
        len(embeddings),
    )

    print(
        "Failed videos:",
        len(failed_paths),
    )

    print(
        "Embedding shape:",
        np.stack(
            embeddings
        ).shape,
    )

    print()
    print("CATEGORY COUNTS")

    for category in sorted(
        set(categories)
    ):
        count = int(
            np.sum(
                np.asarray(
                    categories
                )
                == category
            )
        )

        print(
            f"  {category:<15}: "
            f"{count}"
        )

    print()
    print("LANDMARK COVERAGE")

    rates = np.asarray(
        detection_rates
    )

    print(
        "  mean:",
        f"{rates.mean():.4f}",
    )

    print(
        "  median:",
        f"{np.median(rates):.4f}",
    )

    print(
        "  min:",
        f"{rates.min():.4f}",
    )

    print(
        "  max:",
        f"{rates.max():.4f}",
    )

    print()
    print(
        "Saved:",
        output,
    )


if __name__ == "__main__":
    main()