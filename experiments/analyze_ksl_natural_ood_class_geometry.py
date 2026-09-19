from pathlib import Path
import csv

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"

TRAIN_PATH = (
    RESULTS / "ksl_train_embedding_bank.npz"
)

OOD_PATH = (
    RESULTS / "ksl_natural_ood.npz"
)

OUTPUT_NPZ = (
    RESULTS
    / "ksl_natural_ood_class_geometry.npz"
)

OUTPUT_CSV = (
    RESULTS
    / "ksl_natural_ood_class_geometry.csv"
)


def pairwise_l2(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    squared = (
        np.sum(a * a, axis=1)[:, None]
        + np.sum(b * b, axis=1)[None, :]
        - 2.0 * (a @ b.T)
    )

    return np.sqrt(
        np.maximum(squared, 0.0)
    )


def derive_glosses(paths):
    glosses = []

    for path in paths:
        stem = Path(str(path)).stem

        # Current natural-OOD filenames:
        #
        # apple_01_signer_01.mov
        # friend_03_signer_03.mov
        #
        glosses.append(
            stem.split("_")[0]
        )

    return np.asarray(glosses)


def main():
    print("=" * 78)
    print(
        "LughaLab KSL Natural OOD "
        "Class Geometry"
    )
    print("=" * 78)

    train_bank = np.load(
        TRAIN_PATH,
        allow_pickle=True,
    )

    natural = np.load(
        OOD_PATH,
        allow_pickle=True,
    )

    train_embeddings = np.asarray(
        train_bank["embeddings"],
        dtype=np.float64,
    )

    train_labels = np.asarray(
        train_bank["labels"]
    )

    ood_embeddings = np.asarray(
        natural["embeddings"],
        dtype=np.float64,
    )

    paths = np.asarray(
        natural["paths"]
    )

    predicted_labels = np.asarray(
        natural["predicted_labels"]
    )

    predicted_probabilities = np.asarray(
        natural[
            "predicted_probabilities"
        ],
        dtype=np.float64,
    )

    detection_rates = np.asarray(
        natural["detection_rates"],
        dtype=np.float64,
    )

    glosses = derive_glosses(
        paths
    )

    classes = np.asarray(
        sorted(
            np.unique(
                train_labels
            )
        )
    )

    print()
    print("DATA")
    print("-" * 78)

    print(
        "Training embeddings:",
        train_embeddings.shape,
    )

    print(
        "Natural OOD embeddings:",
        ood_embeddings.shape,
    )

    print(
        "Known classes:",
        list(classes),
    )

    print(
        "Natural glosses:",
        list(
            sorted(
                np.unique(
                    glosses
                )
            )
        ),
    )

    # -------------------------------------------------
    # Class centroids
    # -------------------------------------------------

    centroids = np.stack(
        [
            train_embeddings[
                train_labels == label
            ].mean(axis=0)
            for label in classes
        ],
        axis=0,
    )

    centroid_distances = pairwise_l2(
        ood_embeddings,
        centroids,
    )

    nearest_centroid_indices = (
        np.argmin(
            centroid_distances,
            axis=1,
        )
    )

    nearest_centroid_labels = (
        classes[
            nearest_centroid_indices
        ]
    )

    nearest_centroid_distances = (
        centroid_distances[
            np.arange(
                len(ood_embeddings)
            ),
            nearest_centroid_indices,
        ]
    )

    # -------------------------------------------------
    # Nearest training neighbours
    # -------------------------------------------------

    train_distances = pairwise_l2(
        ood_embeddings,
        train_embeddings,
    )

    nearest_train_indices = np.argmin(
        train_distances,
        axis=1,
    )

    nearest_train_labels = (
        train_labels[
            nearest_train_indices
        ]
    )

    nearest_train_distances = (
        train_distances[
            np.arange(
                len(ood_embeddings)
            ),
            nearest_train_indices,
        ]
    )

    # -------------------------------------------------
    # Five nearest neighbours
    # -------------------------------------------------

    k = 5

    knn_indices = np.argsort(
        train_distances,
        axis=1,
    )[:, :k]

    knn_labels = train_labels[
        knn_indices
    ]

    knn_distances = np.take_along_axis(
        train_distances,
        knn_indices,
        axis=1,
    )

    knn_mean_distance = (
        knn_distances.mean(
            axis=1
        )
    )

    knn_counts = np.zeros(
        (
            len(ood_embeddings),
            len(classes),
        ),
        dtype=np.int32,
    )

    for sample_index in range(
        len(ood_embeddings)
    ):
        for class_index, label in enumerate(
            classes
        ):
            knn_counts[
                sample_index,
                class_index,
            ] = np.sum(
                knn_labels[
                    sample_index
                ]
                == label
            )

    knn_majority_labels = []

    for sample_index in range(
        len(ood_embeddings)
    ):
        counts = knn_counts[
            sample_index
        ]

        maximum = counts.max()

        winners = classes[
            counts == maximum
        ]

        # k=5 means a unique majority is
        # normally available. Keep explicit
        # tie handling nevertheless.
        if len(winners) == 1:
            knn_majority_labels.append(
                winners[0]
            )
        else:
            knn_majority_labels.append(
                "tie"
            )

    knn_majority_labels = np.asarray(
        knn_majority_labels
    )

    # -------------------------------------------------
    # Overall agreement
    # -------------------------------------------------

    print()
    print("SOFTMAX VS GEOMETRY AGREEMENT")
    print("-" * 78)

    print(
        f"{'Class':<12}"
        f"{'Classifier':>12}"
        f"{'Centroid':>12}"
        f"{'1-NN':>12}"
        f"{'5-NN maj.':>12}"
    )

    print("-" * 60)

    for label in classes:
        classifier_count = int(
            np.sum(
                predicted_labels
                == label
            )
        )

        centroid_count = int(
            np.sum(
                nearest_centroid_labels
                == label
            )
        )

        nn_count = int(
            np.sum(
                nearest_train_labels
                == label
            )
        )

        knn_count = int(
            np.sum(
                knn_majority_labels
                == label
            )
        )

        print(
            f"{label:<12}"
            f"{classifier_count:>12}"
            f"{centroid_count:>12}"
            f"{nn_count:>12}"
            f"{knn_count:>12}"
        )

    classifier_centroid_agreement = (
        predicted_labels
        == nearest_centroid_labels
    )

    classifier_nn_agreement = (
        predicted_labels
        == nearest_train_labels
    )

    classifier_knn_agreement = (
        predicted_labels
        == knn_majority_labels
    )

    print()
    print(
        "Classifier ↔ centroid agreement: "
        f"{classifier_centroid_agreement.sum()}/"
        f"{len(predicted_labels)} "
        f"= "
        f"{classifier_centroid_agreement.mean():.4f}"
    )

    print(
        "Classifier ↔ 1-NN agreement: "
        f"{classifier_nn_agreement.sum()}/"
        f"{len(predicted_labels)} "
        f"= "
        f"{classifier_nn_agreement.mean():.4f}"
    )

    print(
        "Classifier ↔ 5-NN majority agreement: "
        f"{classifier_knn_agreement.sum()}/"
        f"{len(predicted_labels)} "
        f"= "
        f"{classifier_knn_agreement.mean():.4f}"
    )

    # -------------------------------------------------
    # Per-gloss centroid behaviour
    # -------------------------------------------------

    print()
    print("GLOSS -> NEAREST CENTROID")
    print("-" * 78)

    for gloss in sorted(
        np.unique(
            glosses
        )
    ):
        mask = glosses == gloss

        print()
        print(
            gloss.upper(),
            f"(n={int(mask.sum())})",
        )

        for label in classes:
            count = int(
                np.sum(
                    nearest_centroid_labels[
                        mask
                    ]
                    == label
                )
            )

            mean_distance = (
                centroid_distances[
                    mask,
                    np.where(
                        classes == label
                    )[0][0],
                ].mean()
            )

            print(
                f"  {label:<8} "
                f"nearest={count}/"
                f"{int(mask.sum())}  "
                f"mean_distance="
                f"{mean_distance:.4f}"
            )

    # -------------------------------------------------
    # Per-gloss 5-NN composition
    # -------------------------------------------------

    print()
    print("GLOSS -> 5-NN LABEL COMPOSITION")
    print("-" * 78)

    print(
        f"{'Gloss':<14}"
        + "".join(
            f"{label:>10}"
            for label in classes
        )
        + f"{'Mean d':>12}"
    )

    print(
        "-" * (
            14
            + 10 * len(classes)
            + 12
        )
    )

    for gloss in sorted(
        np.unique(
            glosses
        )
    ):
        mask = glosses == gloss

        total_counts = (
            knn_counts[
                mask
            ].sum(
                axis=0
            )
        )

        print(
            f"{gloss:<14}"
            + "".join(
                f"{int(value):>10}"
                for value in total_counts
            )
            + (
                f"{knn_mean_distance[mask].mean():>12.4f}"
            )
        )

    # -------------------------------------------------
    # Per-sample detail
    # -------------------------------------------------

    print()
    print("SAMPLE-LEVEL GEOMETRY")
    print("-" * 78)

    for index in range(
        len(ood_embeddings)
    ):
        print(
            f"{glosses[index]:<10} "
            f"softmax="
            f"{predicted_labels[index]:<6} "
            f"p={predicted_probabilities[index]:.4f} "
            f"centroid="
            f"{nearest_centroid_labels[index]:<6} "
            f"1NN="
            f"{nearest_train_labels[index]:<6} "
            f"5NN="
            f"{knn_majority_labels[index]:<6} "
            f"d={knn_mean_distance[index]:.4f}"
        )

    # -------------------------------------------------
    # CSV
    # -------------------------------------------------

    class_to_index = {
        label: index
        for index, label
        in enumerate(classes)
    }

    with OUTPUT_CSV.open(
        "w",
        newline="",
    ) as handle:
        fieldnames = [
            "gloss",
            "path",
            "classifier_prediction",
            "classifier_probability",
            "nearest_centroid_label",
            "nearest_centroid_distance",
            "nearest_train_label",
            "nearest_train_distance",
            "knn_majority_label",
            "knn_mean_distance",
            "detection_rate",
        ]

        for label in classes:
            fieldnames.append(
                f"centroid_distance_{label}"
            )

        for label in classes:
            fieldnames.append(
                f"knn_count_{label}"
            )

        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for index in range(
            len(ood_embeddings)
        ):
            row = {
                "gloss": (
                    glosses[index]
                ),
                "path": (
                    str(paths[index])
                ),
                "classifier_prediction": (
                    predicted_labels[index]
                ),
                "classifier_probability": (
                    float(
                        predicted_probabilities[
                            index
                        ]
                    )
                ),
                "nearest_centroid_label": (
                    nearest_centroid_labels[
                        index
                    ]
                ),
                "nearest_centroid_distance": (
                    float(
                        nearest_centroid_distances[
                            index
                        ]
                    )
                ),
                "nearest_train_label": (
                    nearest_train_labels[
                        index
                    ]
                ),
                "nearest_train_distance": (
                    float(
                        nearest_train_distances[
                            index
                        ]
                    )
                ),
                "knn_majority_label": (
                    knn_majority_labels[
                        index
                    ]
                ),
                "knn_mean_distance": (
                    float(
                        knn_mean_distance[
                            index
                        ]
                    )
                ),
                "detection_rate": (
                    float(
                        detection_rates[
                            index
                        ]
                    )
                ),
            }

            for label in classes:
                class_index = (
                    class_to_index[
                        label
                    ]
                )

                row[
                    f"centroid_distance_{label}"
                ] = float(
                    centroid_distances[
                        index,
                        class_index,
                    ]
                )

                row[
                    f"knn_count_{label}"
                ] = int(
                    knn_counts[
                        index,
                        class_index,
                    ]
                )

            writer.writerow(
                row
            )

    # -------------------------------------------------
    # NPZ
    # -------------------------------------------------

    np.savez_compressed(
        OUTPUT_NPZ,

        glosses=glosses,
        paths=paths,
        classes=classes,

        centroid_distances=(
            centroid_distances
        ),

        nearest_centroid_labels=(
            nearest_centroid_labels
        ),

        nearest_centroid_distances=(
            nearest_centroid_distances
        ),

        nearest_train_labels=(
            nearest_train_labels
        ),

        nearest_train_distances=(
            nearest_train_distances
        ),

        knn_labels=knn_labels,
        knn_distances=knn_distances,
        knn_counts=knn_counts,

        knn_majority_labels=(
            knn_majority_labels
        ),

        knn_mean_distance=(
            knn_mean_distance
        ),

        classifier_predictions=(
            predicted_labels
        ),

        classifier_probabilities=(
            predicted_probabilities
        ),

        detection_rates=(
            detection_rates
        ),

        classifier_centroid_agreement=(
            classifier_centroid_agreement
        ),

        classifier_nn_agreement=(
            classifier_nn_agreement
        ),

        classifier_knn_agreement=(
            classifier_knn_agreement
        ),
    )

    print()
    print("Saved:")
    print(" ", OUTPUT_CSV)
    print(" ", OUTPUT_NPZ)


if __name__ == "__main__":
    main()