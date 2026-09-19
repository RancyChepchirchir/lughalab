from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"

TRAIN_BANK_PATH = (
    RESULTS / "ksl_train_embedding_bank.npz"
)

TEST_BANK_PATH = (
    RESULTS / "ksl_test_embedding_bank.npz"
)

OOD_PATH = (
    RESULTS / "ksl_natural_ood.npz"
)


def describe(name, values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    print(
        f"{name:<28}"
        f"mean={values.mean():.4f}  "
        f"median={np.median(values):.4f}  "
        f"min={values.min():.4f}  "
        f"max={values.max():.4f}"
    )


def pairwise_l2(a, b):
    a = np.asarray(
        a,
        dtype=np.float64,
    )

    b = np.asarray(
        b,
        dtype=np.float64,
    )

    squared = (
        np.sum(
            a * a,
            axis=1,
        )[:, None]
        + np.sum(
            b * b,
            axis=1,
        )[None, :]
        - 2.0 * (
            a @ b.T
        )
    )

    return np.sqrt(
        np.maximum(
            squared,
            0.0,
        )
    )


def knn_score(
    reference,
    query,
    k=5,
):
    distances = pairwise_l2(
        query,
        reference,
    )

    k = min(
        k,
        reference.shape[0],
    )

    nearest = np.partition(
        distances,
        kth=k - 1,
        axis=1,
    )[:, :k]

    return nearest.mean(
        axis=1
    )


def centroid_score(
    train_embeddings,
    train_labels,
    query_embeddings,
):
    classes = np.unique(
        train_labels
    )

    centroids = np.stack(
        [
            train_embeddings[
                train_labels == label
            ].mean(
                axis=0
            )
            for label in classes
        ],
        axis=0,
    )

    distances = pairwise_l2(
        query_embeddings,
        centroids,
    )

    return distances.min(
        axis=1
    )


def rank_auc(
    id_scores,
    ood_scores,
):
    """
    AUROC where larger score means
    more out-of-distribution.

    Equivalent to the probability
    that a randomly selected OOD
    example receives a larger score
    than a randomly selected ID
    example, with half credit for ties.
    """

    iid = np.asarray(
        id_scores,
        dtype=np.float64,
    )[None, :]

    ood = np.asarray(
        ood_scores,
        dtype=np.float64,
    )[:, None]

    wins = np.sum(
        ood > iid
    )

    ties = np.sum(
        ood == iid
    )

    return float(
        (
            wins
            + 0.5 * ties
        )
        / (
            ood.size
            * iid.size
        )
    )


def derive_glosses(paths):
    """
    Natural-OOD filenames have the form:

        apple_01_signer_01.mov
        friend_03_signer_03.mov

    The first component is therefore
    the source KSL gloss.
    """

    glosses = []

    for value in paths:
        stem = Path(
            str(value)
        ).stem

        gloss = stem.split(
            "_"
        )[0]

        glosses.append(
            gloss
        )

    return np.asarray(
        glosses
    )


def main():
    print("=" * 78)
    print(
        "LughaLab KSL Natural OOD Evaluation"
    )
    print("=" * 78)

    train_bank = np.load(
        TRAIN_BANK_PATH,
        allow_pickle=True,
    )

    test_bank = np.load(
        TEST_BANK_PATH,
        allow_pickle=True,
    )

    natural = np.load(
        OOD_PATH,
        allow_pickle=True,
    )

    train = np.asarray(
        train_bank["embeddings"],
        dtype=np.float32,
    )

    test = np.asarray(
        test_bank["embeddings"],
        dtype=np.float32,
    )

    train_labels = np.asarray(
        train_bank["labels"]
    )

    test_labels = np.asarray(
        test_bank["labels"]
    )

    ood = np.asarray(
        natural["embeddings"],
        dtype=np.float32,
    )

    paths = np.asarray(
        natural["paths"]
    )

    glosses = derive_glosses(
        paths
    )

    coverage = np.asarray(
        natural["detection_rates"],
        dtype=np.float64,
    )

    predicted_labels = np.asarray(
        natural[
            "predicted_labels"
        ]
    )

    predicted_probabilities = (
        np.asarray(
            natural[
                "predicted_probabilities"
            ],
            dtype=np.float64,
        )
    )

    print()
    print("DATA")
    print("-" * 78)

    print(
        "Training ID:",
        train.shape,
    )

    print(
        "Held-out ID:",
        test.shape,
    )

    print(
        "Natural near-OOD:",
        ood.shape,
    )

    print(
        "Training classes:",
        list(
            np.unique(
                train_labels
            )
        ),
    )

    print(
        "Natural OOD glosses:",
        list(
            np.unique(
                glosses
            )
        ),
    )

    # -------------------------------------------------
    # Closed-set behaviour
    # -------------------------------------------------

    print()
    print("CLOSED-SET BEHAVIOUR ON NATURAL OOD")
    print("-" * 78)

    for label in np.unique(
        predicted_labels
    ):
        count = int(
            np.sum(
                predicted_labels
                == label
            )
        )

        print(
            f"{label:<10} "
            f"{count:>3}/"
            f"{len(predicted_labels)} "
            f"= "
            f"{count / len(predicted_labels):.4f}"
        )

    describe(
        "Max softmax probability",
        predicted_probabilities,
    )

    # -------------------------------------------------
    # 5-NN
    # -------------------------------------------------

    id_knn = knn_score(
        train,
        test,
        k=5,
    )

    ood_knn = knn_score(
        train,
        ood,
        k=5,
    )

    print()
    print("5-NN DISTANCE")
    print("-" * 78)

    describe(
        "Held-out ID",
        id_knn,
    )

    describe(
        "Natural OOD",
        ood_knn,
    )

    knn_auc = rank_auc(
        id_knn,
        ood_knn,
    )

    print()
    print(
        "Natural OOD AUROC: "
        f"{knn_auc:.4f}"
    )

    threshold_95 = float(
        np.percentile(
            id_knn,
            95.0,
        )
    )

    rejected = (
        ood_knn
        > threshold_95
    )

    id_false_reject = (
        id_knn
        > threshold_95
    )

    print(
        "95th-percentile ID threshold: "
        f"{threshold_95:.4f}"
    )

    print(
        "Held-out ID rejected: "
        f"{int(id_false_reject.sum())}/"
        f"{len(id_false_reject)} "
        f"= "
        f"{id_false_reject.mean():.4f}"
    )

    print(
        "Natural OOD rejected: "
        f"{int(rejected.sum())}/"
        f"{len(rejected)} "
        f"= "
        f"{rejected.mean():.4f}"
    )

    # -------------------------------------------------
    # Centroid
    # -------------------------------------------------

    id_centroid = centroid_score(
        train,
        train_labels,
        test,
    )

    ood_centroid = centroid_score(
        train,
        train_labels,
        ood,
    )

    print()
    print("NEAREST-CENTROID DISTANCE")
    print("-" * 78)

    describe(
        "Held-out ID",
        id_centroid,
    )

    describe(
        "Natural OOD",
        ood_centroid,
    )

    centroid_auc = rank_auc(
        id_centroid,
        ood_centroid,
    )

    print()
    print(
        "Natural OOD AUROC: "
        f"{centroid_auc:.4f}"
    )

    # -------------------------------------------------
    # Per-gloss
    # -------------------------------------------------

    print()
    print("PER-GLOSS NATURAL OOD")
    print("-" * 78)

    print(
        f"{'Gloss':<14}"
        f"{'N':>5}"
        f"{'5NN':>10}"
        f"{'Centroid':>12}"
        f"{'Reject':>10}"
        f"{'MSP':>10}"
        f"{'Coverage':>11}"
    )

    print("-" * 72)

    for gloss in sorted(
        np.unique(
            glosses
        )
    ):
        mask = (
            glosses == gloss
        )

        print(
            f"{gloss:<14}"
            f"{int(mask.sum()):>5}"
            f"{ood_knn[mask].mean():>10.4f}"
            f"{ood_centroid[mask].mean():>12.4f}"
            f"{rejected[mask].mean():>10.4f}"
            f"{predicted_probabilities[mask].mean():>10.4f}"
            f"{coverage[mask].mean():>11.4f}"
        )

    # -------------------------------------------------
    # Coverage confound
    # -------------------------------------------------

    print()
    print("LANDMARK COVERAGE CONFOUND")
    print("-" * 78)

    describe(
        "Detection rate",
        coverage,
    )

    if np.std(
        coverage
    ) > 0:
        coverage_knn_corr = float(
            np.corrcoef(
                coverage,
                ood_knn,
            )[0, 1]
        )

        coverage_centroid_corr = float(
            np.corrcoef(
                coverage,
                ood_centroid,
            )[0, 1]
        )

        print(
            "Correlation("
            "coverage, 5-NN"
            ") = "
            f"{coverage_knn_corr:.4f}"
        )

        print(
            "Correlation("
            "coverage, centroid"
            ") = "
            f"{coverage_centroid_corr:.4f}"
        )

    else:
        coverage_knn_corr = np.nan
        coverage_centroid_corr = np.nan

        print(
            "Coverage has zero variance; "
            "correlation undefined."
        )

    # -------------------------------------------------
    # MSP versus geometry
    # -------------------------------------------------

    print()
    print("SOFTMAX VS GEOMETRIC NOVELTY")
    print("-" * 78)

    if np.std(
        predicted_probabilities
    ) > 0:
        msp_knn_corr = float(
            np.corrcoef(
                predicted_probabilities,
                ood_knn,
            )[0, 1]
        )

        print(
            "Correlation("
            "max softmax, 5-NN"
            ") = "
            f"{msp_knn_corr:.4f}"
        )

    else:
        msp_knn_corr = np.nan

    # -------------------------------------------------
    # Most / least OOD-like samples
    # -------------------------------------------------

    order = np.argsort(
        ood_knn
    )

    print()
    print("NATURAL OOD CLOSEST TO TRAINING SET")
    print("-" * 78)

    for index in order[:5]:
        print(
            f"{glosses[index]:<10} "
            f"5NN={ood_knn[index]:.4f} "
            f"pred={predicted_labels[index]:<6} "
            f"p={predicted_probabilities[index]:.4f} "
            f"coverage={coverage[index]:.4f} "
            f"{Path(str(paths[index])).name}"
        )

    print()
    print("NATURAL OOD FARTHEST FROM TRAINING SET")
    print("-" * 78)

    for index in order[-5:][::-1]:
        print(
            f"{glosses[index]:<10} "
            f"5NN={ood_knn[index]:.4f} "
            f"pred={predicted_labels[index]:<6} "
            f"p={predicted_probabilities[index]:.4f} "
            f"coverage={coverage[index]:.4f} "
            f"{Path(str(paths[index])).name}"
        )

    # -------------------------------------------------
    # Save
    # -------------------------------------------------

    output = (
        RESULTS
        / "ksl_natural_ood_evaluation.npz"
    )

    np.savez_compressed(
        output,

        id_knn=id_knn,
        ood_knn=ood_knn,

        id_centroid=id_centroid,
        ood_centroid=ood_centroid,

        glosses=glosses,
        paths=paths,

        predicted_labels=(
            predicted_labels
        ),

        predicted_probabilities=(
            predicted_probabilities
        ),

        detection_rates=coverage,

        rejected=rejected,

        threshold_95=np.asarray(
            threshold_95
        ),

        knn_auroc=np.asarray(
            knn_auc
        ),

        centroid_auroc=np.asarray(
            centroid_auc
        ),

        coverage_knn_correlation=(
            np.asarray(
                coverage_knn_corr
            )
        ),

        coverage_centroid_correlation=(
            np.asarray(
                coverage_centroid_corr
            )
        ),

        msp_knn_correlation=np.asarray(
            msp_knn_corr
        ),
    )

    print()
    print(
        "Saved:",
        output,
    )


if __name__ == "__main__":
    main()