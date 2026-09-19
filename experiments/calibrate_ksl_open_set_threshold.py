from pathlib import Path

import json
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"

TRAIN_PATH = (
    RESULTS / "ksl_train_embedding_bank.npz"
)

TEST_PATH = (
    RESULTS / "ksl_test_embedding_bank.npz"
)

NATURAL_PATH = (
    RESULTS / "ksl_natural_ood.npz"
)

OUTPUT_PATH = (
    RESULTS / "ksl_open_set_calibration.json"
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
        np.sum(a * a, axis=1)[:, None]
        + np.sum(b * b, axis=1)[None, :]
        - 2.0 * (a @ b.T)
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

    nearest = np.partition(
        distances,
        kth=k - 1,
        axis=1,
    )[:, :k]

    return nearest.mean(
        axis=1
    )


def main():
    print("=" * 78)
    print(
        "LughaLab KSL Open-Set "
        "Threshold Calibration"
    )
    print("=" * 78)

    train_bank = np.load(
        TRAIN_PATH,
        allow_pickle=True,
    )

    test_bank = np.load(
        TEST_PATH,
        allow_pickle=True,
    )

    natural = np.load(
        NATURAL_PATH,
        allow_pickle=True,
    )

    train = np.asarray(
        train_bank["embeddings"],
        dtype=np.float64,
    )

    test = np.asarray(
        test_bank["embeddings"],
        dtype=np.float64,
    )

    natural_ood = np.asarray(
        natural["embeddings"],
        dtype=np.float64,
    )

    id_scores = knn_score(
        train,
        test,
        k=5,
    )

    ood_scores = knn_score(
        train,
        natural_ood,
        k=5,
    )

    quantiles = [
        90.0,
        95.0,
        97.5,
        99.0,
    ]

    print()
    print("THRESHOLD TRADE-OFF")
    print("-" * 78)

    print(
        f"{'ID quantile':>12}"
        f"{'Threshold':>12}"
        f"{'ID accept':>12}"
        f"{'ID reject':>12}"
        f"{'OOD reject':>13}"
    )

    print("-" * 61)

    results = []

    for quantile in quantiles:
        threshold = float(
            np.percentile(
                id_scores,
                quantile,
            )
        )

        id_rejected = (
            id_scores > threshold
        )

        ood_rejected = (
            ood_scores > threshold
        )

        result = {
            "id_quantile": quantile,
            "threshold": threshold,
            "id_acceptance_rate": float(
                (~id_rejected).mean()
            ),
            "id_rejection_rate": float(
                id_rejected.mean()
            ),
            "natural_ood_rejection_rate": float(
                ood_rejected.mean()
            ),
            "id_rejected_count": int(
                id_rejected.sum()
            ),
            "natural_ood_rejected_count": int(
                ood_rejected.sum()
            ),
        }

        results.append(
            result
        )

        print(
            f"{quantile:>11.1f}%"
            f"{threshold:>12.4f}"
            f"{result['id_acceptance_rate']:>12.4f}"
            f"{result['id_rejection_rate']:>12.4f}"
            f"{result['natural_ood_rejection_rate']:>13.4f}"
        )

    # Separation margin gives us another
    # useful descriptive statistic.
    max_id = float(
        id_scores.max()
    )

    min_ood = float(
        ood_scores.min()
    )

    print()
    print("EXTREME SCORE OVERLAP")
    print("-" * 78)

    print(
        f"Maximum held-out ID 5-NN: "
        f"{max_id:.4f}"
    )

    print(
        f"Minimum natural OOD 5-NN: "
        f"{min_ood:.4f}"
    )

    print(
        "Natural minimum - ID maximum: "
        f"{min_ood - max_id:.4f}"
    )

    payload = {
        "detector": "5-NN mean L2 distance",
        "k": 5,
        "training_reference_size": int(
            len(train)
        ),
        "held_out_id_size": int(
            len(test)
        ),
        "natural_ood_size": int(
            len(natural_ood)
        ),
        "id_score_mean": float(
            id_scores.mean()
        ),
        "natural_ood_score_mean": float(
            ood_scores.mean()
        ),
        "max_id_score": max_id,
        "min_natural_ood_score": min_ood,
        "extreme_margin": float(
            min_ood - max_id
        ),
        "thresholds": results,
    }

    with OUTPUT_PATH.open("w") as handle:
        json.dump(
            payload,
            handle,
            indent=2,
        )

    print()
    print("Saved:")
    print(" ", OUTPUT_PATH)


if __name__ == "__main__":
    main()