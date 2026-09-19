"""
Expanded natural-OOD evaluation for LughaLab KSL.

Compares:
    - 572 training ID embeddings
    - 124 held-out ID embeddings
    - 450 official-test natural OOD embeddings

Evaluates k-NN embedding-distance detectors for:
    k = 1, 3, 5, 10, 20

Reports:
    - AUROC
    - AUPR
    - FPR@95TPR
    - bootstrap 95% confidence intervals
    - ID-quantile rejection operating points
    - per-gloss performance
    - landmark-coverage relationships
    - softmax / geometry relationships
"""

from pathlib import Path
import json

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"

TRAIN_PATH = RESULTS / "ksl_train_embedding_bank.npz"
TEST_PATH = RESULTS / "ksl_test_embedding_bank.npz"
OOD_PATH = RESULTS / "ksl_expanded_natural_ood.npz"

OUTPUT_NPZ = (
    RESULTS
    / "ksl_expanded_natural_ood_evaluation.npz"
)

OUTPUT_JSON = (
    RESULTS
    / "ksl_expanded_natural_ood_evaluation.json"
)

KS = [1, 3, 5, 10, 20]
QUANTILES = [0.90, 0.95, 0.975, 0.99]

BOOTSTRAPS = 2000
SEED = 42


def pairwise_l2(
    x: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    x2 = np.sum(
        x * x,
        axis=1,
        keepdims=True,
    )

    r2 = np.sum(
        reference * reference,
        axis=1,
        keepdims=True,
    ).T

    squared = (
        x2
        + r2
        - 2.0 * x @ reference.T
    )

    return np.sqrt(
        np.maximum(
            squared,
            0.0,
        )
    )


def knn_scores(
    embeddings: np.ndarray,
    reference: np.ndarray,
    k: int,
) -> np.ndarray:
    distances = pairwise_l2(
        embeddings,
        reference,
    )

    nearest = np.partition(
        distances,
        kth=k - 1,
        axis=1,
    )[:, :k]

    return nearest.mean(axis=1)


def roc_curve_manual(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
):
    scores = np.concatenate(
        [
            id_scores,
            ood_scores,
        ]
    )

    labels = np.concatenate(
        [
            np.zeros(
                len(id_scores),
                dtype=np.int32,
            ),
            np.ones(
                len(ood_scores),
                dtype=np.int32,
            ),
        ]
    )

    order = np.argsort(
        scores
    )[::-1]

    labels = labels[order]
    scores = scores[order]

    positives = int(
        labels.sum()
    )

    negatives = (
        len(labels)
        - positives
    )

    tp = 0
    fp = 0

    tpr = [0.0]
    fpr = [0.0]

    previous = None

    for score, label in zip(
        scores,
        labels,
    ):
        if (
            previous is not None
            and score != previous
        ):
            tpr.append(
                tp / positives
            )
            fpr.append(
                fp / negatives
            )

        if label == 1:
            tp += 1
        else:
            fp += 1

        previous = score

    tpr.append(1.0)
    fpr.append(1.0)

    return (
        np.asarray(fpr),
        np.asarray(tpr),
    )


def auroc(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> float:
    # Probability that a random OOD score
    # exceeds a random ID score, with ties
    # receiving half credit.
    comparisons = (
        ood_scores[:, None]
        - id_scores[None, :]
    )

    wins = np.sum(
        comparisons > 0
    )

    ties = np.sum(
        comparisons == 0
    )

    total = comparisons.size

    return float(
        (
            wins
            + 0.5 * ties
        )
        / total
    )


def aupr(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> float:
    scores = np.concatenate(
        [
            id_scores,
            ood_scores,
        ]
    )

    labels = np.concatenate(
        [
            np.zeros(
                len(id_scores),
                dtype=np.int32,
            ),
            np.ones(
                len(ood_scores),
                dtype=np.int32,
            ),
        ]
    )

    order = np.argsort(
        scores
    )[::-1]

    labels = labels[order]

    tp = np.cumsum(
        labels
    )

    fp = np.cumsum(
        1 - labels
    )

    precision = (
        tp
        / np.maximum(
            tp + fp,
            1,
        )
    )

    recall = (
        tp
        / max(
            int(labels.sum()),
            1,
        )
    )

    recall_previous = np.concatenate(
        [
            np.asarray([0.0]),
            recall[:-1],
        ]
    )

    return float(
        np.sum(
            (
                recall
                - recall_previous
            )
            * precision
        )
    )


def fpr95(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
) -> float:
    # Threshold at which at least 95% of
    # OOD examples are detected.
    threshold = np.quantile(
        ood_scores,
        0.05,
    )

    return float(
        np.mean(
            id_scores
            >= threshold
        )
    )


def bootstrap_auroc(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    rng: np.random.Generator,
):
    values = np.empty(
        BOOTSTRAPS,
        dtype=np.float64,
    )

    for index in range(
        BOOTSTRAPS
    ):
        id_sample = id_scores[
            rng.integers(
                0,
                len(id_scores),
                len(id_scores),
            )
        ]

        ood_sample = ood_scores[
            rng.integers(
                0,
                len(ood_scores),
                len(ood_scores),
            )
        ]

        values[index] = auroc(
            id_sample,
            ood_sample,
        )

    return (
        float(
            np.quantile(
                values,
                0.025,
            )
        ),
        float(
            np.quantile(
                values,
                0.975,
            )
        ),
    )


def describe(
    values: np.ndarray,
):
    return {
        "mean": float(
            np.mean(values)
        ),
        "std": float(
            np.std(values)
        ),
        "min": float(
            np.min(values)
        ),
        "median": float(
            np.median(values)
        ),
        "max": float(
            np.max(values)
        ),
    }


def correlation(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    if (
        np.std(x) == 0
        or np.std(y) == 0
    ):
        return float("nan")

    return float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )


def main():
    print("=" * 78)
    print(
        "LughaLab KSL Expanded Natural OOD Evaluation"
    )
    print("=" * 78)

    train = np.load(
        TRAIN_PATH,
        allow_pickle=True,
    )

    test = np.load(
        TEST_PATH,
        allow_pickle=True,
    )

    ood = np.load(
        OOD_PATH,
        allow_pickle=True,
    )

    train_embeddings = (
        train["embeddings"]
        .astype(np.float32)
    )

    test_embeddings = (
        test["embeddings"]
        .astype(np.float32)
    )

    ood_embeddings = (
        ood["embeddings"]
        .astype(np.float32)
    )

    glosses = ood["glosses"]
    coverage = (
        ood["detection_rates"]
        .astype(np.float32)
    )

    ood_max_softmax = (
        ood[
            "predicted_probabilities"
        ]
        .astype(np.float32)
    )

    print()
    print("DATA")
    print("-" * 78)

    print(
        "Training ID:",
        train_embeddings.shape,
    )

    print(
        "Held-out ID:",
        test_embeddings.shape,
    )

    print(
        "Natural OOD:",
        ood_embeddings.shape,
    )

    print(
        "Natural glosses:",
        len(
            np.unique(
                glosses
            )
        ),
    )

    rng = np.random.default_rng(
        SEED
    )

    results = {
        "dataset": {
            "training_id": int(
                len(train_embeddings)
            ),
            "heldout_id": int(
                len(test_embeddings)
            ),
            "natural_ood": int(
                len(ood_embeddings)
            ),
            "natural_glosses": int(
                len(
                    np.unique(
                        glosses
                    )
                )
            ),
        },
        "detectors": {},
        "per_gloss": {},
    }

    all_scores = {}

    print()
    print("K-NN OOD DETECTORS")
    print("-" * 78)

    print(
        f"{'k':>3} "
        f"{'AUROC':>9} "
        f"{'95% CI':>22} "
        f"{'AUPR':>9} "
        f"{'FPR95':>9} "
        f"{'ID mean':>10} "
        f"{'OOD mean':>10}"
    )

    print("-" * 78)

    for k in KS:
        id_scores = knn_scores(
            test_embeddings,
            train_embeddings,
            k,
        )

        ood_scores = knn_scores(
            ood_embeddings,
            train_embeddings,
            k,
        )

        all_scores[k] = (
            id_scores,
            ood_scores,
        )

        roc = auroc(
            id_scores,
            ood_scores,
        )

        precision_recall = aupr(
            id_scores,
            ood_scores,
        )

        fpr = fpr95(
            id_scores,
            ood_scores,
        )

        ci_low, ci_high = (
            bootstrap_auroc(
                id_scores,
                ood_scores,
                rng,
            )
        )

        results[
            "detectors"
        ][str(k)] = {
            "auroc": roc,
            "auroc_ci_95": [
                ci_low,
                ci_high,
            ],
            "aupr": precision_recall,
            "fpr95": fpr,
            "id_scores": describe(
                id_scores
            ),
            "ood_scores": describe(
                ood_scores
            ),
        }

        print(
            f"{k:>3d} "
            f"{roc:>9.4f} "
            f"[{ci_low:.4f}, "
            f"{ci_high:.4f}] "
            f"{precision_recall:>9.4f} "
            f"{fpr:>9.4f} "
            f"{id_scores.mean():>10.4f} "
            f"{ood_scores.mean():>10.4f}"
        )

    # Keep k=5 as the pre-existing
    # LughaLab detector for detailed analysis.
    id_5nn, ood_5nn = (
        all_scores[5]
    )

    print()
    print(
        "K=5 ID-QUANTILE OPERATING POINTS"
    )
    print("-" * 78)

    print(
        f"{'Quantile':>10} "
        f"{'Threshold':>12} "
        f"{'ID accept':>12} "
        f"{'ID reject':>12} "
        f"{'OOD reject':>12}"
    )

    operating_points = {}

    for quantile in QUANTILES:
        threshold = float(
            np.quantile(
                id_5nn,
                quantile,
            )
        )

        id_reject = float(
            np.mean(
                id_5nn > threshold
            )
        )

        ood_reject = float(
            np.mean(
                ood_5nn > threshold
            )
        )

        operating_points[
            str(quantile)
        ] = {
            "threshold": threshold,
            "id_accept": (
                1.0 - id_reject
            ),
            "id_reject": id_reject,
            "ood_reject": ood_reject,
        }

        print(
            f"{quantile:>10.3f} "
            f"{threshold:>12.4f} "
            f"{1-id_reject:>12.4f} "
            f"{id_reject:>12.4f} "
            f"{ood_reject:>12.4f}"
        )

    results[
        "k5_operating_points"
    ] = operating_points

    print()
    print(
        "K=5 SCORE OVERLAP"
    )
    print("-" * 78)

    maximum_id = float(
        np.max(
            id_5nn
        )
    )

    minimum_ood = float(
        np.min(
            ood_5nn
        )
    )

    print(
        "Maximum held-out ID:",
        f"{maximum_id:.4f}",
    )

    print(
        "Minimum natural OOD:",
        f"{minimum_ood:.4f}",
    )

    print(
        "OOD minimum - ID maximum:",
        f"{minimum_ood - maximum_id:.4f}",
    )

    results[
        "k5_score_overlap"
    ] = {
        "maximum_id": maximum_id,
        "minimum_ood": minimum_ood,
        "gap": (
            minimum_ood
            - maximum_id
        ),
    }

    print()
    print(
        "COVERAGE AND SOFTMAX"
    )
    print("-" * 78)

    coverage_corr = correlation(
        coverage,
        ood_5nn,
    )

    softmax_corr = correlation(
        ood_max_softmax,
        ood_5nn,
    )

    print(
        "Detection rate:",
        f"mean={coverage.mean():.4f}",
        f"median={np.median(coverage):.4f}",
        f"min={coverage.min():.4f}",
        f"max={coverage.max():.4f}",
    )

    print(
        "Correlation(coverage, 5-NN):",
        f"{coverage_corr:.4f}",
    )

    print(
        "Max softmax:",
        f"mean={ood_max_softmax.mean():.4f}",
        f"median={np.median(ood_max_softmax):.4f}",
        f"min={ood_max_softmax.min():.4f}",
        f"max={ood_max_softmax.max():.4f}",
    )

    print(
        "Correlation(max softmax, 5-NN):",
        f"{softmax_corr:.4f}",
    )

    results[
        "coverage"
    ] = {
        **describe(
            coverage
        ),
        "correlation_with_k5": (
            coverage_corr
        ),
    }

    results[
        "softmax"
    ] = {
        **describe(
            ood_max_softmax
        ),
        "correlation_with_k5": (
            softmax_corr
        ),
    }

    threshold_99 = float(
        np.quantile(
            id_5nn,
            0.99,
        )
    )

    print()
    print(
        "PER-GLOSS K=5 NATURAL OOD"
    )
    print("-" * 78)

    print(
        f"{'Gloss':<18} "
        f"{'N':>4} "
        f"{'Mean':>9} "
        f"{'Median':>9} "
        f"{'Min':>9} "
        f"{'Max':>9} "
        f"{'Reject':>9} "
        f"{'MSP':>9} "
        f"{'Coverage':>9}"
    )

    print("-" * 96)

    for gloss in sorted(
        np.unique(
            glosses
        )
    ):
        mask = (
            glosses == gloss
        )

        scores = (
            ood_5nn[mask]
        )

        gloss_coverage = (
            coverage[mask]
        )

        gloss_softmax = (
            ood_max_softmax[
                mask
            ]
        )

        reject = float(
            np.mean(
                scores
                > threshold_99
            )
        )

        results[
            "per_gloss"
        ][str(gloss)] = {
            "n": int(
                np.sum(mask)
            ),
            "score": describe(
                scores
            ),
            "rejection_rate_q99": (
                reject
            ),
            "mean_softmax": float(
                gloss_softmax.mean()
            ),
            "mean_coverage": float(
                gloss_coverage.mean()
            ),
        }

        print(
            f"{str(gloss):<18} "
            f"{np.sum(mask):>4d} "
            f"{scores.mean():>9.4f} "
            f"{np.median(scores):>9.4f} "
            f"{scores.min():>9.4f} "
            f"{scores.max():>9.4f} "
            f"{reject:>9.4f} "
            f"{gloss_softmax.mean():>9.4f} "
            f"{gloss_coverage.mean():>9.4f}"
        )

    print()
    print(
        "CLOSED-SET COLLAPSE"
    )
    print("-" * 78)

    prediction_labels = (
        ood["predicted_labels"]
    )

    unique_predictions, counts = (
        np.unique(
            prediction_labels,
            return_counts=True,
        )
    )

    collapse = {}

    for label, count in zip(
        unique_predictions,
        counts,
    ):
        fraction = (
            int(count)
            / len(
                prediction_labels
            )
        )

        collapse[
            str(label)
        ] = {
            "count": int(count),
            "fraction": float(
                fraction
            ),
        }

        print(
            f"{str(label):<12} "
            f"{int(count):>4}/"
            f"{len(prediction_labels)} "
            f"= {fraction:.4f}"
        )

    results[
        "closed_set_predictions"
    ] = collapse

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            results,
            handle,
            indent=2,
        )

    np.savez_compressed(
        OUTPUT_NPZ,
        id_1nn=all_scores[1][0],
        ood_1nn=all_scores[1][1],
        id_3nn=all_scores[3][0],
        ood_3nn=all_scores[3][1],
        id_5nn=all_scores[5][0],
        ood_5nn=all_scores[5][1],
        id_10nn=all_scores[10][0],
        ood_10nn=all_scores[10][1],
        id_20nn=all_scores[20][0],
        ood_20nn=all_scores[20][1],
        glosses=glosses,
        detection_rates=coverage,
        max_softmax=ood_max_softmax,
    )

    print()
    print("Saved:")
    print(
        f"  {OUTPUT_JSON}"
    )
    print(
        f"  {OUTPUT_NPZ}"
    )


if __name__ == "__main__":
    main()