from pathlib import Path
import json

import numpy as np

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    roc_curve,
)


ROOT = Path(__file__).resolve().parents[1]

BANK_PATH = (
    ROOT
    / "experiments"
    / "results"
    / "ksl_ekitabu_open_set_bank.npz"
)

OUTPUT_JSON = (
    ROOT
    / "experiments"
    / "results"
    / "ksl_ekitabu_open_set_evaluation.json"
)

OUTPUT_NPZ = (
    ROOT
    / "experiments"
    / "results"
    / "ksl_ekitabu_open_set_evaluation.npz"
)

K_VALUES = [1, 3, 5, 10, 20]

QUANTILES = [
    0.90,
    0.95,
    0.975,
    0.99,
]

BOOTSTRAP_ITERATIONS = 2000
RNG_SEED = 42


def mean_knn_distance(
    query,
    reference,
    k,
):
    """
    Mean Euclidean distance to the k nearest
    reference embeddings.
    """

    scores = np.empty(
        len(query),
        dtype=np.float32,
    )

    for i, vector in enumerate(query):
        distances = np.linalg.norm(
            reference - vector,
            axis=1,
        )

        nearest = np.partition(
            distances,
            k - 1,
        )[:k]

        scores[i] = nearest.mean()

    return scores


def bootstrap_auroc_ci(
    known_scores,
    unknown_scores,
    iterations=2000,
    seed=42,
):
    rng = np.random.default_rng(seed)

    values = []

    n_known = len(known_scores)
    n_unknown = len(unknown_scores)

    for _ in range(iterations):
        known_sample = rng.choice(
            known_scores,
            size=n_known,
            replace=True,
        )

        unknown_sample = rng.choice(
            unknown_scores,
            size=n_unknown,
            replace=True,
        )

        y_true = np.concatenate(
            [
                np.zeros(n_known),
                np.ones(n_unknown),
            ]
        )

        y_score = np.concatenate(
            [
                known_sample,
                unknown_sample,
            ]
        )

        values.append(
            roc_auc_score(
                y_true,
                y_score,
            )
        )

    lower, upper = np.quantile(
        values,
        [0.025, 0.975],
    )

    return float(lower), float(upper)


def calculate_fpr95(
    known_scores,
    unknown_scores,
):
    """
    False-positive rate on known samples when
    true-positive rate for unknown detection
    reaches at least 95%.
    """

    y_true = np.concatenate(
        [
            np.zeros(len(known_scores)),
            np.ones(len(unknown_scores)),
        ]
    )

    y_score = np.concatenate(
        [
            known_scores,
            unknown_scores,
        ]
    )

    fpr, tpr, _ = roc_curve(
        y_true,
        y_score,
    )

    valid = np.where(
        tpr >= 0.95
    )[0]

    if len(valid) == 0:
        return float("nan")

    return float(
        np.min(fpr[valid])
    )


def metric_block(
    known_scores,
    unknown_scores,
):
    y_true = np.concatenate(
        [
            np.zeros(len(known_scores)),
            np.ones(len(unknown_scores)),
        ]
    )

    y_score = np.concatenate(
        [
            known_scores,
            unknown_scores,
        ]
    )

    auroc = roc_auc_score(
        y_true,
        y_score,
    )

    aupr = average_precision_score(
        y_true,
        y_score,
    )

    lower, upper = bootstrap_auroc_ci(
        known_scores,
        unknown_scores,
        iterations=BOOTSTRAP_ITERATIONS,
        seed=RNG_SEED,
    )

    return {
        "auroc": float(auroc),
        "auroc_ci_95": [
            lower,
            upper,
        ],
        "aupr": float(aupr),
        "fpr95": calculate_fpr95(
            known_scores,
            unknown_scores,
        ),
        "known_mean": float(
            known_scores.mean()
        ),
        "known_median": float(
            np.median(known_scores)
        ),
        "known_min": float(
            known_scores.min()
        ),
        "known_max": float(
            known_scores.max()
        ),
        "unknown_mean": float(
            unknown_scores.mean()
        ),
        "unknown_median": float(
            np.median(unknown_scores)
        ),
        "unknown_min": float(
            unknown_scores.min()
        ),
        "unknown_max": float(
            unknown_scores.max()
        ),
    }


def main():
    print("=" * 78)
    print(
        "LughaLab eKitabu Signer-Disjoint "
        "Open-Set Evaluation"
    )
    print("=" * 78)

    data = np.load(
        BANK_PATH,
        allow_pickle=True,
    )

    embeddings = data["embeddings"]
    classes = data["classes"]
    roles = data["roles"]
    signers = data["signers"]
    detection_rates = data[
        "detection_rates"
    ]
    max_softmax = data[
        "predicted_probabilities"
    ]

    reference_mask = (
        roles == "known_reference"
    )

    validation_mask = (
        roles == "known_validation"
    )

    known_test_mask = (
        roles == "known_test"
    )

    unknown_test_mask = (
        roles == "unknown_test"
    )

    reference = embeddings[
        reference_mask
    ]

    validation = embeddings[
        validation_mask
    ]

    known_test = embeddings[
        known_test_mask
    ]

    unknown_test = embeddings[
        unknown_test_mask
    ]

    print("\nDATA")
    print("-" * 78)

    print(
        "Reference:",
        reference.shape,
    )
    print(
        "Validation:",
        validation.shape,
    )
    print(
        "Known test:",
        known_test.shape,
    )
    print(
        "Unknown test:",
        unknown_test.shape,
    )

    results = {
        "protocol": {
            "reference_samples": int(
                reference_mask.sum()
            ),
            "validation_samples": int(
                validation_mask.sum()
            ),
            "known_test_samples": int(
                known_test_mask.sum()
            ),
            "unknown_test_samples": int(
                unknown_test_mask.sum()
            ),
            "known_classes": sorted(
                np.unique(
                    classes[
                        reference_mask
                    ]
                ).tolist()
            ),
            "unknown_classes": sorted(
                np.unique(
                    classes[
                        unknown_test_mask
                    ]
                ).tolist()
            ),
            "reference_signers": sorted(
                np.unique(
                    signers[
                        reference_mask
                    ]
                ).tolist()
            ),
            "validation_signers": sorted(
                np.unique(
                    signers[
                        validation_mask
                    ]
                ).tolist()
            ),
            "test_signers": sorted(
                np.unique(
                    signers[
                        known_test_mask
                        | unknown_test_mask
                    ]
                ).tolist()
            ),
        },
        "detectors": {},
        "operating_points": {},
        "per_unknown_gloss": {},
        "per_test_signer": {},
    }

    saved_arrays = {}

    print(
        "\nK-NN OPEN-SET DETECTORS"
    )
    print("-" * 78)

    print(
        f"{'k':>3} "
        f"{'AUROC':>8} "
        f"{'95% CI':>21} "
        f"{'AUPR':>8} "
        f"{'FPR95':>8} "
        f"{'Known':>10} "
        f"{'Unknown':>10}"
    )

    print("-" * 78)

    score_cache = {}

    for k in K_VALUES:
        validation_scores = (
            mean_knn_distance(
                validation,
                reference,
                k,
            )
        )

        known_scores = (
            mean_knn_distance(
                known_test,
                reference,
                k,
            )
        )

        unknown_scores = (
            mean_knn_distance(
                unknown_test,
                reference,
                k,
            )
        )

        score_cache[k] = {
            "validation": validation_scores,
            "known": known_scores,
            "unknown": unknown_scores,
        }

        metrics = metric_block(
            known_scores,
            unknown_scores,
        )

        results["detectors"][
            str(k)
        ] = metrics

        saved_arrays[
            f"validation_{k}nn"
        ] = validation_scores

        saved_arrays[
            f"known_test_{k}nn"
        ] = known_scores

        saved_arrays[
            f"unknown_test_{k}nn"
        ] = unknown_scores

        ci = metrics[
            "auroc_ci_95"
        ]

        print(
            f"{k:>3} "
            f"{metrics['auroc']:>8.4f} "
            f"[{ci[0]:.4f}, "
            f"{ci[1]:.4f}] "
            f"{metrics['aupr']:>8.4f} "
            f"{metrics['fpr95']:>8.4f} "
            f"{metrics['known_mean']:>10.4f} "
            f"{metrics['unknown_mean']:>10.4f}"
        )

    # --------------------------------------------------------
    # Primary detector: k = 5
    # --------------------------------------------------------

    k = 5

    val_scores = score_cache[k][
        "validation"
    ]

    known_scores = score_cache[k][
        "known"
    ]

    unknown_scores = score_cache[k][
        "unknown"
    ]

    print(
        "\nK=5 VALIDATION-CALIBRATED "
        "OPERATING POINTS"
    )
    print("-" * 78)

    print(
        f"{'Quantile':>10} "
        f"{'Threshold':>12} "
        f"{'Val accept':>12} "
        f"{'Known accept':>14} "
        f"{'Known reject':>14} "
        f"{'OOD reject':>12}"
    )

    for q in QUANTILES:
        threshold = float(
            np.quantile(
                val_scores,
                q,
            )
        )

        val_accept = float(
            np.mean(
                val_scores
                <= threshold
            )
        )

        known_accept = float(
            np.mean(
                known_scores
                <= threshold
            )
        )

        known_reject = (
            1.0
            - known_accept
        )

        unknown_reject = float(
            np.mean(
                unknown_scores
                > threshold
            )
        )

        results[
            "operating_points"
        ][str(q)] = {
            "threshold": threshold,
            "validation_acceptance": (
                val_accept
            ),
            "known_test_acceptance": (
                known_accept
            ),
            "known_test_rejection": (
                known_reject
            ),
            "unknown_test_rejection": (
                unknown_reject
            ),
        }

        print(
            f"{q:>10.3f} "
            f"{threshold:>12.4f} "
            f"{val_accept:>12.4f} "
            f"{known_accept:>14.4f} "
            f"{known_reject:>14.4f} "
            f"{unknown_reject:>12.4f}"
        )

    overlap = {
        "known_test_max": float(
            known_scores.max()
        ),
        "unknown_test_min": float(
            unknown_scores.min()
        ),
        "margin": float(
            unknown_scores.min()
            - known_scores.max()
        ),
    }

    results[
        "score_overlap"
    ] = overlap

    print(
        "\nK=5 SCORE OVERLAP"
    )
    print("-" * 78)

    print(
        "Maximum known-test score:",
        f"{overlap['known_test_max']:.4f}",
    )

    print(
        "Minimum unknown-test score:",
        f"{overlap['unknown_test_min']:.4f}",
    )

    print(
        "Unknown min - known max:",
        f"{overlap['margin']:.4f}",
    )

    # --------------------------------------------------------
    # Per unknown gloss
    # --------------------------------------------------------

    unknown_classes = classes[
        unknown_test_mask
    ]

    unknown_detection = (
        detection_rates[
            unknown_test_mask
        ]
    )

    unknown_softmax = (
        max_softmax[
            unknown_test_mask
        ]
    )

    primary_threshold = float(
        np.quantile(
            val_scores,
            0.95,
        )
    )

    print(
        "\nPER-UNKNOWN-GLOSS K=5"
    )
    print("-" * 78)

    print(
        f"{'Gloss':<15} "
        f"{'N':>4} "
        f"{'Mean':>9} "
        f"{'Median':>9} "
        f"{'Min':>9} "
        f"{'Max':>9} "
        f"{'Reject':>9} "
        f"{'MSP':>9} "
        f"{'Coverage':>9}"
    )

    print("-" * 78)

    for gloss in sorted(
        np.unique(
            unknown_classes
        )
    ):
        mask = (
            unknown_classes
            == gloss
        )

        scores = unknown_scores[
            mask
        ]

        rejection = float(
            np.mean(
                scores
                > primary_threshold
            )
        )

        item = {
            "n": int(mask.sum()),
            "mean": float(
                scores.mean()
            ),
            "median": float(
                np.median(scores)
            ),
            "min": float(
                scores.min()
            ),
            "max": float(
                scores.max()
            ),
            "rejection_at_val95": (
                rejection
            ),
            "mean_max_softmax": float(
                unknown_softmax[
                    mask
                ].mean()
            ),
            "mean_detection_rate": float(
                unknown_detection[
                    mask
                ].mean()
            ),
        }

        results[
            "per_unknown_gloss"
        ][str(gloss)] = item

        print(
            f"{str(gloss):<15} "
            f"{item['n']:>4} "
            f"{item['mean']:>9.4f} "
            f"{item['median']:>9.4f} "
            f"{item['min']:>9.4f} "
            f"{item['max']:>9.4f} "
            f"{item['rejection_at_val95']:>9.4f} "
            f"{item['mean_max_softmax']:>9.4f} "
            f"{item['mean_detection_rate']:>9.4f}"
        )

    # --------------------------------------------------------
    # Per test signer
    # --------------------------------------------------------

    known_signers = signers[
        known_test_mask
    ]

    unknown_signers = signers[
        unknown_test_mask
    ]

    print(
        "\nPER-TEST-SIGNER K=5"
    )
    print("-" * 78)

    for signer in sorted(
        np.unique(
            np.concatenate(
                [
                    known_signers,
                    unknown_signers,
                ]
            )
        )
    ):
        kmask = (
            known_signers
            == signer
        )

        umask = (
            unknown_signers
            == signer
        )

        signer_known = (
            known_scores[
                kmask
            ]
        )

        signer_unknown = (
            unknown_scores[
                umask
            ]
        )

        metrics = metric_block(
            signer_known,
            signer_unknown,
        )

        known_accept = float(
            np.mean(
                signer_known
                <= primary_threshold
            )
        )

        unknown_reject = float(
            np.mean(
                signer_unknown
                > primary_threshold
            )
        )

        metrics[
            "known_acceptance_at_val95"
        ] = known_accept

        metrics[
            "unknown_rejection_at_val95"
        ] = unknown_reject

        results[
            "per_test_signer"
        ][str(signer)] = metrics

        print(
            f"Signer {signer}: "
            f"AUROC="
            f"{metrics['auroc']:.4f} "
            f"known_accept="
            f"{known_accept:.4f} "
            f"unknown_reject="
            f"{unknown_reject:.4f}"
        )

    # --------------------------------------------------------
    # Confound diagnostics
    # --------------------------------------------------------

    all_test_scores = np.concatenate(
        [
            known_scores,
            unknown_scores,
        ]
    )

    all_test_detection = np.concatenate(
        [
            detection_rates[
                known_test_mask
            ],
            detection_rates[
                unknown_test_mask
            ],
        ]
    )

    all_test_softmax = np.concatenate(
        [
            max_softmax[
                known_test_mask
            ],
            max_softmax[
                unknown_test_mask
            ],
        ]
    )

    coverage_corr = float(
        np.corrcoef(
            all_test_detection,
            all_test_scores,
        )[0, 1]
    )

    softmax_corr = float(
        np.corrcoef(
            all_test_softmax,
            all_test_scores,
        )[0, 1]
    )

    results["diagnostics"] = {
        "coverage_score_correlation": (
            coverage_corr
        ),
        "softmax_score_correlation": (
            softmax_corr
        ),
    }

    print(
        "\nDIAGNOSTICS"
    )
    print("-" * 78)

    print(
        "Correlation("
        "coverage, 5-NN):",
        f"{coverage_corr:.4f}",
    )

    print(
        "Correlation("
        "max softmax, 5-NN):",
        f"{softmax_corr:.4f}",
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        **saved_arrays,
        known_test_classes=classes[
            known_test_mask
        ],
        unknown_test_classes=classes[
            unknown_test_mask
        ],
        known_test_signers=signers[
            known_test_mask
        ],
        unknown_test_signers=signers[
            unknown_test_mask
        ],
    )

    print(
        "\nSaved:"
    )
    print(
        f"  {OUTPUT_JSON}"
    )
    print(
        f"  {OUTPUT_NPZ}"
    )


if __name__ == "__main__":
    main()