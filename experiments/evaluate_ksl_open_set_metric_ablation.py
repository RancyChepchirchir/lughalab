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
    / "ksl_ekitabu_open_set_metric_ablation.json"
)

OUTPUT_NPZ = (
    ROOT
    / "experiments"
    / "results"
    / "ksl_ekitabu_open_set_metric_ablation.npz"
)

K = 5
QUANTILE = 0.95


def pairwise_distance(
    query,
    reference,
    metric,
    scale=None,
):
    if metric == "euclidean":
        return np.linalg.norm(
            reference - query,
            axis=1,
        )

    if metric == "manhattan":
        return np.abs(
            reference - query
        ).sum(axis=1)

    if metric == "cosine":
        q_norm = np.linalg.norm(query)
        r_norm = np.linalg.norm(
            reference,
            axis=1,
        )

        denom = np.maximum(
            r_norm * q_norm,
            1e-12,
        )

        similarity = (
            reference @ query
        ) / denom

        return 1.0 - similarity

    if metric == "standardized_euclidean":
        if scale is None:
            raise ValueError(
                "scale is required for "
                "standardized Euclidean"
            )

        difference = (
            reference - query
        ) / scale

        return np.linalg.norm(
            difference,
            axis=1,
        )

    raise ValueError(
        f"Unknown metric: {metric}"
    )


def knn_scores(
    query,
    reference,
    metric,
    k=5,
    scale=None,
):
    scores = np.empty(
        len(query),
        dtype=np.float32,
    )

    for i, vector in enumerate(query):
        distances = pairwise_distance(
            vector,
            reference,
            metric,
            scale=scale,
        )

        nearest = np.partition(
            distances,
            k - 1,
        )[:k]

        scores[i] = nearest.mean()

    return scores


def metrics(
    known,
    unknown,
):
    y_true = np.concatenate(
        [
            np.zeros(len(known)),
            np.ones(len(unknown)),
        ]
    )

    scores = np.concatenate(
        [
            known,
            unknown,
        ]
    )

    auroc = roc_auc_score(
        y_true,
        scores,
    )

    aupr = average_precision_score(
        y_true,
        scores,
    )

    fpr, tpr, _ = roc_curve(
        y_true,
        scores,
    )

    valid = np.where(
        tpr >= 0.95
    )[0]

    fpr95 = (
        float(np.min(fpr[valid]))
        if len(valid)
        else float("nan")
    )

    return {
        "auroc": float(auroc),
        "aupr": float(aupr),
        "fpr95": fpr95,
        "known_mean": float(
            known.mean()
        ),
        "unknown_mean": float(
            unknown.mean()
        ),
    }


def main():
    print("=" * 78)
    print(
        "LughaLab KSL Open-Set "
        "Metric Ablation"
    )
    print("=" * 78)

    data = np.load(
        BANK_PATH,
        allow_pickle=True,
    )

    embeddings = data["embeddings"]
    roles = data["roles"]
    coverage = data[
        "detection_rates"
    ]

    ref_mask = (
        roles == "known_reference"
    )
    val_mask = (
        roles == "known_validation"
    )
    known_mask = (
        roles == "known_test"
    )
    unknown_mask = (
        roles == "unknown_test"
    )

    reference = embeddings[
        ref_mask
    ]
    validation = embeddings[
        val_mask
    ]
    known = embeddings[
        known_mask
    ]
    unknown = embeddings[
        unknown_mask
    ]

    ref_coverage = coverage[
        ref_mask
    ]
    val_coverage = coverage[
        val_mask
    ]
    known_coverage = coverage[
        known_mask
    ]
    unknown_coverage = coverage[
        unknown_mask
    ]

    # Scale estimated ONLY from reference data.
    scale = reference.std(
        axis=0
    )

    scale = np.where(
        scale < 1e-6,
        1.0,
        scale,
    )

    metric_names = [
        "euclidean",
        "cosine",
        "manhattan",
        "standardized_euclidean",
    ]

    result = {
        "k": K,
        "validation_quantile": QUANTILE,
        "metrics": {},
    }

    saved = {}

    print("\nMETRIC COMPARISON")
    print("-" * 78)

    print(
        f"{'Metric':<25}"
        f"{'AUROC':>10}"
        f"{'AUPR':>10}"
        f"{'FPR95':>10}"
        f"{'ID mean':>12}"
        f"{'OOD mean':>12}"
        f"{'OOD reject':>12}"
    )

    print("-" * 78)

    for metric in metric_names:
        current_scale = (
            scale
            if metric
            == "standardized_euclidean"
            else None
        )

        val_scores = knn_scores(
            validation,
            reference,
            metric,
            k=K,
            scale=current_scale,
        )

        known_scores = knn_scores(
            known,
            reference,
            metric,
            k=K,
            scale=current_scale,
        )

        unknown_scores = knn_scores(
            unknown,
            reference,
            metric,
            k=K,
            scale=current_scale,
        )

        threshold = float(
            np.quantile(
                val_scores,
                QUANTILE,
            )
        )

        known_accept = float(
            np.mean(
                known_scores
                <= threshold
            )
        )

        unknown_reject = float(
            np.mean(
                unknown_scores
                > threshold
            )
        )

        block = metrics(
            known_scores,
            unknown_scores,
        )

        block.update(
            {
                "threshold": threshold,
                "known_acceptance": (
                    known_accept
                ),
                "unknown_rejection": (
                    unknown_reject
                ),
                "score_coverage_correlation":
                    float(
                        np.corrcoef(
                            np.concatenate(
                                [
                                    known_scores,
                                    unknown_scores,
                                ]
                            ),
                            np.concatenate(
                                [
                                    known_coverage,
                                    unknown_coverage,
                                ]
                            ),
                        )[0, 1]
                    ),
            }
        )

        result["metrics"][
            metric
        ] = block

        saved[
            f"{metric}_validation"
        ] = val_scores

        saved[
            f"{metric}_known"
        ] = known_scores

        saved[
            f"{metric}_unknown"
        ] = unknown_scores

        print(
            f"{metric:<25}"
            f"{block['auroc']:>10.4f}"
            f"{block['aupr']:>10.4f}"
            f"{block['fpr95']:>10.4f}"
            f"{block['known_mean']:>12.4f}"
            f"{block['unknown_mean']:>12.4f}"
            f"{unknown_reject:>12.4f}"
        )

    # --------------------------------------------------------
    # Coverage-matched analysis
    # --------------------------------------------------------

    print(
        "\nCOVERAGE-MATCHED EUCLIDEAN"
    )
    print("-" * 78)

    # Restrict evaluation to samples with
    # reasonably strong landmark extraction.
    #
    # This threshold is NOT used as an OOD
    # decision rule. It is a diagnostic subset.
    coverage_thresholds = [
        0.50,
        0.75,
        0.90,
        0.95,
    ]

    euclidean_known = saved[
        "euclidean_known"
    ]

    euclidean_unknown = saved[
        "euclidean_unknown"
    ]

    result[
        "coverage_conditioned"
    ] = {}

    print(
        f"{'Coverage':>10}"
        f"{'Known N':>10}"
        f"{'OOD N':>10}"
        f"{'AUROC':>10}"
        f"{'AUPR':>10}"
        f"{'FPR95':>10}"
    )

    print("-" * 78)

    for threshold in coverage_thresholds:
        kmask = (
            known_coverage
            >= threshold
        )

        umask = (
            unknown_coverage
            >= threshold
        )

        if (
            kmask.sum() < 2
            or umask.sum() < 2
        ):
            continue

        block = metrics(
            euclidean_known[
                kmask
            ],
            euclidean_unknown[
                umask
            ],
        )

        block[
            "known_n"
        ] = int(kmask.sum())

        block[
            "unknown_n"
        ] = int(umask.sum())

        result[
            "coverage_conditioned"
        ][str(threshold)] = block

        print(
            f"{threshold:>10.2f}"
            f"{kmask.sum():>10}"
            f"{umask.sum():>10}"
            f"{block['auroc']:>10.4f}"
            f"{block['aupr']:>10.4f}"
            f"{block['fpr95']:>10.4f}"
        )

    # --------------------------------------------------------
    # Coverage distributions
    # --------------------------------------------------------

    print(
        "\nLANDMARK COVERAGE"
    )
    print("-" * 78)

    groups = {
        "reference": ref_coverage,
        "validation": val_coverage,
        "known_test": known_coverage,
        "unknown_test": unknown_coverage,
    }

    result[
        "coverage"
    ] = {}

    for name, values in groups.items():
        stats = {
            "mean": float(
                values.mean()
            ),
            "median": float(
                np.median(values)
            ),
            "min": float(
                values.min()
            ),
            "max": float(
                values.max()
            ),
        }

        result[
            "coverage"
        ][name] = stats

        print(
            f"{name:<18}"
            f"mean={stats['mean']:.4f} "
            f"median={stats['median']:.4f} "
            f"min={stats['min']:.4f} "
            f"max={stats['max']:.4f}"
        )

    OUTPUT_JSON.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2,
        )

    np.savez_compressed(
        OUTPUT_NPZ,
        **saved,
        known_coverage=known_coverage,
        unknown_coverage=unknown_coverage,
        validation_coverage=val_coverage,
    )

    print("\nSaved:")
    print(f"  {OUTPUT_JSON}")
    print(f"  {OUTPUT_NPZ}")


if __name__ == "__main__":
    main()