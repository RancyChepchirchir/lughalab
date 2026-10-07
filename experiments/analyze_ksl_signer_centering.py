from pathlib import Path
import json

import numpy as np

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]

BANK_PATH = (
    ROOT
    / "experiments"
    / "results"
    / "ksl_ekitabu_open_set_bank.npz"
)

OUTPUT_PATH = (
    ROOT
    / "experiments"
    / "results"
    / "ksl_signer_centering_analysis.json"
)

K = 5


def center_by_signer(
    embeddings,
    signers,
):
    centered = np.empty_like(
        embeddings
    )

    for signer in np.unique(signers):
        mask = signers == signer

        mean = embeddings[
            mask
        ].mean(
            axis=0,
            keepdims=True,
        )

        centered[mask] = (
            embeddings[mask] - mean
        )

    return centered


def knn_scores(
    query,
    reference,
    k=5,
):
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


def open_set_metrics(
    known_scores,
    unknown_scores,
):
    y_true = np.concatenate(
        [
            np.zeros(
                len(known_scores)
            ),
            np.ones(
                len(unknown_scores)
            ),
        ]
    )

    scores = np.concatenate(
        [
            known_scores,
            unknown_scores,
        ]
    )

    return {
        "auroc": float(
            roc_auc_score(
                y_true,
                scores,
            )
        ),
        "aupr": float(
            average_precision_score(
                y_true,
                scores,
            )
        ),
        "known_mean": float(
            known_scores.mean()
        ),
        "unknown_mean": float(
            unknown_scores.mean()
        ),
    }


def cross_group_nn(
    query,
    query_labels,
    reference,
    reference_labels,
):
    correct = 0

    for vector, label in zip(
        query,
        query_labels,
    ):
        distances = np.linalg.norm(
            reference - vector,
            axis=1,
        )

        prediction = reference_labels[
            np.argmin(distances)
        ]

        correct += (
            prediction == label
        )

    return (
        correct / len(query)
    )


def loo_agreement(
    embeddings,
    labels,
):
    correct = 0

    for i, vector in enumerate(
        embeddings
    ):
        distances = np.linalg.norm(
            embeddings - vector,
            axis=1,
        )

        distances[i] = np.inf

        neighbour = np.argmin(
            distances
        )

        correct += (
            labels[neighbour]
            == labels[i]
        )

    return (
        correct / len(embeddings)
    )


def evaluate_representation(
    embeddings,
    classes,
    roles,
    signers,
):
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

    known_all_mask = np.isin(
        roles,
        [
            "known_reference",
            "known_validation",
            "known_test",
        ],
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

    known_scores = knn_scores(
        known_test,
        reference,
        k=K,
    )

    unknown_scores = knn_scores(
        unknown_test,
        reference,
        k=K,
    )

    metrics = open_set_metrics(
        known_scores,
        unknown_scores,
    )

    validation_gloss = (
        cross_group_nn(
            validation,
            classes[
                validation_mask
            ],
            reference,
            classes[
                reference_mask
            ],
        )
    )

    test_gloss = (
        cross_group_nn(
            known_test,
            classes[
                known_test_mask
            ],
            reference,
            classes[
                reference_mask
            ],
        )
    )

    signer_loo = loo_agreement(
        embeddings[
            known_all_mask
        ],
        signers[
            known_all_mask
        ],
    )

    gloss_loo = loo_agreement(
        embeddings[
            known_all_mask
        ],
        classes[
            known_all_mask
        ],
    )

    metrics.update(
        {
            "validation_gloss_1nn":
                float(
                    validation_gloss
                ),
            "test_gloss_1nn":
                float(
                    test_gloss
                ),
            "signer_loo_1nn":
                float(
                    signer_loo
                ),
            "gloss_loo_1nn":
                float(
                    gloss_loo
                ),
        }
    )

    return metrics


def main():
    print("=" * 78)
    print(
        "LughaLab KSL Signer-Centering "
        "Diagnostic"
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

    raw = evaluate_representation(
        embeddings,
        classes,
        roles,
        signers,
    )

    centered = center_by_signer(
        embeddings,
        signers,
    )

    transformed = (
        evaluate_representation(
            centered,
            classes,
            roles,
            signers,
        )
    )

    print("\nRAW VS SIGNER-CENTERED")
    print("-" * 78)

    names = [
        (
            "Open-set AUROC",
            "auroc",
        ),
        (
            "Open-set AUPR",
            "aupr",
        ),
        (
            "Known 5-NN mean",
            "known_mean",
        ),
        (
            "Unknown 5-NN mean",
            "unknown_mean",
        ),
        (
            "Validation gloss 1-NN",
            "validation_gloss_1nn",
        ),
        (
            "Test gloss 1-NN",
            "test_gloss_1nn",
        ),
        (
            "LOO gloss 1-NN",
            "gloss_loo_1nn",
        ),
        (
            "LOO signer 1-NN",
            "signer_loo_1nn",
        ),
    ]

    print(
        f"{'Measure':<28}"
        f"{'Raw':>12}"
        f"{'Centered':>12}"
        f"{'Delta':>12}"
    )

    print("-" * 78)

    for label, key in names:
        delta = (
            transformed[key]
            - raw[key]
        )

        print(
            f"{label:<28}"
            f"{raw[key]:>12.4f}"
            f"{transformed[key]:>12.4f}"
            f"{delta:>+12.4f}"
        )

    result = {
        "method": (
            "Per-signer mean centering; "
            "transductive diagnostic only."
        ),
        "raw": raw,
        "signer_centered": (
            transformed
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2,
        )

    print("\nIMPORTANT")
    print("-" * 78)

    print(
        "Signer centering uses multiple "
        "samples from each test signer."
    )

    print(
        "It is therefore a transductive "
        "diagnostic, not a deployable "
        "single-sample OOD method."
    )

    print("\nSaved:")
    print(f"  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()