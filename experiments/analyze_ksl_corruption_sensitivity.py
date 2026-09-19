from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"

BENCHMARK_PATH = (
    RESULTS / "ksl_controlled_ood.npz"
)


def load_data():
    data = np.load(
        BENCHMARK_PATH,
        allow_pickle=False,
    )

    return {
        key: data[key]
        for key in data.files
    }


def cosine_similarity(
    a,
    b,
):
    numerator = np.sum(
        a * b,
        axis=1,
    )

    denominator = (
        np.linalg.norm(
            a,
            axis=1,
        )
        * np.linalg.norm(
            b,
            axis=1,
        )
    )

    return (
        numerator
        / np.maximum(
            denominator,
            1e-12,
        )
    )


def summarize(
    name,
    values,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    print(
        f"{name:<28}"
        f"mean={values.mean():.4f}  "
        f"median={np.median(values):.4f}  "
        f"p90={np.percentile(values, 90):.4f}  "
        f"max={values.max():.4f}"
    )


def main():
    print("=" * 78)
    print(
        "LughaLab KSL Corruption Sensitivity"
    )
    print("=" * 78)

    data = load_data()

    id_embeddings = data[
        "id_embeddings"
    ].astype(np.float64)

    id_probabilities = data[
        "id_probabilities"
    ].astype(np.float64)

    id_paths = data[
        "id_paths"
    ]

    ood_embeddings = data[
        "ood_embeddings"
    ].astype(np.float64)

    ood_probabilities = data[
        "ood_probabilities"
    ].astype(np.float64)

    ood_types = data[
        "ood_types"
    ]

    ood_paths = data[
        "ood_paths"
    ]

    families = sorted(
        np.unique(
            ood_types
        ).tolist()
    )

    print()
    print(
        "ID samples:",
        len(id_embeddings),
    )

    print(
        "Corruption families:",
        families,
    )

    print()
    print(
        "PAIRED REPRESENTATION CHANGE"
    )
    print("-" * 78)

    rows = []

    for family in families:
        family_l2 = []
        family_cosine_change = []
        probability_change = []
        prediction_flip = []

        for index, path in enumerate(
            id_paths
        ):
            matches = np.where(
                (ood_types == family)
                & (ood_paths == path)
            )[0]

            if len(matches) != 1:
                raise RuntimeError(
                    "Expected exactly one "
                    f"{family} corruption for "
                    f"{path}, found "
                    f"{len(matches)}."
                )

            ood_index = int(
                matches[0]
            )

            original_embedding = (
                id_embeddings[
                    index:index + 1
                ]
            )

            corrupted_embedding = (
                ood_embeddings[
                    ood_index:
                    ood_index + 1
                ]
            )

            l2 = float(
                np.linalg.norm(
                    original_embedding
                    - corrupted_embedding
                )
            )

            cosine = float(
                cosine_similarity(
                    original_embedding,
                    corrupted_embedding,
                )[0]
            )

            original_probs = (
                id_probabilities[
                    index
                ]
            )

            corrupted_probs = (
                ood_probabilities[
                    ood_index
                ]
            )

            prob_l1 = float(
                np.abs(
                    original_probs
                    - corrupted_probs
                ).sum()
            )

            original_prediction = int(
                np.argmax(
                    original_probs
                )
            )

            corrupted_prediction = int(
                np.argmax(
                    corrupted_probs
                )
            )

            family_l2.append(
                l2
            )

            family_cosine_change.append(
                1.0 - cosine
            )

            probability_change.append(
                prob_l1
            )

            prediction_flip.append(
                original_prediction
                != corrupted_prediction
            )

        family_l2 = np.asarray(
            family_l2
        )

        family_cosine_change = np.asarray(
            family_cosine_change
        )

        probability_change = np.asarray(
            probability_change
        )

        prediction_flip = np.asarray(
            prediction_flip
        )

        print()
        print(
            family.upper()
        )

        summarize(
            "Embedding L2 change",
            family_l2,
        )

        summarize(
            "Cosine distance",
            family_cosine_change,
        )

        summarize(
            "Probability L1 change",
            probability_change,
        )

        flip_rate = float(
            prediction_flip.mean()
        )

        print(
            "Prediction flip rate:",
            f"{prediction_flip.sum()}/"
            f"{len(prediction_flip)} "
            f"= {flip_rate:.4f}",
        )

        rows.append(
            (
                family,
                float(
                    family_l2.mean()
                ),
                float(
                    family_cosine_change.mean()
                ),
                float(
                    probability_change.mean()
                ),
                flip_rate,
            )
        )

    print()
    print(
        "SUMMARY"
    )
    print("-" * 78)

    print(
        f"{'Family':<18}"
        f"{'Δ embedding':>14}"
        f"{'cos distance':>14}"
        f"{'Δ probability':>16}"
        f"{'flip rate':>12}"
    )

    print("-" * 74)

    for (
        family,
        l2,
        cosine_distance,
        probability_delta,
        flip_rate,
    ) in rows:
        print(
            f"{family:<18}"
            f"{l2:>14.4f}"
            f"{cosine_distance:>14.4f}"
            f"{probability_delta:>16.4f}"
            f"{flip_rate:>12.4f}"
        )

    output = (
        RESULTS
        / "ksl_corruption_sensitivity.npz"
    )

    np.savez_compressed(
        output,
        families=np.asarray(
            [
                row[0]
                for row in rows
            ]
        ),
        embedding_l2=np.asarray(
            [
                row[1]
                for row in rows
            ]
        ),
        cosine_distance=np.asarray(
            [
                row[2]
                for row in rows
            ]
        ),
        probability_change=np.asarray(
            [
                row[3]
                for row in rows
            ]
        ),
        prediction_flip_rate=np.asarray(
            [
                row[4]
                for row in rows
            ]
        ),
    )

    print()
    print(
        "Saved:",
        output,
    )


if __name__ == "__main__":
    main()