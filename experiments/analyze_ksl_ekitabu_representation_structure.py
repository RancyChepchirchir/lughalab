from pathlib import Path
import json

import numpy as np

from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler


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
    / "ksl_ekitabu_representation_structure.json"
)


def nearest_neighbour_agreement(
    embeddings,
    labels,
):
    """
    Leave-one-out nearest-neighbour label
    agreement over a single embedding bank.
    """
    n = len(embeddings)

    correct = 0

    for i in range(n):
        distances = np.linalg.norm(
            embeddings - embeddings[i],
            axis=1,
        )

        distances[i] = np.inf

        j = np.argmin(distances)

        if labels[j] == labels[i]:
            correct += 1

    return correct / n


def cross_group_nn(
    query_embeddings,
    query_labels,
    reference_embeddings,
    reference_labels,
):
    correct = 0

    for vector, label in zip(
        query_embeddings,
        query_labels,
    ):
        distances = np.linalg.norm(
            reference_embeddings - vector,
            axis=1,
        )

        j = np.argmin(distances)

        if reference_labels[j] == label:
            correct += 1

    return correct / len(query_embeddings)


def centroid_accuracy(
    train_embeddings,
    train_labels,
    test_embeddings,
    test_labels,
):
    classes = np.unique(train_labels)

    centroids = {}

    for cls in classes:
        centroids[cls] = (
            train_embeddings[
                train_labels == cls
            ].mean(axis=0)
        )

    correct = 0

    for vector, label in zip(
        test_embeddings,
        test_labels,
    ):
        distances = {
            cls: np.linalg.norm(
                vector - centroid
            )
            for cls, centroid
            in centroids.items()
        }

        prediction = min(
            distances,
            key=distances.get,
        )

        if prediction == label:
            correct += 1

    return correct / len(test_embeddings)


def mean_pair_distance(
    embeddings,
    labels,
    same=True,
    max_pairs=200000,
    seed=42,
):
    rng = np.random.default_rng(seed)

    n = len(embeddings)

    values = []

    attempts = 0

    while (
        len(values) < max_pairs
        and attempts < max_pairs * 20
    ):
        i = rng.integers(0, n)
        j = rng.integers(0, n)

        attempts += 1

        if i == j:
            continue

        condition = (
            labels[i] == labels[j]
        )

        if condition != same:
            continue

        values.append(
            np.linalg.norm(
                embeddings[i]
                - embeddings[j]
            )
        )

    return np.asarray(
        values,
        dtype=np.float32,
    )


def summarize(values):
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "median": float(
            np.median(values)
        ),
        "std": float(values.std()),
    }


def main():
    print("=" * 78)
    print(
        "LughaLab KSL Representation "
        "Structure Analysis"
    )
    print("=" * 78)

    data = np.load(
        BANK_PATH,
        allow_pickle=True,
    )

    embeddings = data["embeddings"]
    classes = data["classes"]
    roles = data["roles"]
    splits = data["splits"]
    signers = data["signers"]

    # Restrict lexical analysis to known
    # classes so the same gloss vocabulary
    # exists across train/val/test signers.
    known_mask = np.isin(
        roles,
        [
            "known_reference",
            "known_validation",
            "known_test",
        ],
    )

    x = embeddings[known_mask]
    y_gloss = classes[known_mask]
    y_signer = signers[known_mask]
    y_split = splits[known_mask]

    train = y_split == "train"
    val = y_split == "val"
    test = y_split == "test"

    print("\nDATA")
    print("-" * 78)

    print("Known samples:", len(x))
    print(
        "Known glosses:",
        len(np.unique(y_gloss)),
    )
    print(
        "Signers:",
        len(np.unique(y_signer)),
    )

    result = {}

    # --------------------------------------------------------
    # Within-bank nearest-neighbour structure
    # --------------------------------------------------------

    print(
        "\nLEAVE-ONE-OUT 1-NN AGREEMENT"
    )
    print("-" * 78)

    gloss_nn = nearest_neighbour_agreement(
        x,
        y_gloss,
    )

    signer_nn = nearest_neighbour_agreement(
        x,
        y_signer,
    )

    split_nn = nearest_neighbour_agreement(
        x,
        y_split,
    )

    print(
        f"Gloss agreement:  {gloss_nn:.4f}"
    )
    print(
        f"Signer agreement: {signer_nn:.4f}"
    )
    print(
        f"Split agreement:  {split_nn:.4f}"
    )

    result["loo_1nn"] = {
        "gloss": float(gloss_nn),
        "signer": float(signer_nn),
        "split": float(split_nn),
    }

    # --------------------------------------------------------
    # Cross-signer lexical recognition
    # --------------------------------------------------------

    print(
        "\nCROSS-SIGNER GLOSS RETRIEVAL"
    )
    print("-" * 78)

    val_nn = cross_group_nn(
        x[val],
        y_gloss[val],
        x[train],
        y_gloss[train],
    )

    test_nn = cross_group_nn(
        x[test],
        y_gloss[test],
        x[train],
        y_gloss[train],
    )

    val_centroid = centroid_accuracy(
        x[train],
        y_gloss[train],
        x[val],
        y_gloss[val],
    )

    test_centroid = centroid_accuracy(
        x[train],
        y_gloss[train],
        x[test],
        y_gloss[test],
    )

    print(
        f"Validation 1-NN gloss: "
        f"{val_nn:.4f}"
    )

    print(
        f"Test 1-NN gloss:       "
        f"{test_nn:.4f}"
    )

    print(
        f"Validation centroid:   "
        f"{val_centroid:.4f}"
    )

    print(
        f"Test centroid:         "
        f"{test_centroid:.4f}"
    )

    print(
        "Chance gloss accuracy: "
        f"{1 / len(np.unique(y_gloss)):.4f}"
    )

    result[
        "cross_signer_gloss"
    ] = {
        "validation_1nn": float(
            val_nn
        ),
        "test_1nn": float(
            test_nn
        ),
        "validation_centroid": float(
            val_centroid
        ),
        "test_centroid": float(
            test_centroid
        ),
        "chance": float(
            1 / len(
                np.unique(y_gloss)
            )
        ),
    }

    # --------------------------------------------------------
    # Pairwise geometry
    # --------------------------------------------------------

    print(
        "\nPAIRWISE DISTANCE STRUCTURE"
    )
    print("-" * 78)

    same_gloss = mean_pair_distance(
        x,
        y_gloss,
        same=True,
    )

    different_gloss = mean_pair_distance(
        x,
        y_gloss,
        same=False,
    )

    same_signer = mean_pair_distance(
        x,
        y_signer,
        same=True,
    )

    different_signer = mean_pair_distance(
        x,
        y_signer,
        same=False,
    )

    blocks = {
        "same_gloss": same_gloss,
        "different_gloss": different_gloss,
        "same_signer": same_signer,
        "different_signer": different_signer,
    }

    result[
        "pairwise_distance"
    ] = {}

    for name, values in blocks.items():
        stats = summarize(values)

        result[
            "pairwise_distance"
        ][name] = stats

        print(
            f"{name:<20}"
            f"mean={stats['mean']:.4f} "
            f"median={stats['median']:.4f}"
        )

    gloss_effect = (
        different_gloss.mean()
        - same_gloss.mean()
    )

    signer_effect = (
        different_signer.mean()
        - same_signer.mean()
    )

    print()
    print(
        "Gloss distance effect:  "
        f"{gloss_effect:.4f}"
    )

    print(
        "Signer distance effect: "
        f"{signer_effect:.4f}"
    )

    result["distance_effects"] = {
        "gloss": float(
            gloss_effect
        ),
        "signer": float(
            signer_effect
        ),
    }

    # --------------------------------------------------------
    # Embedding norm by split
    # --------------------------------------------------------

    print(
        "\nEMBEDDING NORMS"
    )
    print("-" * 78)

    result["embedding_norms"] = {}

    for split_name in [
        "train",
        "val",
        "test",
    ]:
        mask = (
            y_split == split_name
        )

        norms = np.linalg.norm(
            x[mask],
            axis=1,
        )

        block = {
            "mean": float(
                norms.mean()
            ),
            "std": float(
                norms.std()
            ),
            "min": float(
                norms.min()
            ),
            "max": float(
                norms.max()
            ),
        }

        result[
            "embedding_norms"
        ][split_name] = block

        print(
            f"{split_name:<10}"
            f"mean={block['mean']:.4f} "
            f"std={block['std']:.4f} "
            f"min={block['min']:.4f} "
            f"max={block['max']:.4f}"
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

    print("\nSaved:")
    print(f"  {OUTPUT_JSON}")


if __name__ == "__main__":
    main()