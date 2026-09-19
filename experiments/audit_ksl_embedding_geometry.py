from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"

TRAIN_PATH = (
    RESULTS
    / "ksl_train_embedding_bank.npz"
)

TEST_PATH = (
    RESULTS
    / "ksl_test_embedding_bank.npz"
)


def load_bank(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Embedding bank not found: {path}"
        )

    data = np.load(
        path,
        allow_pickle=False,
    )

    return {
        key: data[key]
        for key in data.files
    }


def l2_normalize(
    x: np.ndarray,
) -> np.ndarray:
    norms = np.linalg.norm(
        x,
        axis=1,
        keepdims=True,
    )

    norms = np.maximum(
        norms,
        1e-12,
    )

    return x / norms


def compute_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray,
):
    classes = sorted(
        np.unique(labels).tolist()
    )

    centroids = {}

    for label in classes:
        centroids[label] = (
            embeddings[
                labels == label
            ].mean(axis=0)
        )

    return classes, centroids


def nearest_centroid_predict(
    embeddings: np.ndarray,
    classes,
    centroids,
):
    centroid_matrix = np.stack(
        [
            centroids[label]
            for label in classes
        ]
    )

    distances = np.linalg.norm(
        embeddings[:, None, :]
        - centroid_matrix[None, :, :],
        axis=2,
    )

    indices = distances.argmin(
        axis=1
    )

    predictions = np.asarray(
        [
            classes[index]
            for index in indices
        ]
    )

    minimum_distances = (
        distances[
            np.arange(
                len(embeddings)
            ),
            indices,
        ]
    )

    return (
        predictions,
        minimum_distances,
    )


def cosine_centroid_predict(
    embeddings: np.ndarray,
    classes,
    centroids,
):
    x = l2_normalize(
        embeddings
    )

    centroid_matrix = np.stack(
        [
            centroids[label]
            for label in classes
        ]
    )

    centroid_matrix = l2_normalize(
        centroid_matrix
    )

    similarities = (
        x @ centroid_matrix.T
    )

    indices = similarities.argmax(
        axis=1
    )

    predictions = np.asarray(
        [
            classes[index]
            for index in indices
        ]
    )

    maximum_similarity = (
        similarities[
            np.arange(
                len(embeddings)
            ),
            indices,
        ]
    )

    return (
        predictions,
        maximum_similarity,
    )


def nearest_train_neighbor(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    test_embeddings: np.ndarray,
):
    predictions = []
    distances = []

    for embedding in test_embeddings:
        distance = np.linalg.norm(
            train_embeddings
            - embedding[None, :],
            axis=1,
        )

        index = int(
            np.argmin(distance)
        )

        predictions.append(
            train_labels[index]
        )

        distances.append(
            float(distance[index])
        )

    return (
        np.asarray(predictions),
        np.asarray(
            distances,
            dtype=np.float32,
        ),
    )


def pca_fit_transform(
    train_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
):
    mean = train_embeddings.mean(
        axis=0,
        keepdims=True,
    )

    centered_train = (
        train_embeddings - mean
    )

    centered_test = (
        test_embeddings - mean
    )

    _, singular_values, vt = (
        np.linalg.svd(
            centered_train,
            full_matrices=False,
        )
    )

    variance = (
        singular_values ** 2
    )

    explained_ratio = (
        variance
        / variance.sum()
    )

    train_projection = (
        centered_train
        @ vt[:2].T
    )

    test_projection = (
        centered_test
        @ vt[:2].T
    )

    return (
        explained_ratio,
        train_projection,
        test_projection,
    )


def summarize_vector(
    name: str,
    values: np.ndarray,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    print(
        f"{name:<32}"
        f"mean={values.mean():.4f}  "
        f"std={values.std():.4f}  "
        f"min={values.min():.4f}  "
        f"median={np.median(values):.4f}  "
        f"max={values.max():.4f}"
    )


def main():
    print("=" * 78)
    print(
        "LughaLab KSL Embedding Geometry Audit"
    )
    print("=" * 78)

    train = load_bank(
        TRAIN_PATH
    )

    test = load_bank(
        TEST_PATH
    )

    x_train = train[
        "embeddings"
    ].astype(np.float32)

    y_train = train[
        "labels"
    ]

    x_test = test[
        "embeddings"
    ].astype(np.float32)

    y_test = test[
        "labels"
    ]

    print()
    print("DATA")
    print("-" * 78)

    print(
        "Train:",
        x_train.shape,
    )

    print(
        "Test:",
        x_test.shape,
    )

    print(
        "Classes:",
        sorted(
            np.unique(
                y_train
            ).tolist()
        ),
    )

    print()
    print("EMBEDDING NORMS")
    print("-" * 78)

    summarize_vector(
        "Train norm",
        np.linalg.norm(
            x_train,
            axis=1,
        ),
    )

    summarize_vector(
        "Test norm",
        np.linalg.norm(
            x_test,
            axis=1,
        ),
    )

    classes, centroids = (
        compute_centroids(
            x_train,
            y_train,
        )
    )

    print()
    print("CLASS CENTROIDS")
    print("-" * 78)

    for label in classes:
        class_embeddings = (
            x_train[
                y_train == label
            ]
        )

        centroid = centroids[
            label
        ]

        distances = np.linalg.norm(
            class_embeddings
            - centroid[None, :],
            axis=1,
        )

        print(
            f"{label:<8} "
            f"n={len(class_embeddings):<4} "
            f"centroid_norm="
            f"{np.linalg.norm(centroid):.4f} "
            f"within_distance="
            f"{distances.mean():.4f}"
        )

    print()
    print("BETWEEN-CENTROID DISTANCES")
    print("-" * 78)

    for i, label_a in enumerate(
        classes
    ):
        for label_b in classes[
            i + 1:
        ]:
            distance = np.linalg.norm(
                centroids[label_a]
                - centroids[label_b]
            )

            cosine = float(
                np.dot(
                    centroids[label_a],
                    centroids[label_b],
                )
                /
                (
                    np.linalg.norm(
                        centroids[label_a]
                    )
                    * np.linalg.norm(
                        centroids[label_b]
                    )
                    + 1e-12
                )
            )

            print(
                f"{label_a:<8} ↔ "
                f"{label_b:<8} "
                f"L2={distance:.4f} "
                f"cos={cosine:.4f}"
            )

    print()
    print("NEAREST CENTROID — L2")
    print("-" * 78)

    (
        centroid_predictions,
        centroid_distances,
    ) = nearest_centroid_predict(
        x_test,
        classes,
        centroids,
    )

    centroid_accuracy = np.mean(
            centroid_predictions
            == y_test
        )

    centroid_correct = int(
    np.sum(
        centroid_predictions
        == y_test
    )
)

    print(
        "Accuracy:",
        f"{centroid_correct}/{len(y_test)} "
        f"= {centroid_accuracy:.4f}",
    )

    summarize_vector(
        "Nearest centroid distance",
        centroid_distances,
    )

    print()
    print("NEAREST CENTROID — COSINE")
    print("-" * 78)

    (
        cosine_predictions,
        cosine_similarities,
    ) = cosine_centroid_predict(
        x_test,
        classes,
        centroids,
    )

    cosine_accuracy = np.mean(
        cosine_predictions
        == y_test
    )

    cosine_correct = int(
        np.sum(
            cosine_predictions
            == y_test
        )
    )

    print(
        "Accuracy:",
        f"{cosine_correct}/{len(y_test)} "
        f"= {cosine_accuracy:.4f}",
    )

    summarize_vector(
        "Maximum centroid cosine",
        cosine_similarities,
    )

    print()
    print("1-NEAREST TRAINING NEIGHBOUR")
    print("-" * 78)

    (
        knn_predictions,
        knn_distances,
    ) = nearest_train_neighbor(
        x_train,
        y_train,
        x_test,
    )

    knn_accuracy = np.mean(
        knn_predictions
        == y_test
    )

    knn_correct = int(
        np.sum(
            knn_predictions
            == y_test
        )
    )

    print(
        "Accuracy:",
        f"{knn_correct}/{len(y_test)} "
        f"= {knn_accuracy:.4f}",
    )

    summarize_vector(
        "Nearest-neighbour distance",
        knn_distances,
    )

    print()
    print("PCA")
    print("-" * 78)

    (
        explained_ratio,
        train_pca,
        test_pca,
    ) = pca_fit_transform(
        x_train,
        x_test,
    )

    print(
        "PC1 explained variance:",
        f"{explained_ratio[0]:.4f}",
    )

    print(
        "PC2 explained variance:",
        f"{explained_ratio[1]:.4f}",
    )

    print(
        "PC1 + PC2:",
        f"{explained_ratio[:2].sum():.4f}",
    )

    cumulative = np.cumsum(
        explained_ratio
    )

    dimensions_90 = int(
        np.searchsorted(
            cumulative,
            0.90,
        )
        + 1
    )

    dimensions_95 = int(
        np.searchsorted(
            cumulative,
            0.95,
        )
        + 1
    )

    print(
        "Dimensions for 90% variance:",
        dimensions_90,
    )

    print(
        "Dimensions for 95% variance:",
        dimensions_95,
    )

    output = (
        RESULTS
        / "ksl_embedding_geometry.npz"
    )

    np.savez_compressed(
        output,
        classes=np.asarray(
            classes
        ),
        centroid_labels=np.asarray(
            classes
        ),
        centroids=np.stack(
            [
                centroids[label]
                for label in classes
            ]
        ),
        train_pca=train_pca,
        test_pca=test_pca,
        train_labels=y_train,
        test_labels=y_test,
        centroid_distances=(
            centroid_distances
        ),
        cosine_similarities=(
            cosine_similarities
        ),
        knn_distances=knn_distances,
        explained_variance_ratio=(
            explained_ratio
        ),
    )

    print()
    print(
        "Saved:",
        output,
    )


if __name__ == "__main__":
    main()