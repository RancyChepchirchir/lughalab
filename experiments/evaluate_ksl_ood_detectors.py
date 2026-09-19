from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"

TRAIN_PATH = (
    RESULTS / "ksl_train_embedding_bank.npz"
)

OOD_PATH = (
    RESULTS / "ksl_controlled_ood.npz"
)


EPS = 1e-12
K_NEIGHBORS = 5
PCA_COMPONENTS = 6
MAHALANOBIS_REG = 1e-4


def load_npz(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Missing experiment file: {path}"
        )

    data = np.load(
        path,
        allow_pickle=False,
    )

    return {
        key: data[key]
        for key in data.files
    }


def softmax(logits):
    shifted = (
        logits
        - logits.max(
            axis=1,
            keepdims=True,
        )
    )

    exp = np.exp(shifted)

    return (
        exp
        / exp.sum(
            axis=1,
            keepdims=True,
        )
    )


def msp_score(logits):
    probabilities = softmax(
        logits
    )

    # Larger = more OOD-like.
    return (
        1.0
        - probabilities.max(axis=1)
    )


def energy_score(logits):
    # Standard energy:
    #
    # E(x) = -log sum exp(f_k(x))
    #
    # Larger / less negative generally means
    # more OOD-like.
    maximum = logits.max(
        axis=1,
        keepdims=True,
    )

    logsumexp = (
        maximum[:, 0]
        + np.log(
            np.exp(
                logits - maximum
            ).sum(axis=1)
        )
    )

    return -logsumexp


def fit_centroids(
    embeddings,
    labels,
):
    classes = np.asarray(
        sorted(
            np.unique(labels).tolist()
        )
    )

    centroids = np.stack(
        [
            embeddings[
                labels == label
            ].mean(axis=0)
            for label in classes
        ]
    )

    return classes, centroids


def centroid_l2_score(
    embeddings,
    centroids,
):
    distances = np.linalg.norm(
        embeddings[:, None, :]
        - centroids[None, :, :],
        axis=2,
    )

    return distances.min(
        axis=1
    )


def l2_normalize(x):
    norms = np.linalg.norm(
        x,
        axis=1,
        keepdims=True,
    )

    return (
        x
        / np.maximum(
            norms,
            EPS,
        )
    )


def centroid_cosine_score(
    embeddings,
    centroids,
):
    x = l2_normalize(
        embeddings
    )

    c = l2_normalize(
        centroids
    )

    similarities = (
        x @ c.T
    )

    # High similarity = ID-like,
    # therefore 1 - similarity = OOD score.
    return (
        1.0
        - similarities.max(axis=1)
    )


def knn_score(
    reference,
    query,
    k=5,
):
    if k < 1:
        raise ValueError(
            "k must be at least 1."
        )

    if k > len(reference):
        raise ValueError(
            "k cannot exceed reference size."
        )

    scores = []

    for x in query:
        distances = np.linalg.norm(
            reference
            - x[None, :],
            axis=1,
        )

        nearest = np.partition(
            distances,
            k - 1,
        )[:k]

        # Mean distance to k nearest
        # training examples.
        scores.append(
            nearest.mean()
        )

    return np.asarray(
        scores,
        dtype=np.float64,
    )


def fit_pca(
    embeddings,
    n_components,
):
    mean = embeddings.mean(
        axis=0,
        keepdims=True,
    )

    centered = (
        embeddings - mean
    )

    _, _, vt = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    components = (
        vt[:n_components]
    )

    return mean, components


def project_pca(
    embeddings,
    mean,
    components,
):
    return (
        (embeddings - mean)
        @ components.T
    )


def fit_class_mahalanobis(
    embeddings,
    labels,
    regularization=1e-4,
):
    classes = np.asarray(
        sorted(
            np.unique(labels).tolist()
        )
    )

    means = []

    residuals = []

    for label in classes:
        class_x = embeddings[
            labels == label
        ]

        class_mean = class_x.mean(
            axis=0
        )

        means.append(
            class_mean
        )

        residuals.append(
            class_x
            - class_mean[None, :]
        )

    means = np.stack(
        means
    )

    residuals = np.concatenate(
        residuals,
        axis=0,
    )

    covariance = np.cov(
        residuals,
        rowvar=False,
    )

    covariance = (
        covariance
        + regularization
        * np.eye(
            covariance.shape[0]
        )
    )

    inverse_covariance = (
        np.linalg.inv(
            covariance
        )
    )

    return (
        classes,
        means,
        inverse_covariance,
    )


def mahalanobis_score(
    embeddings,
    class_means,
    inverse_covariance,
):
    all_distances = []

    for mean in class_means:
        delta = (
            embeddings
            - mean[None, :]
        )

        squared = np.einsum(
            "ni,ij,nj->n",
            delta,
            inverse_covariance,
            delta,
        )

        squared = np.maximum(
            squared,
            0.0,
        )

        all_distances.append(
            np.sqrt(squared)
        )

    matrix = np.stack(
        all_distances,
        axis=1,
    )

    return matrix.min(
        axis=1
    )


def roc_curve_manual(
    id_scores,
    ood_scores,
):
    # OOD = positive class.
    y = np.concatenate(
        [
            np.zeros(
                len(id_scores),
                dtype=np.int64,
            ),
            np.ones(
                len(ood_scores),
                dtype=np.int64,
            ),
        ]
    )

    scores = np.concatenate(
        [
            id_scores,
            ood_scores,
        ]
    )

    order = np.argsort(
        scores
    )[::-1]

    y = y[order]
    scores = scores[order]

    positives = int(
        y.sum()
    )

    negatives = int(
        len(y) - positives
    )

    tp = 0
    fp = 0

    tpr = [0.0]
    fpr = [0.0]

    previous_score = None

    for label, score in zip(
        y,
        scores,
    ):
        if (
            previous_score is not None
            and score != previous_score
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

        previous_score = score

    tpr.append(1.0)
    fpr.append(1.0)

    return (
        np.asarray(fpr),
        np.asarray(tpr),
    )


def auroc(
    id_scores,
    ood_scores,
):
    fpr, tpr = roc_curve_manual(
        id_scores,
        ood_scores,
    )

    return float(
        np.trapezoid(
            tpr,
            fpr,
        )
    )


def aupr_ood(
    id_scores,
    ood_scores,
):
    y = np.concatenate(
        [
            np.zeros(
                len(id_scores),
                dtype=np.int64,
            ),
            np.ones(
                len(ood_scores),
                dtype=np.int64,
            ),
        ]
    )

    scores = np.concatenate(
        [
            id_scores,
            ood_scores,
        ]
    )

    order = np.argsort(
        scores
    )[::-1]

    y = y[order]

    tp = np.cumsum(
        y == 1
    )

    fp = np.cumsum(
        y == 0
    )

    total_positive = int(
        np.sum(y == 1)
    )

    recall = (
        tp / total_positive
    )

    precision = (
        tp
        / np.maximum(
            tp + fp,
            1,
        )
    )

    recall = np.concatenate(
        [[0.0], recall]
    )

    precision = np.concatenate(
        [[1.0], precision]
    )

    return float(
        np.sum(
            (
                recall[1:]
                - recall[:-1]
            )
            * precision[1:]
        )
    )


def fpr_at_95_tpr(
    id_scores,
    ood_scores,
):
    fpr, tpr = roc_curve_manual(
        id_scores,
        ood_scores,
    )

    valid = np.where(
        tpr >= 0.95
    )[0]

    if len(valid) == 0:
        return 1.0

    return float(
        fpr[valid].min()
    )


def evaluate(
    id_scores,
    ood_scores,
):
    return {
        "auroc": auroc(
            id_scores,
            ood_scores,
        ),
        "aupr": aupr_ood(
            id_scores,
            ood_scores,
        ),
        "fpr95": fpr_at_95_tpr(
            id_scores,
            ood_scores,
        ),
        "id_mean": float(
            np.mean(id_scores)
        ),
        "ood_mean": float(
            np.mean(ood_scores)
        ),
    }


def main():
    print("=" * 78)
    print(
        "LughaLab KSL OOD Detector Evaluation"
    )
    print("=" * 78)

    train = load_npz(
        TRAIN_PATH
    )

    benchmark = load_npz(
        OOD_PATH
    )

    train_x = train[
        "embeddings"
    ].astype(np.float64)

    train_y = train[
        "labels"
    ]

    id_x = benchmark[
        "id_embeddings"
    ].astype(np.float64)

    id_logits = benchmark[
        "id_logits"
    ].astype(np.float64)

    ood_x = benchmark[
        "ood_embeddings"
    ].astype(np.float64)

    ood_logits = benchmark[
        "ood_logits"
    ].astype(np.float64)

    ood_types = benchmark[
        "ood_types"
    ]

    print()
    print("DATA")
    print("-" * 78)

    print(
        "Training reference:",
        train_x.shape,
    )

    print(
        "Held-out ID:",
        id_x.shape,
    )

    print(
        "Controlled OOD:",
        ood_x.shape,
    )

    print(
        "OOD families:",
        sorted(
            np.unique(
                ood_types
            ).tolist()
        ),
    )

    classes, centroids = (
        fit_centroids(
            train_x,
            train_y,
        )
    )

    pca_mean, pca_components = (
        fit_pca(
            train_x,
            PCA_COMPONENTS,
        )
    )

    train_pca = project_pca(
        train_x,
        pca_mean,
        pca_components,
    )

    id_pca = project_pca(
        id_x,
        pca_mean,
        pca_components,
    )

    ood_pca = project_pca(
        ood_x,
        pca_mean,
        pca_components,
    )

    (
        _,
        mahalanobis_means,
        inverse_covariance,
    ) = fit_class_mahalanobis(
        train_pca,
        train_y,
        regularization=(
            MAHALANOBIS_REG
        ),
    )

    detectors = {
        "MSP": (
            msp_score(
                id_logits
            ),
            msp_score(
                ood_logits
            ),
        ),

        "Energy": (
            energy_score(
                id_logits
            ),
            energy_score(
                ood_logits
            ),
        ),

        "Centroid-L2": (
            centroid_l2_score(
                id_x,
                centroids,
            ),
            centroid_l2_score(
                ood_x,
                centroids,
            ),
        ),

        "Centroid-Cosine": (
            centroid_cosine_score(
                id_x,
                centroids,
            ),
            centroid_cosine_score(
                ood_x,
                centroids,
            ),
        ),

        f"{K_NEIGHBORS}-NN": (
            knn_score(
                train_x,
                id_x,
                K_NEIGHBORS,
            ),
            knn_score(
                train_x,
                ood_x,
                K_NEIGHBORS,
            ),
        ),

        f"PCA{PCA_COMPONENTS}-Mahalanobis": (
            mahalanobis_score(
                id_pca,
                mahalanobis_means,
                inverse_covariance,
            ),
            mahalanobis_score(
                ood_pca,
                mahalanobis_means,
                inverse_covariance,
            ),
        ),
    }

    print()
    print("OVERALL OOD DETECTION")
    print("-" * 78)

    print(
        f"{'Detector':<22}"
        f"{'AUROC':>9}"
        f"{'AUPR':>9}"
        f"{'FPR95':>9}"
        f"{'ID mean':>12}"
        f"{'OOD mean':>12}"
    )

    print("-" * 73)

    overall_results = {}

    for name, (
        id_scores,
        ood_scores,
    ) in detectors.items():
        metrics = evaluate(
            id_scores,
            ood_scores,
        )

        overall_results[
            name
        ] = metrics

        print(
            f"{name:<22}"
            f"{metrics['auroc']:>9.4f}"
            f"{metrics['aupr']:>9.4f}"
            f"{metrics['fpr95']:>9.4f}"
            f"{metrics['id_mean']:>12.4f}"
            f"{metrics['ood_mean']:>12.4f}"
        )

    families = sorted(
        np.unique(
            ood_types
        ).tolist()
    )

    print()
    print("AUROC BY OOD FAMILY")
    print("-" * 78)

    print(
        f"{'Detector':<22}"
        + "".join(
            f"{family[:12]:>14}"
            for family in families
        )
    )

    family_results = {}

    for name, (
        id_scores,
        ood_scores,
    ) in detectors.items():
        values = []

        family_results[name] = {}

        for family in families:
            mask = (
                ood_types
                == family
            )

            family_auroc = auroc(
                id_scores,
                ood_scores[mask],
            )

            family_results[
                name
            ][
                family
            ] = family_auroc

            values.append(
                family_auroc
            )

        print(
            f"{name:<22}"
            + "".join(
                f"{value:>14.4f}"
                for value in values
            )
        )

    output = (
        RESULTS
        / "ksl_ood_detector_evaluation.npz"
    )

    save_data = {
        "ood_types": ood_types,
        "classes": classes,
    }

    for name, (
        id_scores,
        ood_scores,
    ) in detectors.items():
        safe_name = (
            name
            .lower()
            .replace("-", "_")
        )

        save_data[
            f"{safe_name}_id"
        ] = id_scores

        save_data[
            f"{safe_name}_ood"
        ] = ood_scores

    np.savez_compressed(
        output,
        **save_data,
    )

    print()
    print(
        "Saved:",
        output,
    )


if __name__ == "__main__":
    main()