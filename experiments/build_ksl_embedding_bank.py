from pathlib import Path
import sys

import kagglehub
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
RESULTS = ROOT / "experiments" / "results"

sys.path.insert(
    0,
    str(BACKEND),
)


from app.services.ksl_classifier import (
    load_ksl_classifier,
)


DATASET_ID = "joanwachuka/ksl-hand-landmarks"

HAND_DIMENSION = 126
CHECKPOINT_DIMENSION = 225
TARGET_LENGTH = 64


def pad_features(
    sequence: np.ndarray,
) -> np.ndarray:
    sequence = np.asarray(
        sequence,
        dtype=np.float32,
    )

    if (
        sequence.ndim != 2
        or sequence.shape[1] != HAND_DIMENSION
    ):
        raise ValueError(
            "Expected raw KSL sample with shape "
            f"(T, {HAND_DIMENSION}), got "
            f"{sequence.shape}."
        )

    output = np.zeros(
        (
            sequence.shape[0],
            CHECKPOINT_DIMENSION,
        ),
        dtype=np.float32,
    )

    output[
        :,
        :HAND_DIMENSION,
    ] = sequence

    return output


def temporal_prepare(
    sequence: np.ndarray,
) -> np.ndarray:
    """
    Reconstructed temporal strategy:
      T < 64: repeat final frame
      T = 64: unchanged
      T > 64: uniform nearest-frame downsampling
    """
    source_length = sequence.shape[0]

    if source_length == 0:
        raise ValueError(
            "Empty KSL sequence."
        )

    if source_length == TARGET_LENGTH:
        return sequence.copy()

    if source_length < TARGET_LENGTH:
        missing = (
            TARGET_LENGTH
            - source_length
        )

        repeated = np.repeat(
            sequence[-1:],
            missing,
            axis=0,
        )

        return np.concatenate(
            [
                sequence,
                repeated,
            ],
            axis=0,
        ).astype(np.float32)

    indices = np.linspace(
        0,
        source_length - 1,
        TARGET_LENGTH,
    )

    indices = np.rint(
        indices
    ).astype(np.int64)

    return sequence[
        indices
    ].astype(np.float32)


def standardize(
    sequence: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    mean = np.asarray(
        mean,
        dtype=np.float32,
    )

    std = np.asarray(
        std,
        dtype=np.float32,
    )

    safe_std = np.where(
        np.abs(std) < 1e-8,
        1.0,
        std,
    )

    return (
        (
            sequence - mean
        )
        / safe_std
    ).astype(np.float32)


def prepare_sample(
    sequence: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    padded = pad_features(
        sequence
    )

    temporal = temporal_prepare(
        padded
    )

    return standardize(
        temporal,
        mean,
        std,
    )


def locate_dataset() -> Path:
    root = Path(
        kagglehub.dataset_download(
            DATASET_ID
        )
    )

    candidates = [
        root
        / "data_split"
        / "data_split",

        root
        / "data_split",
    ]

    for candidate in candidates:
        if (
            (candidate / "train").exists()
            and (candidate / "test").exists()
        ):
            return candidate

    raise FileNotFoundError(
        "Could not locate train/test "
        f"directories under {root}."
    )


def process_split(
    split_directory: Path,
    bundle,
):
    model = bundle["model"]
    device = bundle["device"]
    mean = bundle["mean"]
    std = bundle["std"]

    embeddings = []
    labels = []
    predictions = []
    probabilities = []
    paths = []

    class_directories = sorted(
        [
            path
            for path
            in split_directory.iterdir()
            if path.is_dir()
        ]
    )

    for class_directory in class_directories:
        label = class_directory.name

        files = sorted(
            class_directory.glob(
                "*.npy"
            )
        )

        print(
            f"  {label:<8}: "
            f"{len(files)}"
        )

        for path in files:
            raw = np.load(
                path
            )

            prepared = prepare_sample(
                raw,
                mean,
                std,
            )

            tensor = (
                torch.from_numpy(
                    prepared
                )
                .unsqueeze(0)
                .to(device)
            )

            with torch.inference_mode():
                embedding = model.encode(
                    tensor
                )

                logits = model.head(
                    embedding
                )

                probs = torch.softmax(
                    logits,
                    dim=-1,
                )

            embedding_np = (
                embedding
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            probs_np = (
                probs
                .squeeze(0)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            predicted_index = int(
                np.argmax(
                    probs_np
                )
            )

            predicted_label = (
                bundle[
                    "index_to_label"
                ][
                    predicted_index
                ]
            )

            embeddings.append(
                embedding_np
            )

            labels.append(
                label
            )

            predictions.append(
                predicted_label
            )

            probabilities.append(
                probs_np
            )

            paths.append(
                str(path)
            )

    return {
        "embeddings": np.stack(
            embeddings
        ),
        "labels": np.asarray(
            labels
        ),
        "predictions": np.asarray(
            predictions
        ),
        "probabilities": np.stack(
            probabilities
        ),
        "paths": np.asarray(
            paths
        ),
    }


def report_split(
    name: str,
    result,
):
    labels = result["labels"]
    predictions = result[
        "predictions"
    ]

    correct = (
        labels == predictions
    )

    accuracy = float(
        np.mean(correct)
    )

    max_probabilities = (
        result["probabilities"]
        .max(axis=1)
    )

    embeddings = result[
        "embeddings"
    ]

    print()
    print(name.upper())
    print("-" * 78)

    print(
        "Samples:",
        len(labels),
    )

    print(
        "Embedding shape:",
        embeddings.shape,
    )

    print(
        "Accuracy:",
        f"{correct.sum()}/{len(correct)} "
        f"= {accuracy:.4f}",
    )

    print(
        "Mean max probability:",
        f"{max_probabilities.mean():.4f}",
    )

    embedding_norms = np.linalg.norm(
        embeddings,
        axis=1,
    )

    mean_embedding_norm = float(
        embedding_norms.mean()
    )

    print(
        "Embedding norm mean:",
        f"{mean_embedding_norm:.4f}",
    )


def save_split(
    name: str,
    result,
):
    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        RESULTS
        / f"ksl_{name}_embedding_bank.npz"
    )

    np.savez_compressed(
        output,
        **result,
    )

    print(
        "Saved:",
        output,
    )


def main():
    print("=" * 78)
    print(
        "LughaLab KSL Embedding Bank"
    )
    print("=" * 78)

    dataset_root = (
        locate_dataset()
    )

    print()
    print(
        "Dataset:",
        dataset_root,
    )

    print()
    print(
        "Loading checkpoint..."
    )

    bundle = (
        load_ksl_classifier()
    )

    print(
        "Device:",
        bundle["device"],
    )

    print()
    print(
        "TRAIN SPLIT"
    )
    print("-" * 78)

    train = process_split(
        dataset_root / "train",
        bundle,
    )

    report_split(
        "train",
        train,
    )

    save_split(
        "train",
        train,
    )

    print()
    print(
        "TEST SPLIT"
    )
    print("-" * 78)

    test = process_split(
        dataset_root / "test",
        bundle,
    )

    report_split(
        "test",
        test,
    )

    save_split(
        "test",
        test,
    )


if __name__ == "__main__":
    main()