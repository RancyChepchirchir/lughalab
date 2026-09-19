from pathlib import Path
import sys

import kagglehub
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
RESULTS = ROOT / "experiments" / "results"

sys.path.insert(0, str(BACKEND))


from app.services.ksl_classifier import (
    load_ksl_classifier,
)


DATASET_ID = "joanwachuka/ksl-hand-landmarks"

HAND_DIM = 126
CHECKPOINT_DIM = 225
TARGET_LENGTH = 64

RNG = np.random.default_rng(42)


def locate_test_directory():
    root = Path(
        kagglehub.dataset_download(
            DATASET_ID
        )
    )

    candidates = [
        root / "data_split" / "data_split" / "test",
        root / "data_split" / "test",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not locate test split under {root}."
    )


def pad_features(x):
    x = np.asarray(
        x,
        dtype=np.float32,
    )

    output = np.zeros(
        (len(x), CHECKPOINT_DIM),
        dtype=np.float32,
    )

    output[:, :HAND_DIM] = x

    return output


def temporal_prepare(x):
    length = len(x)

    if length == 0:
        raise ValueError("Empty sequence.")

    if length == TARGET_LENGTH:
        return x.copy()

    if length < TARGET_LENGTH:
        repeated = np.repeat(
            x[-1:],
            TARGET_LENGTH - length,
            axis=0,
        )

        return np.concatenate(
            [x, repeated],
            axis=0,
        )

    indices = np.rint(
        np.linspace(
            0,
            length - 1,
            TARGET_LENGTH,
        )
    ).astype(np.int64)

    return x[indices]


def prepare(x, mean, std):
    x = pad_features(x)
    x = temporal_prepare(x)

    safe_std = np.where(
        np.abs(std) < 1e-8,
        1.0,
        std,
    )

    return (
        (x - mean) / safe_std
    ).astype(np.float32)


def corrupt(
    raw,
    kind,
):
    x = np.asarray(
        raw,
        dtype=np.float32,
    ).copy()

    if kind == "frame_shuffle":
        indices = RNG.permutation(
            len(x)
        )
        return x[indices]

    if kind == "frame_reverse":
        return x[::-1].copy()

    if kind == "single_frame":
        middle = x[
            len(x) // 2
        ].copy()

        return np.repeat(
            middle[None, :],
            len(x),
            axis=0,
        )

    if kind == "hand_dropout":
        # Randomly remove either the first
        # or second 63D hand representation.
        if RNG.random() < 0.5:
            x[:, :63] = 0.0
        else:
            x[:, 63:126] = 0.0

        return x

    if kind == "gaussian_noise":
        # Scale noise using each feature's
        # empirical variation within sequence.
        feature_std = x.std(
            axis=0,
            keepdims=True,
        )

        scale = np.maximum(
            feature_std,
            1e-3,
        )

        noise = RNG.normal(
            0.0,
            0.5,
            size=x.shape,
        ).astype(np.float32)

        return (
            x + noise * scale
        ).astype(np.float32)

    raise ValueError(
        f"Unknown corruption: {kind}"
    )


def infer(
    prepared,
    bundle,
):
    tensor = (
        torch.from_numpy(
            prepared
        )
        .unsqueeze(0)
        .to(bundle["device"])
    )

    with torch.inference_mode():
        embedding = bundle[
            "model"
        ].encode(tensor)

        logits = bundle[
            "model"
        ].head(embedding)

        probabilities = torch.softmax(
            logits,
            dim=-1,
        )

    return (
        embedding
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float32),

        logits
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float32),

        probabilities
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float32),
    )


def main():
    print("=" * 78)
    print(
        "LughaLab KSL Controlled OOD Benchmark"
    )
    print("=" * 78)

    test_dir = (
        locate_test_directory()
    )

    bundle = (
        load_ksl_classifier()
    )

    mean = np.asarray(
        bundle["mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        bundle["std"],
        dtype=np.float32,
    )

    corruption_types = [
        "frame_shuffle",
        "frame_reverse",
        "single_frame",
        "hand_dropout",
        "gaussian_noise",
    ]

    id_embeddings = []
    id_logits = []
    id_probabilities = []
    id_labels = []
    id_paths = []

    ood_embeddings = []
    ood_logits = []
    ood_probabilities = []
    ood_labels = []
    ood_types = []
    ood_paths = []

    files = []

    for class_dir in sorted(
        test_dir.iterdir()
    ):
        if not class_dir.is_dir():
            continue

        for path in sorted(
            class_dir.glob("*.npy")
        ):
            files.append(
                (
                    class_dir.name,
                    path,
                )
            )

    print()
    print(
        "ID test samples:",
        len(files),
    )

    print(
        "OOD variants per sample:",
        len(corruption_types),
    )

    print(
        "Expected OOD samples:",
        len(files)
        * len(corruption_types),
    )

    for index, (
        label,
        path,
    ) in enumerate(
        files,
        start=1,
    ):
        raw = np.load(path)

        prepared = prepare(
            raw,
            mean,
            std,
        )

        embedding, logits, probs = (
            infer(
                prepared,
                bundle,
            )
        )

        id_embeddings.append(
            embedding
        )
        id_logits.append(
            logits
        )
        id_probabilities.append(
            probs
        )
        id_labels.append(
            label
        )
        id_paths.append(
            str(path)
        )

        for kind in corruption_types:
            corrupted = corrupt(
                raw,
                kind,
            )

            prepared_ood = prepare(
                corrupted,
                mean,
                std,
            )

            (
                ood_embedding,
                ood_logit,
                ood_probability,
            ) = infer(
                prepared_ood,
                bundle,
            )

            ood_embeddings.append(
                ood_embedding
            )

            ood_logits.append(
                ood_logit
            )

            ood_probabilities.append(
                ood_probability
            )

            ood_labels.append(
                label
            )

            ood_types.append(
                kind
            )

            ood_paths.append(
                str(path)
            )

        if (
            index % 20 == 0
            or index == len(files)
        ):
            print(
                f"Processed "
                f"{index}/{len(files)}"
            )

    output = (
        RESULTS
        / "ksl_controlled_ood.npz"
    )

    RESULTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output,

        id_embeddings=np.stack(
            id_embeddings
        ),

        id_logits=np.stack(
            id_logits
        ),

        id_probabilities=np.stack(
            id_probabilities
        ),

        id_labels=np.asarray(
            id_labels
        ),

        id_paths=np.asarray(
            id_paths
        ),

        ood_embeddings=np.stack(
            ood_embeddings
        ),

        ood_logits=np.stack(
            ood_logits
        ),

        ood_probabilities=np.stack(
            ood_probabilities
        ),

        ood_labels=np.asarray(
            ood_labels
        ),

        ood_types=np.asarray(
            ood_types
        ),

        ood_paths=np.asarray(
            ood_paths
        ),
    )

    print()
    print("RESULT")
    print("-" * 78)

    print(
        "ID embeddings:",
        np.stack(
            id_embeddings
        ).shape,
    )

    print(
        "OOD embeddings:",
        np.stack(
            ood_embeddings
        ).shape,
    )

    print(
        "OOD types:",
        sorted(
            set(ood_types)
        ),
    )

    print()
    print(
        "Saved:",
        output,
    )


if __name__ == "__main__":
    main()