import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    ROOT
    / "experiments"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "ksl_checkpoint_audit.json"
)

REPO_ID = (
    "luciayen/"
    "afrisign-exp1-ksl-baseline"
)

CHECKPOINT_FILE = (
    "pytorch_model.bin"
)


def to_numpy(
    value,
) -> np.ndarray:
    if isinstance(
        value,
        torch.Tensor,
    ):
        return (
            value
            .detach()
            .cpu()
            .numpy()
        )

    if isinstance(
        value,
        np.ndarray,
    ):
        return value

    return np.asarray(
        value
    )


def array_summary(
    value,
):
    array = (
        to_numpy(
            value
        )
        .astype(
            np.float32
        )
    )

    return {
        "shape": list(
            array.shape
        ),
        "dtype": str(
            array.dtype
        ),
        "min": float(
            np.min(
                array
            )
        ),
        "max": float(
            np.max(
                array
            )
        ),
        "mean": float(
            np.mean(
                array
            )
        ),
        "std": float(
            np.std(
                array
            )
        ),
        "finite": bool(
            np.isfinite(
                array
            ).all()
        ),
    }


def main():
    print()
    print("=" * 78)
    print(
        "LughaLab KSL "
        "Checkpoint Audit"
    )
    print("=" * 78)
    print()

    checkpoint_path = (
        hf_hub_download(
            repo_id=REPO_ID,
            filename=CHECKPOINT_FILE,
        )
    )

    print(
        "Downloaded checkpoint:"
    )
    print(
        checkpoint_path
    )
    print()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise TypeError(
            "Expected checkpoint "
            "to be a dictionary."
        )

    print(
        "Checkpoint keys:"
    )

    for key in checkpoint:
        print(
            f"  - {key}"
        )

    print()

    label_to_index = (
        checkpoint.get(
            "l2i",
            {},
        )
    )

    index_to_label = {
        int(index): label
        for label, index
        in label_to_index.items()
    }

    mean = checkpoint.get(
        "mean"
    )

    std = checkpoint.get(
        "std"
    )

    state_dict = checkpoint.get(
        "model",
        {}
    )

    if mean is None:
        raise RuntimeError(
            "Checkpoint does not "
            "contain mean."
        )

    if std is None:
        raise RuntimeError(
            "Checkpoint does not "
            "contain std."
        )

    mean_np = (
        to_numpy(
            mean
        )
        .astype(
            np.float32
        )
    )

    std_np = (
        to_numpy(
            std
        )
        .astype(
            np.float32
        )
    )

    print(
        "Labels:"
    )

    for index in sorted(
        index_to_label
    ):
        print(
            f"  {index}: "
            f"{index_to_label[index]}"
        )

    print()
    print(
        "Mean shape:",
        tuple(
            mean_np.shape
        ),
    )

    print(
        "Std shape:",
        tuple(
            std_np.shape
        ),
    )

    print()

    state_shapes = {}

    print(
        "Model state_dict:"
    )

    for name, tensor in (
        state_dict.items()
    ):
        state_shapes[name] = (
            list(
                tensor.shape
            )
            if hasattr(
                tensor,
                "shape",
            )
            else None
        )

        print(
            f"  {name:<55} "
            f"{state_shapes[name]}"
        )

    zero_mean_features = (
        np.isclose(
            mean_np,
            0.0,
            atol=1e-8,
        )
    )

    zero_std_features = (
        np.isclose(
            std_np,
            0.0,
            atol=1e-8,
        )
    )

    payload = {
        "repository": (
            REPO_ID
        ),
        "checkpoint_file": (
            CHECKPOINT_FILE
        ),
        "checkpoint_path": (
            str(
                checkpoint_path
            )
        ),
        "epoch": (
            checkpoint.get(
                "epoch"
            )
        ),
        "validation_accuracy": (
            checkpoint.get(
                "val_acc"
            )
        ),
        "label_to_index": (
            label_to_index
        ),
        "index_to_label": (
            index_to_label
        ),
        "number_of_classes": (
            len(
                label_to_index
            )
        ),
        "mean": (
            array_summary(
                mean_np
            )
        ),
        "std": (
            array_summary(
                std_np
            )
        ),
        "zero_mean_feature_count": int(
            zero_mean_features.sum()
        ),
        "zero_std_feature_count": int(
            zero_std_features.sum()
        ),
        "model_state_shapes": (
            state_shapes
        ),
        "architecture_inference": {
            "input_dimension": 225,
            "embedding_dimension": 256,
            "sequence_positions": 65,
            "transformer_layers": 4,
            "feedforward_dimension": 1024,
            "number_of_classes": 4,
            "uses_cls_token": True,
        },
        "compatibility_note": (
            "The checkpoint was trained "
            "on hands-only KSL landmarks. "
            "Its first 126 features contain "
            "42 hand landmarks x 3 coordinates, "
            "while features 126:225 are zero "
            "padding. This differs from the "
            "native LughaLab ordering of "
            "pose + left hand + right hand."
        ),
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(
        "Zero-mean features:",
        int(
            zero_mean_features.sum()
        ),
    )

    print(
        "Zero-std features:",
        int(
            zero_std_features.sum()
        ),
    )

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )
    print()


if __name__ == "__main__":
    main()