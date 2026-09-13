from functools import lru_cache
from typing import Any, Dict

import numpy as np
import torch
from huggingface_hub import (
    hf_hub_download,
)

from app.models.ksl_transformer import (
    LandmarkTransformer,
)


REPO_ID = (
    "luciayen/"
    "afrisign-exp1-ksl-baseline"
)

CHECKPOINT_FILE = (
    "pytorch_model.bin"
)


def _select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device(
            "cuda"
        )

    if (
        hasattr(
            torch.backends,
            "mps",
        )
        and torch.backends.mps.is_available()
    ):
        return torch.device(
            "mps"
        )

    return torch.device(
        "cpu"
    )


def _to_numpy(
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
            .astype(
                np.float32
            )
        )

    return np.asarray(
        value,
        dtype=np.float32,
    )


@lru_cache(
    maxsize=1
)
def load_ksl_classifier():
    checkpoint_path = (
        hf_hub_download(
            repo_id=REPO_ID,
            filename=CHECKPOINT_FILE,
        )
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint[
        "model"
    ]

    label_to_index = checkpoint[
        "l2i"
    ]

    index_to_label = {
        int(index): label
        for label, index
        in label_to_index.items()
    }

    mean = _to_numpy(
        checkpoint[
            "mean"
        ]
    )

    std = _to_numpy(
        checkpoint[
            "std"
        ]
    )

    model = LandmarkTransformer(
        input_dim=225,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        num_classes=len(
            label_to_index
        ),
        max_frames=64,
        dropout=0.1,
    )

    # Deliberately strict.
    #
    # If this fails, our reconstruction
    # is not checkpoint-compatible and
    # inference must not proceed.
    model.load_state_dict(
        state_dict,
        strict=True,
    )

    device = (
        _select_device()
    )

    model = model.to(
        device
    )

    model.eval()

    return {
        "model": model,
        "device": device,
        "mean": mean,
        "std": std,
        "label_to_index": (
            label_to_index
        ),
        "index_to_label": (
            index_to_label
        ),
        "checkpoint": (
            checkpoint
        ),
        "checkpoint_path": (
            checkpoint_path
        ),
    }


def classify_checkpoint_sequence(
    sequence: np.ndarray,
) -> Dict[str, Any]:
    bundle = (
        load_ksl_classifier()
    )

    sequence = np.asarray(
        sequence,
        dtype=np.float32,
    )

    if sequence.shape != (
        64,
        225,
    ):
        raise ValueError(
            "Expected checkpoint-ready "
            "sequence with shape "
            f"(64, 225), got "
            f"{sequence.shape}."
        )

    if not np.isfinite(
        sequence
    ).all():
        raise ValueError(
            "Sequence contains NaN "
            "or infinite values."
        )

    tensor = (
        torch.from_numpy(
            sequence
        )
        .unsqueeze(0)
        .to(
            bundle[
                "device"
            ]
        )
    )

    with torch.inference_mode():
        logits = (
            bundle[
                "model"
            ](
                tensor
            )
        )

        probabilities = (
            torch.softmax(
                logits,
                dim=-1,
            )
            .squeeze(0)
            .detach()
            .cpu()
            .numpy()
        )

    ranking = np.argsort(
        probabilities
    )[::-1]

    predictions = []

    for index in ranking:
        predictions.append(
            {
                "index": int(
                    index
                ),
                "label": (
                    bundle[
                        "index_to_label"
                    ][
                        int(index)
                    ]
                ),
                "probability": float(
                    probabilities[
                        index
                    ]
                ),
            }
        )

    return {
        "predicted_label": (
            predictions[0][
                "label"
            ]
        ),
        "predicted_probability": (
            predictions[0][
                "probability"
            ]
        ),
        "predictions": (
            predictions
        ),
        "device": str(
            bundle[
                "device"
            ]
        ),
        "model": REPO_ID,
        "closed_set": True,
        "supported_labels": [
            bundle[
                "index_to_label"
            ][i]
            for i in sorted(
                bundle[
                    "index_to_label"
                ]
            )
        ],
        "warning": (
            "This is a closed-set "
            "four-class research baseline. "
            "The model must assign every "
            "input to father, hello, is, "
            "or my. A high softmax "
            "probability is not evidence "
            "that an arbitrary input "
            "actually represents that KSL "
            "gloss."
        ),
    }