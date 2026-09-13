import json
import sys
from pathlib import Path

import kagglehub
import numpy as np
import torch
from huggingface_hub import hf_hub_download


ROOT = Path(
    __file__
).resolve().parents[1]

BACKEND = (
    ROOT
    / "backend"
)

sys.path.insert(
    0,
    str(BACKEND),
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

KAGGLE_DATASET = (
    "joanwachuka/"
    "ksl-hand-landmarks"
)

LABELS = [
    "father",
    "hello",
    "is",
    "my",
]

TARGET_LENGTH = 64
HAND_DIM = 126
MODEL_DIM = 225

RESULTS_DIR = (
    ROOT
    / "experiments"
    / "results"
)

OUTPUT_PATH = (
    RESULTS_DIR
    / "ksl_forward_architecture_audit.json"
)


CONFIGURATIONS = [
    {
        "activation": "relu",
        "norm_first": False,
    },
    {
        "activation": "gelu",
        "norm_first": False,
    },
    {
        "activation": "relu",
        "norm_first": True,
    },
    {
        "activation": "gelu",
        "norm_first": True,
    },
]


def linear_interpolation(
    sequence: np.ndarray,
) -> np.ndarray:

    frames = sequence.shape[0]

    old_positions = np.linspace(
        0.0,
        1.0,
        frames,
        dtype=np.float32,
    )

    new_positions = np.linspace(
        0.0,
        1.0,
        TARGET_LENGTH,
        dtype=np.float32,
    )

    result = np.empty(
        (
            TARGET_LENGTH,
            HAND_DIM,
        ),
        dtype=np.float32,
    )

    for feature in range(
        HAND_DIM
    ):
        result[
            :,
            feature
        ] = np.interp(
            new_positions,
            old_positions,
            sequence[
                :,
                feature
            ],
        )

    return result


def pad_features(
    sequence: np.ndarray,
) -> np.ndarray:

    result = np.zeros(
        (
            TARGET_LENGTH,
            MODEL_DIM,
        ),
        dtype=np.float32,
    )

    result[
        :,
        :HAND_DIM
    ] = sequence

    return result


def standardize(
    sequence,
    mean,
    std,
):

    mean = np.asarray(
        mean,
        dtype=np.float32,
    )

    std = np.asarray(
        std,
        dtype=np.float32,
    )

    safe_std = np.where(
        np.abs(
            std
        ) < 1e-8,
        1.0,
        std,
    )

    return (
        (
            sequence
            - mean
        )
        / safe_std
    ).astype(
        np.float32
    )


def load_samples():
    root = Path(
        kagglehub.dataset_download(
            KAGGLE_DATASET
        )
    )

    test_dir = (
        root
        / "data_split"
        / "data_split"
        / "test"
    )

    samples = []

    for label in LABELS:
        for path in sorted(
            (
                test_dir
                / label
            ).glob(
                "*.npy"
            )
        ):
            array = np.load(
                path,
                allow_pickle=False,
            ).astype(
                np.float32
            )

            samples.append(
                {
                    "label": label,
                    "path": path,
                    "sequence": array,
                }
            )

    return samples


def load_checkpoint():
    checkpoint_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=CHECKPOINT_FILE,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    return checkpoint


def evaluate_configuration(
    configuration,
    checkpoint,
    samples,
):
    model = LandmarkTransformer(
        input_dim=225,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        num_classes=4,
        max_frames=64,
        dropout=0.1,
        activation=configuration[
            "activation"
        ],
        norm_first=configuration[
            "norm_first"
        ],
    )

    model.load_state_dict(
        checkpoint[
            "model"
        ],
        strict=True,
    )

    model.eval()

    mean = checkpoint[
        "mean"
    ]

    std = checkpoint[
        "std"
    ]

    l2i = checkpoint[
        "l2i"
    ]

    i2l = {
        int(index): label
        for label, index
        in l2i.items()
    }

    correct = 0
    errors = []
    confidences = []

    with torch.inference_mode():

        for sample in samples:
            temporal = (
                linear_interpolation(
                    sample[
                        "sequence"
                    ]
                )
            )

            padded = pad_features(
                temporal
            )

            prepared = standardize(
                padded,
                mean,
                std,
            )

            tensor = (
                torch
                .from_numpy(
                    prepared
                )
                .unsqueeze(0)
            )

            logits = model(
                tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=-1,
            )[0]

            probability, index = (
                torch.max(
                    probabilities,
                    dim=0,
                )
            )

            predicted = i2l[
                int(
                    index.item()
                )
            ]

            confidence = float(
                probability.item()
            )

            confidences.append(
                confidence
            )

            if (
                predicted
                == sample[
                    "label"
                ]
            ):
                correct += 1

            else:
                errors.append(
                    {
                        "file": (
                            sample[
                                "path"
                            ].name
                        ),
                        "expected": (
                            sample[
                                "label"
                            ]
                        ),
                        "predicted": (
                            predicted
                        ),
                        "confidence": (
                            confidence
                        ),
                        "frames": int(
                            sample[
                                "sequence"
                            ].shape[0]
                        ),
                        "probabilities": {
                            i2l[i]: float(
                                probabilities[
                                    i
                                ].item()
                            )
                            for i
                            in range(
                                4
                            )
                        },
                    }
                )

    return {
        "activation": (
            configuration[
                "activation"
            ]
        ),
        "norm_first": (
            configuration[
                "norm_first"
            ]
        ),
        "correct": (
            correct
        ),
        "total": (
            len(
                samples
            )
        ),
        "accuracy": float(
            correct
            / len(
                samples
            )
        ),
        "mean_max_probability": float(
            np.mean(
                confidences
            )
        ),
        "errors": (
            errors
        ),
    }


def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 78)
    print(
        "LughaLab KSL Forward "
        "Architecture Audit"
    )
    print("=" * 78)
    print()

    checkpoint = (
        load_checkpoint()
    )

    samples = (
        load_samples()
    )

    print(
        "Samples:",
        len(
            samples
        ),
    )

    print()

    results = []

    for configuration in CONFIGURATIONS:

        label = (
            f"activation="
            f"{configuration['activation']}, "
            f"norm_first="
            f"{configuration['norm_first']}"
        )

        print(
            "Testing",
            label,
        )

        try:
            result = (
                evaluate_configuration(
                    configuration,
                    checkpoint,
                    samples,
                )
            )

            results.append(
                result
            )

            print(
                "  accuracy:",
                f"{result['correct']}/"
                f"{result['total']} "
                f"="
                f"{result['accuracy']:.4f}",
            )

            print(
                "  mean confidence:",
                f"{result['mean_max_probability']:.4f}",
            )

            if result[
                "errors"
            ]:
                print(
                    "  errors:"
                )

                for error in (
                    result[
                        "errors"
                    ]
                ):
                    print(
                        "   ",
                        error[
                            "file"
                        ],
                        "| expected:",
                        error[
                            "expected"
                        ],
                        "| predicted:",
                        error[
                            "predicted"
                        ],
                        "| confidence:",
                        f"{error['confidence']:.4f}",
                    )

        except RuntimeError as error:
            print(
                "  configuration incompatible:"
            )
            print(
                " ",
                error,
            )

        print()

    results = sorted(
        results,
        key=lambda item: (
            item[
                "accuracy"
            ],
            item[
                "mean_max_probability"
            ],
        ),
        reverse=True,
    )

    print(
        "RANKING"
    )
    print("-" * 78)

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"{rank}. "
            f"activation="
            f"{result['activation']:<4} "
            f"norm_first="
            f"{str(result['norm_first']):<5} "
            f"accuracy="
            f"{result['accuracy']:.4f} "
            f"confidence="
            f"{result['mean_max_probability']:.4f}"
        )

    payload = {
        "experiment": (
            "KSL forward architecture "
            "sensitivity audit"
        ),
        "temporal_method": (
            "linear_interpolation"
        ),
        "results": (
            results
        ),
        "note": (
            "Strict state-dict compatibility "
            "does not establish non-parameter "
            "forward choices such as activation "
            "or pre/post LayerNorm. These are "
            "tested empirically."
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
        )

    print()
    print(
        f"Saved: {OUTPUT_PATH}"
    )
    print()


if __name__ == "__main__":
    main()