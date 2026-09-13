import sys
from pathlib import Path

import torch


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


from app.services.ksl_classifier import (
    load_ksl_classifier,
)


def main():
    print()
    print("=" * 78)
    print(
        "LughaLab KSL "
        "Checkpoint Load Test"
    )
    print("=" * 78)
    print()

    bundle = (
        load_ksl_classifier()
    )

    model = bundle[
        "model"
    ]

    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )

    print(
        "Strict state_dict load: PASS"
    )

    print(
        "Device:",
        bundle[
            "device"
        ],
    )

    print(
        "Parameters:",
        f"{parameter_count:,}",
    )

    print(
        "Labels:",
        bundle[
            "index_to_label"
        ],
    )

    print(
        "Mean shape:",
        bundle[
            "mean"
        ].shape,
    )

    print(
        "Std shape:",
        bundle[
            "std"
        ].shape,
    )

    dummy = torch.zeros(
        (
            1,
            64,
            225,
        ),
        dtype=torch.float32,
        device=bundle[
            "device"
        ],
    )

    with torch.inference_mode():
        logits = model(
            dummy
        )

    print(
        "Dummy input:",
        tuple(
            dummy.shape
        ),
    )

    print(
        "Output logits:",
        tuple(
            logits.shape
        ),
    )

    if logits.shape != (
        1,
        4,
    ):
        raise RuntimeError(
            "Unexpected output shape."
        )

    print()
    print(
        "Forward pass: PASS"
    )
    print()


if __name__ == "__main__":
    main()