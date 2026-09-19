import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"

sys.path.insert(
    0,
    str(BACKEND),
)


from app.services.ksl_classifier import (
    load_ksl_classifier,
)


def main():
    bundle = load_ksl_classifier()

    model = bundle["model"]
    device = bundle["device"]

    torch.manual_seed(42)

    x = torch.randn(
        1,
        64,
        225,
        device=device,
    )

    with torch.inference_mode():
        logits_forward = model(
            x
        )

        embedding = model.encode(
            x
        )

        logits_manual = model.head(
            embedding
        )

    difference = (
        logits_forward
        - logits_manual
    ).abs()

    max_difference = float(
        difference.max().cpu()
    )

    print("=" * 78)
    print(
        "LughaLab KSL Embedding Refactor Verification"
    )
    print("=" * 78)

    print()
    print(
        "Input shape:",
        tuple(x.shape),
    )

    print(
        "Embedding shape:",
        tuple(embedding.shape),
    )

    print(
        "Logit shape:",
        tuple(logits_forward.shape),
    )

    print()
    print(
        "Maximum logit difference:",
        max_difference,
    )

    print()

    if max_difference < 1e-6:
        print(
            "PASS: forward(x) == "
            "head(encode(x))"
        )
    else:
        raise RuntimeError(
            "Embedding refactor changed "
            "model inference."
        )


if __name__ == "__main__":
    main()