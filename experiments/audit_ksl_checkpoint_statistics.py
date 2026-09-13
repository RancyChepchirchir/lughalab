import sys
from pathlib import Path

import numpy as np


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
    bundle = (
        load_ksl_classifier()
    )

    mean = np.asarray(
        bundle[
            "mean"
        ],
        dtype=np.float32,
    )

    std = np.asarray(
        bundle[
            "std"
        ],
        dtype=np.float32,
    )

    hand_mean = mean[
        :126
    ]

    hand_std = std[
        :126
    ]

    padded_mean = mean[
        126:
    ]

    padded_std = std[
        126:
    ]

    print()
    print("=" * 78)
    print(
        "LughaLab KSL Checkpoint "
        "Statistics Audit"
    )
    print("=" * 78)
    print()

    print(
        "Full feature dimension:",
        mean.shape[0],
    )

    print()
    print(
        "HAND FEATURES [0:126]"
    )
    print(
        "  mean abs:",
        float(
            np.mean(
                np.abs(
                    hand_mean
                )
            )
        ),
    )
    print(
        "  std mean:",
        float(
            np.mean(
                hand_std
            )
        ),
    )
    print(
        "  zero std count:",
        int(
            np.isclose(
                hand_std,
                0.0,
                atol=1e-8,
            ).sum()
        ),
    )

    print()
    print(
        "PADDED FEATURES [126:225]"
    )
    print(
        "  max abs mean:",
        float(
            np.max(
                np.abs(
                    padded_mean
                )
            )
        ),
    )
    print(
        "  max abs std:",
        float(
            np.max(
                np.abs(
                    padded_std
                )
            )
        ),
    )
    print(
        "  zero mean count:",
        int(
            np.isclose(
                padded_mean,
                0.0,
                atol=1e-8,
            ).sum()
        ),
        "/",
        len(
            padded_mean
        ),
    )
    print(
        "  zero std count:",
        int(
            np.isclose(
                padded_std,
                0.0,
                atol=1e-8,
            ).sum()
        ),
        "/",
        len(
            padded_std
        ),
    )

    print()


if __name__ == "__main__":
    main()