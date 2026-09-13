from pathlib import Path
from collections import Counter, defaultdict

import kagglehub
import numpy as np


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


def main():
    root = Path(
        kagglehub.dataset_download(
            KAGGLE_DATASET
        )
    )

    split_root = (
        root
        / "data_split"
        / "data_split"
    )

    print()
    print("=" * 78)
    print(
        "LughaLab KSL Dataset Shape Audit"
    )
    print("=" * 78)
    print()

    for split in [
        "train",
        "test",
    ]:
        split_dir = (
            split_root
            / split
        )

        print(
            f"{split.upper()} SPLIT"
        )
        print("-" * 78)

        shape_counts = Counter()
        frame_lengths = []
        class_counts = defaultdict(
            int
        )

        for label in LABELS:
            label_dir = (
                split_dir
                / label
            )

            files = sorted(
                label_dir.glob(
                    "*.npy"
                )
            )

            class_counts[
                label
            ] = len(
                files
            )

            for path in files:
                array = np.load(
                    path,
                    allow_pickle=False,
                )

                shape_counts[
                    tuple(
                        array.shape
                    )
                ] += 1

                if array.ndim == 2:
                    frame_lengths.append(
                        int(
                            array.shape[0]
                        )
                    )

        print(
            "Class counts:"
        )

        for label in LABELS:
            print(
                f"  {label:<8}: "
                f"{class_counts[label]}"
            )

        print()
        print(
            "Observed shapes:"
        )

        for shape, count in (
            shape_counts
            .most_common()
        ):
            print(
                f"  {shape}: {count}"
            )

        if frame_lengths:
            print()
            print(
                "Frame-length statistics:"
            )
            print(
                "  min:",
                min(
                    frame_lengths
                ),
            )
            print(
                "  max:",
                max(
                    frame_lengths
                ),
            )
            print(
                "  mean:",
                round(
                    float(
                        np.mean(
                            frame_lengths
                        )
                    ),
                    3,
                ),
            )
            print(
                "  median:",
                float(
                    np.median(
                        frame_lengths
                    )
                ),
            )

        print()


if __name__ == "__main__":
    main()