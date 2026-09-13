from pathlib import Path

import kagglehub


ROOT = Path(
    __file__
).resolve().parents[1]

TARGET_DIR = (
    ROOT
    / "data"
    / "ksl"
    / "source"
)

TARGET_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def main():
    print()
    print("=" * 78)
    print(
        "LughaLab KSL Dataset Download"
    )
    print("=" * 78)
    print()

    path = kagglehub.dataset_download(
        "joanwachuka/ksl-hand-landmarks"
    )

    print(
        "Dataset downloaded to:"
    )
    print(
        path
    )
    print()

    print(
        "NOTE:"
    )
    print(
        "Do not use dataset3/ for evaluation. "
        "The published model card reports that "
        "the official test samples are duplicated "
        "inside dataset3/, which would cause leakage."
    )
    print()


if __name__ == "__main__":
    main()