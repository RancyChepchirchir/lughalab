from pathlib import Path
import csv
import shutil

import kagglehub


ROOT = Path(__file__).resolve().parents[1]

DESTINATION = (
    ROOT
    / "data"
    / "ksl"
    / "natural_ood"
    / "near"
)

MANIFEST = (
    ROOT
    / "data"
    / "ksl"
    / "natural_ood"
    / "near_manifest.csv"
)

DATASET = (
    "ekitabu/"
    "kenyan-sign-language-videos"
)

SELECTED_GLOSSES = [
    "Apple",
    "Friend",
    "Market",
    "Monday",
    "Proud",
    "Teach",
    "Tortoise",
    "Ugali",
]

SAMPLES_PER_GLOSS = 4

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".webm",
    ".m4v",
}


def signer_from_filename(
    path: Path,
) -> str:
    """
    Examples:

        Signer_08_3.mov
        Signer_11_A064.mov

    Both correspond to signer 08 / 11.

    We deliberately retain only the
    first two underscore-separated
    components.
    """

    parts = path.stem.split("_")

    if (
        len(parts) >= 2
        and parts[0].lower()
        == "signer"
    ):
        return (
            f"{parts[0]}_{parts[1]}"
        )

    return path.stem


def select_across_signers(
    files,
    count,
):
    """
    Prefer one sample from each signer
    before taking additional recordings
    from an already represented signer.
    """

    files = sorted(
        files,
        key=lambda p: (
            signer_from_filename(p),
            p.name,
        ),
    )

    by_signer = {}

    for path in files:
        signer = (
            signer_from_filename(
                path
            )
        )

        by_signer.setdefault(
            signer,
            [],
        ).append(
            path
        )

    selected = []

    #
    # First pass:
    # one video per signer.
    #
    for signer in sorted(
        by_signer
    ):
        selected.append(
            by_signer[
                signer
            ][0]
        )

        if len(selected) == count:
            return selected

    #
    # Second pass:
    # fill remaining slots if a gloss
    # has fewer unique signers than the
    # requested number of samples.
    #
    if len(selected) < count:
        already = set(
            selected
        )

        remaining = [
            path
            for path in files
            if path not in already
        ]

        selected.extend(
            remaining[
                :count - len(selected)
            ]
        )

    return selected


def main():
    print("=" * 78)
    print(
        "LughaLab KSL Near-OOD "
        "Benchmark Builder"
    )
    print("=" * 78)

    dataset_root = Path(
        kagglehub.dataset_download(
            DATASET
        )
    )

    print()
    print(
        "Dataset:",
        DATASET,
    )

    print(
        "Source root:",
        dataset_root,
    )

    DESTINATION.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    print()
    print("SELECTION")
    print("-" * 78)

    for gloss in SELECTED_GLOSSES:
        gloss_dir = (
            dataset_root / gloss
        )

        if not gloss_dir.exists():
            raise FileNotFoundError(
                "Expected gloss directory "
                f"not found: {gloss_dir}"
            )

        candidates = [
            path
            for path in gloss_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower()
                in VIDEO_EXTENSIONS
            )
        ]

        selected = (
            select_across_signers(
                candidates,
                SAMPLES_PER_GLOSS,
            )
        )

        if (
            len(selected)
            < SAMPLES_PER_GLOSS
        ):
            raise RuntimeError(
                f"{gloss} contains only "
                f"{len(selected)} usable "
                "videos."
            )

        print()
        print(
            f"{gloss}: "
            f"{len(candidates)} available"
        )

        for sample_number, source in enumerate(
            selected,
            start=1,
        ):
            signer = (
                signer_from_filename(
                    source
                )
            )

            destination_name = (
                f"{gloss.lower()}_"
                f"{sample_number:02d}_"
                f"{signer.lower()}"
                f"{source.suffix.lower()}"
            )

            destination = (
                DESTINATION
                / destination_name
            )

            shutil.copy2(
                source,
                destination,
            )

            relative_source = (
                source.relative_to(
                    dataset_root
                )
            )

            relative_destination = (
                destination.relative_to(
                    ROOT
                )
            )

            print(
                f"  {sample_number}. "
                f"{signer:<12} "
                f"{source.name}"
            )

            rows.append(
                {
                    "ood_category": (
                        "near"
                    ),
                    "gloss": gloss,
                    "signer": signer,
                    "source_dataset": (
                        DATASET
                    ),
                    "source_relative_path": (
                        str(
                            relative_source
                        )
                    ),
                    "local_path": (
                        str(
                            relative_destination
                        )
                    ),
                }
            )

    with MANIFEST.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ood_category",
                "gloss",
                "signer",
                "source_dataset",
                "source_relative_path",
                "local_path",
            ],
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print()
    print("SUMMARY")
    print("-" * 78)

    print(
        "Glosses:",
        len(SELECTED_GLOSSES),
    )

    print(
        "Videos:",
        len(rows),
    )

    print(
        "Videos per gloss:",
        SAMPLES_PER_GLOSS,
    )

    unique_signers = sorted(
        {
            row["signer"]
            for row in rows
        }
    )

    print(
        "Unique signer IDs:",
        len(unique_signers),
    )

    print(
        "Signers:",
        ", ".join(
            unique_signers
        ),
    )

    print()
    print(
        "Destination:",
        DESTINATION,
    )

    print(
        "Manifest:",
        MANIFEST,
    )


if __name__ == "__main__":
    main()