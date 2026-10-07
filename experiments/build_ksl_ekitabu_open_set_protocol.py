from __future__ import annotations

import json
from pathlib import Path

import kagglehub
import pandas as pd


SEED = 42
N_KNOWN = 20


def main() -> None:
    root = Path(
        kagglehub.dataset_download(
            "ekitabu/kenyan-sign-language-videos"
        )
    )

    df = pd.read_csv(
        root / "splits_info.csv"
    )

    required = {
        "filename",
        "class",
        "split",
        "relative_path",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns: {sorted(missing)}"
        )

    # Recover signer identity from filenames.
    df["signer"] = (
        df["filename"]
        .astype(str)
        .str.extract(
            r"Signer_(\d+)_",
            expand=False,
        )
    )

    if df["signer"].isna().any():
        bad = df.loc[
            df["signer"].isna(),
            "filename",
        ].tolist()

        raise ValueError(
            "Could not recover signer from: "
            f"{bad[:10]}"
        )

    classes = sorted(
        df["class"].unique().tolist()
    )

    if len(classes) != 30:
        raise ValueError(
            "Expected 30 classes, "
            f"found {len(classes)}."
        )

    # Deterministic random class partition.
    class_series = pd.Series(classes)

    known = (
        class_series.sample(
            n=N_KNOWN,
            random_state=SEED,
        )
        .sort_values()
        .tolist()
    )

    unknown = sorted(
        set(classes) - set(known)
    )

    train_signers = sorted(
        df.loc[
            df["split"] == "train",
            "signer",
        ].unique().tolist()
    )

    val_signers = sorted(
        df.loc[
            df["split"] == "val",
            "signer",
        ].unique().tolist()
    )

    test_signers = sorted(
        df.loc[
            df["split"] == "test",
            "signer",
        ].unique().tolist()
    )

    # Verify signer-disjoint protocol.
    assert not (
        set(train_signers)
        & set(val_signers)
    )

    assert not (
        set(train_signers)
        & set(test_signers)
    )

    assert not (
        set(val_signers)
        & set(test_signers)
    )

    def subset(
        class_names: list[str],
        split: str,
    ) -> pd.DataFrame:
        return df[
            df["class"].isin(class_names)
            & (df["split"] == split)
        ].copy()

    known_train = subset(
        known,
        "train",
    )

    known_val = subset(
        known,
        "val",
    )

    known_test = subset(
        known,
        "test",
    )

    unknown_test = subset(
        unknown,
        "test",
    )

    output = {
        "seed": SEED,
        "dataset": (
            "ekitabu/"
            "kenyan-sign-language-videos"
        ),
        "classes_total": len(classes),
        "known_classes": known,
        "unknown_classes": unknown,
        "train_signers": train_signers,
        "validation_signers": val_signers,
        "test_signers": test_signers,
        "counts": {
            "known_train": len(
                known_train
            ),
            "known_validation": len(
                known_val
            ),
            "known_test": len(
                known_test
            ),
            "unknown_test": len(
                unknown_test
            ),
        },
    }

    output_dir = Path(
        "experiments/results"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        output_dir
        / "ksl_ekitabu_open_set_protocol.json"
    )

    with json_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            output,
            handle,
            indent=2,
        )

    manifest = df.copy()

    manifest["open_set_role"] = (
        "unused"
    )

    manifest.loc[
        manifest["class"].isin(known)
        & (manifest["split"] == "train"),
        "open_set_role",
    ] = "known_reference"

    manifest.loc[
        manifest["class"].isin(known)
        & (manifest["split"] == "val"),
        "open_set_role",
    ] = "known_validation"

    manifest.loc[
        manifest["class"].isin(known)
        & (manifest["split"] == "test"),
        "open_set_role",
    ] = "known_test"

    manifest.loc[
        manifest["class"].isin(unknown)
        & (manifest["split"] == "test"),
        "open_set_role",
    ] = "unknown_test"

    csv_path = (
        output_dir
        / "ksl_ekitabu_open_set_manifest.csv"
    )

    manifest.to_csv(
        csv_path,
        index=False,
    )

    print("=" * 78)
    print(
        "LughaLab eKitabu Open-Set Protocol"
    )
    print("=" * 78)

    print(
        "\nKNOWN CLASSES (20)"
    )
    print("-" * 78)

    for name in known:
        print(name)

    print(
        "\nUNKNOWN CLASSES (10)"
    )
    print("-" * 78)

    for name in unknown:
        print(name)

    print(
        "\nSIGNER PARTITION"
    )
    print("-" * 78)

    print(
        "Train:",
        train_signers,
    )
    print(
        "Validation:",
        val_signers,
    )
    print(
        "Test:",
        test_signers,
    )

    print(
        "\nSAMPLE COUNTS"
    )
    print("-" * 78)

    for key, value in output[
        "counts"
    ].items():
        print(
            f"{key:<25} {value}"
        )

    print(
        "\nSaved:"
    )
    print(
        f"  {json_path.resolve()}"
    )
    print(
        f"  {csv_path.resolve()}"
    )


if __name__ == "__main__":
    main()