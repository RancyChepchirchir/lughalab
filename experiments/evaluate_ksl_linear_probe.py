from pathlib import Path
import json

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)


ROOT = Path(__file__).resolve().parents[1]

BANK_PATH = (
    ROOT
    / "experiments"
    / "results"
    / "ksl_ekitabu_open_set_bank.npz"
)

OUTPUT_PATH = (
    ROOT
    / "experiments"
    / "results"
    / "ksl_linear_probe.json"
)


def center_by_signer(
    embeddings,
    signers,
):
    centered = np.empty_like(
        embeddings
    )

    for signer in np.unique(signers):
        mask = signers == signer

        mean = embeddings[
            mask
        ].mean(
            axis=0,
            keepdims=True,
        )

        centered[mask] = (
            embeddings[mask] - mean
        )

    return centered


def evaluate_probe(
    embeddings,
    classes,
    roles,
):
    train_mask = (
        roles == "known_reference"
    )

    val_mask = (
        roles == "known_validation"
    )

    test_mask = (
        roles == "known_test"
    )

    x_train = embeddings[
        train_mask
    ]
    y_train = classes[
        train_mask
    ]

    x_val = embeddings[
        val_mask
    ]
    y_val = classes[
        val_mask
    ]

    x_test = embeddings[
        test_mask
    ]
    y_test = classes[
        test_mask
    ]

    model = LogisticRegression(
        max_iter=5000,
        solver="lbfgs",
        multi_class="auto",
        random_state=42,
    )

    model.fit(
        x_train,
        y_train,
    )

    result = {}

    for name, x, y in [
        (
            "train",
            x_train,
            y_train,
        ),
        (
            "validation",
            x_val,
            y_val,
        ),
        (
            "test",
            x_test,
            y_test,
        ),
    ]:
        predictions = model.predict(x)

        probabilities = (
            model.predict_proba(x)
        )

        result[name] = {
            "accuracy": float(
                accuracy_score(
                    y,
                    predictions,
                )
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    y,
                    predictions,
                )
            ),
            "macro_f1": float(
                f1_score(
                    y,
                    predictions,
                    average="macro",
                )
            ),
            "mean_max_probability": float(
                probabilities.max(
                    axis=1
                ).mean()
            ),
        }

    return result


def print_block(
    title,
    result,
):
    print()
    print(title)
    print("-" * 78)

    print(
        f"{'Split':<15}"
        f"{'Accuracy':>12}"
        f"{'Bal. Acc.':>12}"
        f"{'Macro-F1':>12}"
        f"{'Mean MSP':>12}"
    )

    print("-" * 78)

    for split in [
        "train",
        "validation",
        "test",
    ]:
        block = result[split]

        print(
            f"{split:<15}"
            f"{block['accuracy']:>12.4f}"
            f"{block['balanced_accuracy']:>12.4f}"
            f"{block['macro_f1']:>12.4f}"
            f"{block['mean_max_probability']:>12.4f}"
        )


def main():
    print("=" * 78)
    print(
        "LughaLab KSL Frozen-Representation "
        "Linear Probe"
    )
    print("=" * 78)

    data = np.load(
        BANK_PATH,
        allow_pickle=True,
    )

    embeddings = data[
        "embeddings"
    ].astype(
        np.float32
    )

    classes = data["classes"]
    roles = data["roles"]
    signers = data["signers"]

    raw_result = evaluate_probe(
        embeddings,
        classes,
        roles,
    )

    centered_embeddings = (
        center_by_signer(
            embeddings,
            signers,
        )
    )

    centered_result = evaluate_probe(
        centered_embeddings,
        classes,
        roles,
    )

    print_block(
        "RAW REPRESENTATION",
        raw_result,
    )

    print_block(
        "SIGNER-CENTERED REPRESENTATION",
        centered_result,
    )

    print()
    print("RAW -> CENTERED DELTA")
    print("-" * 78)

    for split in [
        "validation",
        "test",
    ]:
        raw = raw_result[
            split
        ]["accuracy"]

        centered = centered_result[
            split
        ]["accuracy"]

        print(
            f"{split:<15}"
            f"{raw:.4f} -> "
            f"{centered:.4f} "
            f"({centered - raw:+.4f})"
        )

    chance = (
        1.0
        / len(
            np.unique(
                classes[
                    roles
                    == "known_reference"
                ]
            )
        )
    )

    print()
    print(
        "Chance accuracy: "
        f"{chance:.4f}"
    )

    print()
    print("IMPORTANT")
    print("-" * 78)

    print(
        "The encoder is frozen. "
        "Only a multinomial linear classifier "
        "is fitted to known-reference embeddings."
    )

    print(
        "Training uses signers 01-10; "
        "validation uses 11-12; "
        "test uses 13-15."
    )

    print(
        "Signer-centered evaluation is "
        "transductive because each signer's "
        "mean uses multiple samples from "
        "that signer."
    )

    output = {
        "chance_accuracy": float(
            chance
        ),
        "raw": raw_result,
        "signer_centered": (
            centered_result
        ),
        "note": (
            "Frozen 256-D KSL Transformer "
            "representation. Linear probe "
            "trained only on known-reference "
            "signers. Signer centering is a "
            "transductive diagnostic."
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            output,
            handle,
            indent=2,
        )

    print()
    print("Saved:")
    print(f"  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()