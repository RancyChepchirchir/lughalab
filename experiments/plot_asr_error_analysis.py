import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    ROOT
    / "experiments"
    / "results"
)

FIGURES_DIR = (
    RESULTS_DIR
    / "figures"
)

INPUT_PATH = (
    RESULTS_DIR
    / "swahili_asr_error_analysis.json"
)


MODELS = [
    "whisper-small",
    "sauti-v1",
]


def load_data():
    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_figure(
    fig,
    filename,
):
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = (
        FIGURES_DIR
        / filename
    )

    fig.savefig(
        output,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output}"
    )


def plot_error_composition(
    data,
):
    summary = (
        data[
            "model_summary"
        ]
    )

    substitution = [
        summary[
            model
        ][
            "substitution_rate"
        ]
        for model in MODELS
    ]

    deletion = [
        summary[
            model
        ][
            "deletion_rate"
        ]
        for model in MODELS
    ]

    insertion = [
        summary[
            model
        ][
            "insertion_rate"
        ]
        for model in MODELS
    ]

    x = np.arange(
        len(MODELS)
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    ax.bar(
        x,
        substitution,
        label="Substitution",
    )

    ax.bar(
        x,
        deletion,
        bottom=substitution,
        label="Deletion",
    )

    bottom = (
        np.array(
            substitution
        )
        + np.array(
            deletion
        )
    )

    ax.bar(
        x,
        insertion,
        bottom=bottom,
        label="Insertion",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        MODELS
    )

    ax.set_ylabel(
        "Error rate"
    )

    ax.set_title(
        "Swahili ASR Error Composition"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "swahili_asr_error_composition.png",
    )


def plot_category_wer(
    data,
):
    summary = (
        data[
            "category_summary"
        ]
    )

    categories = list(
        summary.keys()
    )

    x = np.arange(
        len(categories)
    )

    width = 0.36

    whisper = [
        summary[
            category
        ][
            "whisper-small"
        ][
            "corpus_wer"
        ]
        for category in categories
    ]

    sauti = [
        summary[
            category
        ][
            "sauti-v1"
        ][
            "corpus_wer"
        ]
        for category in categories
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        x - width / 2,
        whisper,
        width,
        label="Whisper-small",
    )

    ax.bar(
        x + width / 2,
        sauti,
        width,
        label="Sauti-v1",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        categories,
        rotation=30,
        ha="right",
    )

    ax.set_ylabel(
        "Corpus WER"
    )

    ax.set_title(
        "ASR Error by Speech Category"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "swahili_asr_error_by_category.png",
    )


def main():
    data = load_data()

    print()
    print("=" * 72)
    print(
        "LughaLab ASR Error Visualisation"
    )
    print("=" * 72)
    print()

    plot_error_composition(
        data
    )

    plot_category_wer(
        data
    )

    print()
    print(
        "Visualisation complete."
    )
    print()


if __name__ == "__main__":
    main()