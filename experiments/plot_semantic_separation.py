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
    / "swahili_semantic_separation.json"
)


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


def plot_group_separation(data):
    swahbert = (
        data["models"][
            "swahbert"
        ]
    )

    afroxlmr = (
        data["models"][
            "afroxlmr"
        ]
    )

    groups = [
        "Similar",
        "Related",
        "Contrast",
        "Unrelated",
    ]

    sw_values = [
        swahbert[
            "similar_group"
        ][
            "mean_similarity"
        ],

        swahbert[
            "related_group"
        ][
            "mean_similarity"
        ],

        swahbert[
            "contrast_group"
        ][
            "mean_similarity"
        ],

        swahbert[
            "unrelated_group"
        ][
            "mean_similarity"
        ],
    ]

    afro_values = [
        afroxlmr[
            "similar_group"
        ][
            "mean_similarity"
        ],

        afroxlmr[
            "related_group"
        ][
            "mean_similarity"
        ],

        afroxlmr[
            "contrast_group"
        ][
            "mean_similarity"
        ],

        afroxlmr[
            "unrelated_group"
        ][
            "mean_similarity"
        ],
    ]

    x = np.arange(
        len(groups)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.bar(
        x - width / 2,
        sw_values,
        width,
        label="SwahBERT",
    )

    ax.bar(
        x + width / 2,
        afro_values,
        width,
        label="AfroXLM-R",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        groups
    )

    ax.set_ylabel(
        "Mean cosine similarity"
    )

    ax.set_title(
        "LughaLab — Semantic Separation Diagnostic"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "semantic_separation.png",
    )


def plot_separation_scores(data):
    swahbert = (
        data["models"][
            "swahbert"
        ]
    )

    afroxlmr = (
        data["models"][
            "afroxlmr"
        ]
    )

    models = [
        "SwahBERT",
        "AfroXLM-R",
    ]

    semantic = [
        swahbert[
            "semantic_separation"
        ],
        afroxlmr[
            "semantic_separation"
        ],
    ]

    contrast = [
        swahbert[
            "contrast_separation"
        ],
        afroxlmr[
            "contrast_separation"
        ],
    ]

    x = np.arange(
        len(models)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.bar(
        x - width / 2,
        semantic,
        width,
        label=(
            "Similar − unrelated"
        ),
    )

    ax.bar(
        x + width / 2,
        contrast,
        width,
        label=(
            "Similar − contrast"
        ),
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        models
    )

    ax.set_ylabel(
        "Similarity separation"
    )

    ax.set_title(
        "How Well Does Each Encoder Separate Meaning?"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "semantic_separation_scores.png",
    )


def main():
    data = load_data()

    print()
    print("=" * 72)
    print(
        "LughaLab Semantic Separation Visualisation"
    )
    print("=" * 72)
    print()

    plot_group_separation(
        data
    )

    plot_separation_scores(
        data
    )

    print()
    print(
        "Visualisation complete."
    )
    print()


if __name__ == "__main__":
    main()