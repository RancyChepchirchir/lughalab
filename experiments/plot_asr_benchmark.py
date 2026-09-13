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
    / "swahili_asr_benchmark_results.json"
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

    path = (
        FIGURES_DIR
        / filename
    )

    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {path}"
    )


def plot_category_errors(data):
    summary = (
        data[
            "category_summary"
        ]
    )

    categories = list(
        summary.keys()
    )

    wers = [
        summary[c]["mean_wer"]
        for c in categories
    ]

    cers = [
        summary[c]["mean_cer"]
        for c in categories
    ]

    x = np.arange(
        len(categories)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.bar(
        x - width / 2,
        wers,
        width,
        label="WER",
    )

    ax.bar(
        x + width / 2,
        cers,
        width,
        label="CER",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        categories,
        rotation=30,
        ha="right",
    )

    ax.set_ylabel(
        "Error rate"
    )

    ax.set_title(
        "LughaLab — Swahili ASR Error by Speech Category"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "swahili_asr_category_error.png",
    )


def plot_sample_wer(data):
    records = (
        data["records"]
    )

    records = sorted(
        records,
        key=lambda row: (
            row["wer"]
        ),
    )

    labels = [
        row["id"]
        for row in records
    ]

    values = [
        row["wer"]
        for row in records
    ]

    y = np.arange(
        len(labels)
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.barh(
        y,
        values,
    )

    ax.set_yticks(y)

    ax.set_yticklabels(
        labels
    )

    ax.set_xlabel(
        "Word error rate"
    )

    ax.set_title(
        "Which Swahili Recordings Are Hardest?"
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "swahili_asr_sample_wer.png",
    )


def main():
    data = load_data()

    print()
    print("=" * 72)
    print(
        "LughaLab ASR Benchmark Visualisation"
    )
    print("=" * 72)
    print()

    plot_category_errors(
        data
    )

    plot_sample_wer(
        data
    )

    print()
    print(
        "Visualisation complete."
    )
    print()


if __name__ == "__main__":
    main()