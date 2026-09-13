import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

SUMMARY_PATH = (
    RESULTS_DIR
    / "swahili_similarity_summary.json"
)

RESULTS_PATH = (
    RESULTS_DIR
    / "swahili_similarity_results.json"
)

SWAHBERT = "pranaydeeps/SwahBERT-base-cased"
AFROXLMR = "Davlan/afro-xlmr-base"


def load_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_figure(fig, filename):
    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = FIGURES_DIR / filename

    fig.savefig(
        path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {path}")


# ============================================================
# Figure 1
# Category-level model comparison
# ============================================================


def plot_category_comparison(summary):
    rows = summary[
        "category_summary"
    ]

    categories = [
        row["category"]
        for row in rows
    ]

    swahbert = np.array(
        [
            row[
                "swahbert_mean_similarity"
            ]
            for row in rows
        ]
    )

    afroxlmr = np.array(
        [
            row[
                "afroxlmr_mean_similarity"
            ]
            for row in rows
        ]
    )

    x = np.arange(
        len(categories)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    ax.bar(
        x - width / 2,
        swahbert,
        width,
        label="SwahBERT",
    )

    ax.bar(
        x + width / 2,
        afroxlmr,
        width,
        label="AfroXLM-R",
    )

    ax.set_ylabel(
        "Mean cosine similarity"
    )

    ax.set_title(
        "LughaLab — Swahili Encoder Behaviour by Input Category"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        categories,
        rotation=30,
        ha="right",
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "category_similarity_comparison.png",
    )


# ============================================================
# Figure 2
# Pair-level model disagreement
# ============================================================


def plot_model_disagreement(summary):
    rows = summary[
        "largest_model_disagreements"
    ]

    rows = sorted(
        rows,
        key=lambda row: (
            row[
                "absolute_similarity_gap"
            ]
        ),
    )

    labels = [
        row["id"]
        for row in rows
    ]

    gaps = [
        row["similarity_gap"]
        for row in rows
    ]

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    y = np.arange(
        len(labels)
    )

    ax.barh(
        y,
        gaps,
    )

    ax.axvline(
        0,
        linewidth=1,
    )

    ax.set_yticks(y)

    ax.set_yticklabels(
        labels
    )

    ax.set_xlabel(
        "SwahBERT similarity − AfroXLM-R similarity"
    )

    ax.set_title(
        "Where Do the Swahili Encoders Disagree?"
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "model_disagreement.png",
    )


# ============================================================
# Figure 3
# Similarity-space scatter
# ============================================================


def plot_similarity_scatter(results):
    records = results[
        "records"
    ]

    categories = sorted(
        {
            row["category"]
            for row in records
        }
    )

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    for category in categories:
        rows = [
            row
            for row in records
            if row["category"]
            == category
        ]

        x = [
            row["models"][
                AFROXLMR
            ]["similarity"]
            for row in rows
        ]

        y = [
            row["models"][
                SWAHBERT
            ]["similarity"]
            for row in rows
        ]

        ax.scatter(
            x,
            y,
            label=category,
            s=70,
        )

        for row, x_value, y_value in zip(
            rows,
            x,
            y,
        ):
            ax.annotate(
                row["id"],
                (
                    x_value,
                    y_value,
                ),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
            )

    all_values = []

    for row in records:
        all_values.append(
            row["models"][
                SWAHBERT
            ]["similarity"]
        )

        all_values.append(
            row["models"][
                AFROXLMR
            ]["similarity"]
        )

    minimum = min(
        all_values
    )

    maximum = max(
        all_values
    )

    padding = max(
        0.02,
        (
            maximum - minimum
        ) * 0.08,
    )

    lower = minimum - padding
    upper = maximum + padding

    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=1,
        label="Equal similarity",
    )

    ax.set_xlim(
        lower,
        upper,
    )

    ax.set_ylim(
        lower,
        upper,
    )

    ax.set_xlabel(
        "AfroXLM-R cosine similarity"
    )

    ax.set_ylabel(
        "SwahBERT cosine similarity"
    )

    ax.set_title(
        "SwahBERT vs AfroXLM-R Similarity Space"
    )

    ax.legend(
        fontsize=8,
    )

    ax.grid(
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "similarity_space.png",
    )


# ============================================================
# Figure 4
# Latency comparison
# ============================================================


def plot_latency(results):
    records = results[
        "records"
    ]

    swahbert_latency = [
        row["models"][
            SWAHBERT
        ]["inference_time_ms"]
        for row in records
    ]

    afroxlmr_latency = [
        row["models"][
            AFROXLMR
        ]["inference_time_ms"]
        for row in records
    ]

    labels = [
        row["id"]
        for row in records
    ]

    x = np.arange(
        len(labels)
    )

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    ax.plot(
        x,
        swahbert_latency,
        marker="o",
        label="SwahBERT",
    )

    ax.plot(
        x,
        afroxlmr_latency,
        marker="o",
        label="AfroXLM-R",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right",
    )

    ax.set_ylabel(
        "Inference time (ms)"
    )

    ax.set_title(
        "Observed Encoder Runtime Across Benchmark Pairs"
    )

    ax.legend()

    ax.grid(
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "latency_comparison.png",
    )


# ============================================================
# Figure 5
# Category disagreement magnitude
# ============================================================


def plot_category_disagreement(summary):
    rows = summary[
        "category_summary"
    ]

    rows = sorted(
        rows,
        key=lambda row: (
            row[
                "mean_absolute_gap"
            ]
        ),
    )

    categories = [
        row["category"]
        for row in rows
    ]

    values = [
        row[
            "mean_absolute_gap"
        ]
        for row in rows
    ]

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    y = np.arange(
        len(categories)
    )

    ax.barh(
        y,
        values,
    )

    ax.set_yticks(y)

    ax.set_yticklabels(
        categories
    )

    ax.set_xlabel(
        "Mean absolute similarity gap"
    )

    ax.set_title(
        "Model Disagreement by Swahili Input Category"
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "category_disagreement.png",
    )


def main():
    summary = load_json(
        SUMMARY_PATH
    )

    results = load_json(
        RESULTS_PATH
    )

    print()
    print("=" * 70)
    print(
        "LughaLab Benchmark Visualisation"
    )
    print("=" * 70)
    print()

    plot_category_comparison(
        summary
    )

    plot_model_disagreement(
        summary
    )

    plot_similarity_scatter(
        results
    )

    plot_latency(
        results
    )

    plot_category_disagreement(
        summary
    )

    print()
    print(
        "Visualisation complete."
    )
    print()


if __name__ == "__main__":
    main()