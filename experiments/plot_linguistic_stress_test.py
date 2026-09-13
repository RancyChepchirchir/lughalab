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
    / "swahili_linguistic_stress_test_results.json"
)


SWAHBERT = (
    "pranaydeeps/SwahBERT-base-cased"
)

AFROXLMR = (
    "Davlan/afro-xlmr-base"
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


def plot_phenomenon_sensitivity(data):
    summary = data[
        "phenomenon_summary"
    ]

    phenomena = list(
        summary.keys()
    )

    swahbert = [
        summary[p][SWAHBERT][
            "mean_representational_shift"
        ]
        for p in phenomena
    ]

    afroxlmr = [
        summary[p][AFROXLMR][
            "mean_representational_shift"
        ]
        for p in phenomena
    ]

    x = np.arange(
        len(phenomena)
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

    ax.set_xticks(x)

    ax.set_xticklabels(
        phenomena,
        rotation=30,
        ha="right",
    )

    ax.set_ylabel(
        "Mean representational shift (1 − cosine similarity)"
    )

    ax.set_title(
        "LughaLab — Swahili Linguistic Sensitivity Map"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "linguistic_sensitivity_map.png",
    )


def plot_pair_sensitivity(data):
    records = data[
        "records"
    ]

    labels = [
        row["id"]
        for row in records
    ]

    swahbert = [
        row["models"][
            SWAHBERT
        ][
            "representational_shift"
        ]
        for row in records
    ]

    afroxlmr = [
        row["models"][
            AFROXLMR
        ][
            "representational_shift"
        ]
        for row in records
    ]

    x = np.arange(
        len(labels)
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.plot(
        x,
        swahbert,
        marker="o",
        label="SwahBERT",
    )

    ax.plot(
        x,
        afroxlmr,
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
        "Representational shift"
    )

    ax.set_title(
        "Encoder Response to Controlled Swahili Transformations"
    )

    ax.legend()

    ax.grid(
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "linguistic_pair_sensitivity.png",
    )


def plot_model_difference(data):
    records = data[
        "records"
    ]

    rows = []

    for row in records:
        sw = row["models"][
            SWAHBERT
        ][
            "representational_shift"
        ]

        afro = row["models"][
            AFROXLMR
        ][
            "representational_shift"
        ]

        rows.append(
            (
                row["id"],
                row["phenomenon"],
                sw - afro,
            )
        )

    rows.sort(
        key=lambda item: abs(
            item[2]
        )
    )

    labels = [
        item[0]
        for item in rows
    ]

    differences = [
        item[2]
        for item in rows
    ]

    y = np.arange(
        len(labels)
    )

    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.barh(
        y,
        differences,
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
        "SwahBERT shift − AfroXLM-R shift"
    )

    ax.set_title(
        "Where Do the Encoders Respond Differently?"
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "linguistic_model_difference.png",
    )


def main():
    data = load_data()

    print()
    print("=" * 72)
    print(
        "LughaLab Linguistic Stress-Test Visualisation"
    )
    print("=" * 72)
    print()

    plot_phenomenon_sensitivity(
        data
    )

    plot_pair_sensitivity(
        data
    )

    plot_model_difference(
        data
    )

    print()
    print(
        "Visualisation complete."
    )
    print()


if __name__ == "__main__":
    main()