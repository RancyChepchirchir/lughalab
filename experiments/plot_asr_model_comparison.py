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
    / "swahili_asr_model_comparison.json"
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


def plot_sample_wer(data):
    records = data[
        "records"
    ]

    labels = [
        row["id"]
        for row in records
    ]

    whisper = [
        row["models"][
            "whisper-small"
        ]["wer"]
        for row in records
    ]

    sauti = [
        row["models"][
            "sauti-v1"
        ]["wer"]
        for row in records
    ]

    x = np.arange(
        len(labels)
    )

    width = 0.36

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
        label="Sauti ASR v1",
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        labels,
        rotation=35,
        ha="right",
    )

    ax.set_ylabel(
        "Word error rate"
    )

    ax.set_title(
        "Zero-shot vs Swahili-Adapted ASR"
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "swahili_asr_model_wer.png",
    )


def plot_improvement(data):
    records = sorted(
        data["records"],
        key=lambda row: (
            row[
                "wer_improvement"
            ]
        ),
    )

    labels = [
        row["id"]
        for row in records
    ]

    values = [
        row[
            "wer_improvement"
        ]
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

    ax.axvline(
        0,
        linewidth=1,
    )

    ax.set_yticks(y)

    ax.set_yticklabels(
        labels
    )

    ax.set_xlabel(
        "WER(Whisper-small) − WER(Sauti)"
    )

    ax.set_title(
        "Where Does Swahili-Specific Adaptation Help?"
    )

    ax.grid(
        axis="x",
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "swahili_asr_adaptation_gain.png",
    )


def plot_accuracy_efficiency(data):
    summary = (
        data[
            "model_summary"
        ]
    )

    models = [
        "whisper-small",
        "sauti-v1",
    ]

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    for model in models:
        row = summary[
            model
        ]

        ax.scatter(
            row[
                "realtime_factor"
            ],
            row[
                "mean_wer"
            ],
            s=120,
        )

        ax.annotate(
            model,
            (
                row[
                    "realtime_factor"
                ],
                row[
                    "mean_wer"
                ],
            ),
            xytext=(8, 5),
            textcoords="offset points",
        )

    ax.set_xlabel(
        "Real-time factor"
    )

    ax.set_ylabel(
        "Mean WER"
    )

    ax.set_title(
        "LughaLab ASR Accuracy–Efficiency Trade-off"
    )

    ax.grid(
        alpha=0.2,
    )

    fig.tight_layout()

    save_figure(
        fig,
        "swahili_asr_accuracy_efficiency.png",
    )


def main():
    data = load_data()

    print()
    print("=" * 72)
    print(
        "LughaLab ASR Model Comparison Visualisation"
    )
    print("=" * 72)
    print()

    plot_sample_wer(
        data
    )

    plot_improvement(
        data
    )

    plot_accuracy_efficiency(
        data
    )

    print()
    print(
        "Visualisation complete."
    )
    print()


if __name__ == "__main__":
    main()