from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]

RESULTS = (
    ROOT
    / "experiments"
    / "results"
)

OUTPUT = (
    ROOT
    / "paper"
    / "figures"
)

OUTPUT.mkdir(
    parents=True,
    exist_ok=True,
)


def save_figure(
    fig,
    name: str,
) -> None:
    """
    Save both vector PDF and high-resolution PNG.
    """
    pdf_path = OUTPUT / f"{name}.pdf"
    png_path = OUTPUT / f"{name}.png"

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


def pearson_r(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    return float(
        np.corrcoef(
            x,
            y,
        )[0, 1]
    )


def figure_ood_distribution(
    evaluation,
) -> None:
    """
    Held-out ID versus expanded natural-OOD
    5-NN score distributions.
    """
    id_scores = evaluation["id_5nn"]
    ood_scores = evaluation["ood_5nn"]

    id_max = float(
        np.max(id_scores)
    )

    ood_min = float(
        np.min(ood_scores)
    )

    threshold = float(
        np.quantile(
            id_scores,
            0.99,
        )
    )

    fig, ax = plt.subplots(
        figsize=(7.4, 4.8)
    )

    bins = np.linspace(
        min(
            float(id_scores.min()),
            float(ood_scores.min()),
        ),
        max(
            float(id_scores.max()),
            float(ood_scores.max()),
        ),
        35,
    )

    ax.hist(
        id_scores,
        bins=bins,
        alpha=0.65,
        label=(
            f"Held-out ID "
            f"(n={len(id_scores)})"
        ),
    )

    ax.hist(
        ood_scores,
        bins=bins,
        alpha=0.65,
        label=(
            f"Natural OOD "
            f"(n={len(ood_scores)})"
        ),
    )

    ax.axvline(
        threshold,
        linestyle="--",
        linewidth=1.5,
        label=(
            "99th-percentile "
            f"ID threshold = {threshold:.2f}"
        ),
    )

    ax.axvspan(
        id_max,
        ood_min,
        alpha=0.12,
        label=(
            "Observed separation gap "
            f"= {ood_min - id_max:.2f}"
        ),
    )

    ax.set_xlabel(
        "Mean 5-nearest-neighbour "
        "embedding distance"
    )

    ax.set_ylabel(
        "Number of samples"
    )

    ax.set_title(
        "Held-out ID and Natural-OOD "
        "Embedding Distances"
    )

    ax.legend(
        frameon=False,
        fontsize=8.5,
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    fig.tight_layout()

    save_figure(
        fig,
        "fig02_ood_score_distribution",
    )


def figure_per_gloss(
    evaluation,
) -> None:
    """
    Distribution of 5-NN scores for each
    natural OOD gloss.
    """
    scores = evaluation["ood_5nn"]
    glosses = evaluation["glosses"]

    unique_glosses = sorted(
        np.unique(glosses)
    )

    groups = [
        scores[
            glosses == gloss
        ]
        for gloss in unique_glosses
    ]

    medians = np.array(
        [
            np.median(group)
            for group in groups
        ]
    )

    order = np.argsort(
        medians
    )

    ordered_glosses = [
        unique_glosses[i]
        for i in order
    ]

    ordered_groups = [
        groups[i]
        for i in order
    ]

    fig, ax = plt.subplots(
        figsize=(9.5, 7.5)
    )

    ax.boxplot(
        ordered_groups,
        vert=False,
        tick_labels=ordered_glosses,
        showfliers=True,
    )

    ax.set_xlabel(
        "Mean 5-nearest-neighbour "
        "embedding distance"
    )

    ax.set_ylabel(
        "Unseen KSL gloss"
    )

    ax.set_title(
        "Natural-OOD Distance by Unseen Gloss"
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    fig.tight_layout()

    save_figure(
        fig,
        "fig03_per_gloss_distance",
    )


def figure_softmax_vs_ood(
    evaluation,
) -> None:
    """
    Closed-set confidence against geometric
    novelty.
    """
    x = evaluation["ood_5nn"]
    y = evaluation["max_softmax"]

    r = pearson_r(
        x,
        y,
    )

    slope, intercept = np.polyfit(
        x,
        y,
        1,
    )

    xx = np.linspace(
        float(x.min()),
        float(x.max()),
        200,
    )

    fig, ax = plt.subplots(
        figsize=(6.5, 5.0)
    )

    ax.scatter(
        x,
        y,
        alpha=0.55,
        s=24,
    )

    ax.plot(
        xx,
        slope * xx + intercept,
        linewidth=1.6,
    )

    ax.set_xlabel(
        "5-NN OOD score"
    )

    ax.set_ylabel(
        "Maximum softmax probability"
    )

    ax.set_title(
        "Closed-Set Confidence vs "
        "Embedding Novelty"
    )

    ax.text(
        0.04,
        0.06,
        f"Pearson r = {r:.3f}",
        transform=ax.transAxes,
        fontsize=10,
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    fig.tight_layout()

    save_figure(
        fig,
        "fig04_softmax_vs_ood",
    )


def figure_coverage_vs_ood(
    evaluation,
) -> None:
    """
    Landmark detection coverage against
    geometric novelty.
    """
    x = evaluation[
        "detection_rates"
    ]

    y = evaluation[
        "ood_5nn"
    ]

    r = pearson_r(
        x,
        y,
    )

    slope, intercept = np.polyfit(
        x,
        y,
        1,
    )

    xx = np.linspace(
        float(x.min()),
        float(x.max()),
        200,
    )

    fig, ax = plt.subplots(
        figsize=(6.5, 5.0)
    )

    ax.scatter(
        x,
        y,
        alpha=0.55,
        s=24,
    )

    ax.plot(
        xx,
        slope * xx + intercept,
        linewidth=1.6,
    )

    ax.set_xlabel(
        "Landmark detection rate"
    )

    ax.set_ylabel(
        "5-NN OOD score"
    )

    ax.set_title(
        "Landmark Coverage vs "
        "Embedding Novelty"
    )

    ax.text(
        0.04,
        0.94,
        f"Pearson r = {r:.3f}",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    fig.tight_layout()

    save_figure(
        fig,
        "fig05_coverage_vs_ood",
    )


def figure_closed_set_collapse(
    natural,
) -> None:
    """
    Distribution of forced closed-set labels
    assigned to expanded natural OOD.
    """
    predictions = natural[
        "predicted_labels"
    ]

    labels = [
        "father",
        "hello",
        "is",
        "my",
    ]

    counts = Counter(
        str(value)
        for value in predictions
    )

    values = np.array(
        [
            counts.get(
                label,
                0,
            )
            for label in labels
        ]
    )

    proportions = (
        values
        / len(predictions)
    )

    fig, ax = plt.subplots(
        figsize=(6.5, 4.8)
    )

    bars = ax.bar(
        labels,
        proportions,
    )

    for bar, count, proportion in zip(
        bars,
        values,
        proportions,
    ):
        ax.text(
            bar.get_x()
            + bar.get_width() / 2,
            bar.get_height()
            + 0.015,
            (
                f"{count}\n"
                f"{100 * proportion:.1f}%"
            ),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylim(
        0,
        1.08,
    )

    ax.set_xlabel(
        "Forced closed-set prediction"
    )

    ax.set_ylabel(
        "Proportion of natural-OOD videos"
    )

    ax.set_title(
        "Closed-Set Prediction Collapse "
        "on Unseen KSL Vocabulary"
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    fig.tight_layout()

    save_figure(
        fig,
        "fig06_closed_set_collapse",
    )


def figure_knn_comparison(
    evaluation,
) -> None:
    """
    ID/OOD distance separation for different
    neighbourhood sizes.
    """
    ks = [
        1,
        3,
        5,
        10,
        20,
    ]

    id_means = np.array(
        [
            np.mean(
                evaluation[
                    f"id_{k}nn"
                ]
            )
            for k in ks
        ]
    )

    ood_means = np.array(
        [
            np.mean(
                evaluation[
                    f"ood_{k}nn"
                ]
            )
            for k in ks
        ]
    )

    fig, ax = plt.subplots(
        figsize=(6.7, 4.8)
    )

    ax.plot(
        ks,
        id_means,
        marker="o",
        label="Held-out ID",
    )

    ax.plot(
        ks,
        ood_means,
        marker="o",
        label="Natural OOD",
    )

    ax.set_xticks(
        ks
    )

    ax.set_xlabel(
        "Number of neighbours, k"
    )

    ax.set_ylabel(
        "Mean embedding distance"
    )

    ax.set_title(
        "ID/OOD Separation Across "
        "Neighbourhood Sizes"
    )

    ax.legend(
        frameon=False
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    fig.tight_layout()

    save_figure(
        fig,
        "fig07_knn_comparison",
    )


def main() -> None:
    evaluation_path = (
        RESULTS
        / "ksl_expanded_natural_ood_evaluation.npz"
    )

    natural_path = (
        RESULTS
        / "ksl_expanded_natural_ood.npz"
    )

    evaluation = np.load(
        evaluation_path,
        allow_pickle=True,
    )

    natural = np.load(
        natural_path,
        allow_pickle=True,
    )

    print(
        "=" * 72
    )

    print(
        "LughaLab KSL Paper Figure Generator"
    )

    print(
        "=" * 72
    )

    print(
        "Natural OOD samples:",
        len(
            evaluation[
                "ood_5nn"
            ]
        ),
    )

    print(
        "Held-out ID samples:",
        len(
            evaluation[
                "id_5nn"
            ]
        ),
    )

    print(
        "Unseen glosses:",
        len(
            np.unique(
                evaluation[
                    "glosses"
                ]
            )
        ),
    )

    figure_ood_distribution(
        evaluation
    )

    figure_per_gloss(
        evaluation
    )

    figure_softmax_vs_ood(
        evaluation
    )

    figure_coverage_vs_ood(
        evaluation
    )

    figure_closed_set_collapse(
        natural
    )

    figure_knn_comparison(
        evaluation
    )

    print()
    print(
        "Publication figures complete."
    )


if __name__ == "__main__":
    main()