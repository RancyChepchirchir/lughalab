from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]

OUT = (
    ROOT
    / "paper"
    / "figures"
    / "fig01_open_set_architecture.pdf"
)


def box(
    ax,
    x,
    y,
    width,
    height,
    text,
    fontsize=9,
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02",
        linewidth=1.2,
        facecolor="white",
    )
    ax.add_patch(patch)

    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )


def arrow(
    ax,
    x1,
    y1,
    x2,
    y2,
):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 1.2,
        },
    )


def main():
    fig, ax = plt.subplots(
        figsize=(10, 7)
    )

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    box(
        ax,
        3.7,
        9.0,
        2.6,
        0.6,
        "KSL video",
    )

    box(
        ax,
        3.4,
        7.8,
        3.2,
        0.7,
        "Landmark extraction\n"
        "(225 features / frame)",
    )

    box(
        ax,
        3.2,
        6.5,
        3.6,
        0.8,
        "Checkpoint adapter\n"
        "126 hand features + "
        "99 zero features",
    )

    box(
        ax,
        3.2,
        5.2,
        3.6,
        0.8,
        "Temporal adaptation\n"
        "64 frames + checkpoint z-score",
    )

    box(
        ax,
        3.5,
        3.9,
        3.0,
        0.7,
        "Transformer encoder",
    )

    box(
        ax,
        3.6,
        2.7,
        2.8,
        0.6,
        "256-D embedding",
    )

    box(
        ax,
        0.5,
        1.1,
        3.2,
        0.9,
        "Closed-set head\n"
        "4-class softmax",
    )

    box(
        ax,
        6.3,
        1.1,
        3.2,
        0.9,
        "Open-set head\n"
        "5-NN embedding distance",
    )

    box(
        ax,
        0.8,
        0.0,
        2.6,
        0.6,
        "father / hello / is / my",
        fontsize=8,
    )

    box(
        ax,
        6.6,
        0.0,
        2.6,
        0.6,
        "known / unknown",
        fontsize=8,
    )

    arrow(ax, 5.0, 9.0, 5.0, 8.5)
    arrow(ax, 5.0, 7.8, 5.0, 7.3)
    arrow(ax, 5.0, 6.5, 5.0, 6.0)
    arrow(ax, 5.0, 5.2, 5.0, 4.6)
    arrow(ax, 5.0, 3.9, 5.0, 3.3)

    arrow(
        ax,
        4.3,
        2.7,
        2.1,
        2.0,
    )

    arrow(
        ax,
        5.7,
        2.7,
        7.9,
        2.0,
    )

    arrow(
        ax,
        2.1,
        1.1,
        2.1,
        0.6,
    )

    arrow(
        ax,
        7.9,
        1.1,
        7.9,
        0.6,
    )

    ax.text(
        8.0,
        2.35,
        "572 training\nembeddings",
        ha="center",
        va="center",
        fontsize=8,
    )

    ax.set_title(
        "LughaLab KSL inference and "
        "open-set recognition pipeline",
        fontsize=13,
        pad=15,
    )

    OUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        OUT,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("Saved:")
    print(f"  {OUT}")


if __name__ == "__main__":
    main()