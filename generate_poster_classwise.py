"""Generate poster replacement charts for the Classwise Performance section.

Outputs (300 DPI, blue + orange palette to match the rest of the poster):
    figures/poster_classwise_metrics.png   — grouped bar chart of precision /
                                             recall / F1 across 5 emotions
    figures/poster_confusion_matrix.png    — confusion matrix heatmap
                                             (row-percent terms)

Drop both into Column 3 of the poster to replace the existing two tables.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Palette
BG_PALE     = "#E0E0E0"
PANEL_WHITE = "#FFFFFF"
LIGHT_BLUE  = "#5DADE2"
LB_DEEP     = "#2E86C1"
LB_PALE     = "#AED6F1"
LB_SOFT     = "#D6EAF8"
ORANGE      = "#FF9933"
OR_DEEP     = "#E67E22"
OR_SOFT     = "#FCE4C2"
DARK_TEXT   = "#202020"
GREY        = "#666666"


# Per-class metrics (modality-dropout-trained model, 5,040 test trials)
LABELS = ["Neutral", "Sadness", "Anger", "Happiness", "Calmness"]
PRECISION = [0.831, 0.902, 0.885, 0.852, 0.778]
RECALL    = [0.783, 0.782, 0.871, 0.949, 0.850]
F1        = [0.806, 0.837, 0.878, 0.898, 0.813]

CONFUSION = np.array([
    [78, 1,  1,  3, 14],
    [5, 78,  7,  3,  5],
    [2,  5, 87,  4,  1],
    [0,  0,  2, 94,  2],
    [7,  1,  0,  5, 85],
], dtype=float)


# ---------------------------------------------------------------------------
# Chart 1: Classwise metrics grouped bar chart
# ---------------------------------------------------------------------------

def fig_classwise_metrics():
    out = OUT / "poster_classwise_metrics.png"

    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=300)
    fig.patch.set_facecolor(BG_PALE)
    ax.set_facecolor(BG_PALE)

    x = np.arange(len(LABELS))
    width = 0.26

    # Three bars per emotion: precision (pale blue), recall (light blue), F1 (orange)
    bars_p = ax.bar(x - width, PRECISION, width,
                    color=LB_PALE, edgecolor=LB_DEEP, linewidth=1.4,
                    label="Precision", zorder=3)
    bars_r = ax.bar(x,         RECALL,    width,
                    color=LIGHT_BLUE, edgecolor=LB_DEEP, linewidth=1.4,
                    label="Recall", zorder=3)
    bars_f = ax.bar(x + width, F1,        width,
                    color=ORANGE, edgecolor=OR_DEEP, linewidth=1.8,
                    label="F$_1$  (headline)", zorder=3)

    # Value labels above each bar
    for bars, values in [(bars_p, PRECISION), (bars_r, RECALL),
                         (bars_f, F1)]:
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.008,
                    f"{v:.2f}", ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color=DARK_TEXT)

    # Macro-F1 reference line (thinking dashed orange)
    macro_f1 = float(np.mean(F1))
    ax.axhline(macro_f1, color=OR_DEEP, linestyle="--", linewidth=1.5,
               alpha=0.7, zorder=1)
    ax.text(4.45, macro_f1 + 0.003,
            f"macro-F$_1$ = {macro_f1:.3f}",
            ha="right", va="bottom", fontsize=10,
            fontweight="bold", color=OR_DEEP, style="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=12, color=DARK_TEXT)
    ax.set_ylim(0.70, 1.00)
    ax.set_yticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00])
    ax.set_ylabel("Score", color=LIGHT_BLUE, fontsize=12,
                  fontweight="bold")

    ax.set_title(
        "Per-Class Performance Metrics   "
        "·   modality-dropout-trained model   ·   5,040 test trials",
        color=LIGHT_BLUE, fontsize=13, fontweight="bold", pad=12)

    ax.grid(axis="y", color=GREY, alpha=0.25, linewidth=0.7, zorder=1)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(LB_DEEP)
    ax.spines["bottom"].set_color(LB_DEEP)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", colors=DARK_TEXT, labelsize=11)

    leg = ax.legend(loc="upper left", framealpha=0.95, fontsize=11,
                    facecolor=PANEL_WHITE, edgecolor=LB_DEEP, ncol=3)
    for t in leg.get_texts():
        t.set_color(DARK_TEXT)

    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG_PALE)
    plt.close(fig)
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# Chart 2: Confusion matrix heatmap
# ---------------------------------------------------------------------------

def fig_confusion_matrix():
    out = OUT / "poster_confusion_matrix.png"

    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=300)
    fig.patch.set_facecolor(BG_PALE)
    ax.set_facecolor(BG_PALE)

    cmap = LinearSegmentedColormap.from_list(
        "blue_cmap", [PANEL_WHITE, LB_SOFT, LIGHT_BLUE, LB_DEEP])

    im = ax.imshow(CONFUSION, cmap=cmap, vmin=0, vmax=100, aspect="equal")

    # Annotate every cell, with diagonal bolded and bright text on dark
    for i in range(5):
        for j in range(5):
            val = int(CONFUSION[i, j])
            color = "white" if val > 55 else DARK_TEXT
            weight = "bold" if i == j else "normal"
            size = 15 if i == j else 13
            ax.text(j, i, f"{val}%", ha="center", va="center",
                    color=color, fontweight=weight, fontsize=size)

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(LABELS, fontsize=12, color=DARK_TEXT)
    ax.set_yticklabels(LABELS, fontsize=12, color=DARK_TEXT)
    ax.set_xlabel("Predicted label", color=ORANGE,
                  fontsize=13, fontweight="bold", labelpad=10)
    ax.set_ylabel("True label", color=LIGHT_BLUE,
                  fontsize=13, fontweight="bold", labelpad=10)
    ax.set_title(
        "Confusion Matrix   ·   row-percent terms   "
        "·   modality-dropout-trained model",
        color=LIGHT_BLUE, fontsize=13, fontweight="bold", pad=14)

    ax.tick_params(axis="both", colors=DARK_TEXT)

    # Colour bar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05,
                        ticks=[0, 25, 50, 75, 100])
    cbar.ax.set_yticklabels([f"{v}%" for v in [0, 25, 50, 75, 100]],
                            color=DARK_TEXT)
    cbar.outline.set_edgecolor(LB_DEEP)
    cbar.outline.set_linewidth(1.0)

    plt.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG_PALE)
    plt.close(fig)
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    fig_classwise_metrics()
    fig_confusion_matrix()
