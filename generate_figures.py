"""Generate PNG figures for embedding into the slide deck.

Outputs four files under figures/:
  fig_training.png       — slide 7, per-modality + fusion bar chart
  fig_robustness.png     — slide 9, missing-modality bar chart
  fig_suppression.png    — slide 10, 5x5 suppression heatmap
  fig_per_class.png      — slide 11, confusion heatmap + per-class F1
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# Color palette aligned with generate_slides.py
NAVY       = "#1A3A5C"
BLUE       = "#2E7EB8"
LIGHT_BLUE = "#D9E7F5"
AMBER      = "#C87F12"
GOLD_BG    = "#FDF1E0"
DARK       = "#202020"
GREY       = "#666666"
GREEN      = "#2E8B57"
RED        = "#B01F1F"


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.edgecolor": GREY,
    })


# ---------------------------------------------------------------------------
# Fig 1 — Training Results (replaces slide-7 table)
# ---------------------------------------------------------------------------

def fig_training():
    _style()
    fig, ax = plt.subplots(figsize=(13, 6.5), dpi=180)

    methods = [
        "Audio\n(AST)",
        "Vision\n(ViT)",
        "EEG\n(EEGNet)",
        "Naive late\nfusion",
        "Cross-attn\nfusion",
        "+ Softhard\ndropout",
        "Demo path\n(AV-only)",
    ]
    values = [57.1, 74.6, 43.9, 80.0, 80.2, 84.7, 81.5]
    colors = [LIGHT_BLUE, LIGHT_BLUE, LIGHT_BLUE,
              BLUE, BLUE, AMBER, GREEN]

    bars = ax.bar(methods, values, color=colors,
                  edgecolor=NAVY, linewidth=1.2, width=0.7)
    bars[5].set_edgecolor(AMBER)
    bars[5].set_linewidth(3.0)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.1f}%",
                ha="center", va="bottom", fontweight="bold",
                color=NAVY, fontsize=12)

    # Reference: EAV paper published baselines (dashed grey)
    eav = [36.7, 52.8, 36.7]
    for i, b in enumerate(eav):
        ax.hlines(b, i - 0.35, i + 0.35, colors=GREY,
                  linestyles="--", linewidth=1.7, alpha=0.85)
    ax.text(1.0, 32, "dashed lines = EAV-paper baselines",
            ha="center", va="top", fontsize=9, color=GREY, style="italic")

    # Group separator lines
    for x in (2.5, 4.5, 5.5):
        ax.axvline(x, color=GREY, alpha=0.18, linewidth=0.8)

    # Group annotations
    for cx, label in [(1.0, "Single modality"),
                      (3.5, "Fusion"),
                      (5.0, "+ Dropout"),
                      (6.0, "Demo")]:
        ax.text(cx, 97, label, ha="center", color=GREY,
                fontsize=10, style="italic")

    ax.set_ylabel("Test accuracy (%)", color=DARK, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title("42-subject mean test accuracy by method",
                 color=NAVY, fontsize=14, pad=14)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=10)

    plt.tight_layout()
    fig.savefig(OUT / "fig_training.png", dpi=180, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 2 — Robustness (replaces slide-9 table)
# ---------------------------------------------------------------------------

def fig_robustness():
    _style()
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=180)

    methods = ["Full\n(Audio + Vision + EEG)",
               "AV only\n(no EEG)",
               "VE only\n(no audio)",
               "AE only\n(no video)"]
    values = [84.7, 81.5, 79.7, 54.0]
    colors = [NAVY, GREEN, BLUE, RED]

    bars = ax.bar(methods, values, color=colors,
                  edgecolor=NAVY, linewidth=1.2, width=0.65)
    bars[1].set_edgecolor(AMBER)
    bars[1].set_linewidth(3.0)

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5,
                f"{v:.1f}%", ha="center", va="bottom",
                fontweight="bold", color=NAVY, fontsize=14)

    # Reference: pre-dropout cross-attention baseline
    ax.axhline(80.2, color=GREY, linestyle="--", linewidth=1.5, alpha=0.85)
    ax.text(3.55, 80.5, " cross-attn (no dropout) = 80.2%",
            color=GREY, va="bottom", ha="right", fontsize=9, style="italic")

    # Annotate demo path
    ax.annotate("DEMO PATH\nzero-EEG\ninference",
                xy=(1, 81.5), xytext=(1, 40),
                ha="center", fontsize=11, fontweight="bold", color=AMBER,
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=2))

    ax.set_ylabel("Test accuracy (%)", color=DARK, fontsize=12)
    ax.set_ylim(0, 100)
    ax.set_title("Robustness to missing modalities (modality-dropout trained)",
                 color=NAVY, fontsize=14, pad=14)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=11)

    plt.tight_layout()
    fig.savefig(OUT / "fig_robustness.png", dpi=180, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 3 — Suppression heatmap (replaces slide-10 table)
# ---------------------------------------------------------------------------

def fig_suppression():
    _style()
    fig, ax = plt.subplots(figsize=(8.6, 7.2), dpi=180)

    labels = ["Neutral", "Sadness", "Anger", "Happiness", "Calmness"]
    data = np.array([
        [0,  5,  3,  8, 24],
        [43, 0, 21, 12, 20],
        [5, 13,  0, 65, 10],
        [11, 17, 52, 0, 14],
        [40, 17, 4, 17,  0],
    ], dtype=float)

    cmap = LinearSegmentedColormap.from_list(
        "navy_cmap", ["#FFFFFF", "#D9E7F5", "#2E7EB8", "#1A3A5C"])
    cmap.set_bad(color="#EEEEEE")

    masked = np.ma.masked_where(np.eye(5, dtype=bool), data)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=70, aspect="equal")

    for i in range(5):
        for j in range(5):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center",
                        color=GREY, fontsize=15)
                continue
            val = int(data[i, j])
            color = "white" if val > 35 else NAVY
            weight = "bold" if val >= 40 else "normal"
            ax.text(j, i, str(val), ha="center", va="center",
                    color=color, fontweight=weight, fontsize=14)

    # Highlight dominant cells with rectangles
    from matplotlib.patches import Rectangle
    for (i, j) in [(2, 3), (3, 2), (1, 0), (4, 0)]:
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1,
                               fill=False, edgecolor=AMBER, linewidth=3))

    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Internal — EEG predicts",
                  color=NAVY, fontweight="bold", fontsize=12, labelpad=8)
    ax.set_ylabel("External — AV consensus",
                  color=NAVY, fontweight="bold", fontsize=12, labelpad=8)
    ax.set_title("Suppression matrix: 401 events / 5,040 trials (8.0%)",
                 color=NAVY, fontsize=14, pad=12)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("# suppression events", color=DARK, fontsize=11)

    plt.tight_layout()
    fig.savefig(OUT / "fig_suppression.png", dpi=180, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fig 4 — Confusion heatmap + per-class F1 (replaces slide-11 tables)
# ---------------------------------------------------------------------------

def fig_per_class():
    _style()
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(15, 6.2), dpi=180,
        gridspec_kw={"width_ratios": [1.2, 1.0]}
    )

    labels = ["Neutral", "Sadness", "Anger", "Happiness", "Calmness"]

    # --- Confusion matrix (left) ---
    confusion = np.array([
        [78, 1,   1,   3, 14],
        [5,  78,  7,   3,  5],
        [2,  5,  87,   4,  1],
        [0,  0,   2,  94,  2],
        [7,  1,   0,   5, 85],
    ], dtype=float)

    cmap = LinearSegmentedColormap.from_list(
        "diag_cmap", ["#FFFFFF", "#D9E7F5", "#2E7EB8", "#1A3A5C"])
    im = ax1.imshow(confusion, cmap=cmap, vmin=0, vmax=100, aspect="equal")

    for i in range(5):
        for j in range(5):
            val = int(confusion[i, j])
            color = "white" if val > 50 else NAVY
            weight = "bold" if i == j else "normal"
            ax1.text(j, i, str(val), ha="center", va="center",
                     color=color, fontweight=weight, fontsize=13)

    ax1.set_xticks(range(5))
    ax1.set_yticks(range(5))
    ax1.set_xticklabels(labels, fontsize=11)
    ax1.set_yticklabels(labels, fontsize=11)
    ax1.set_xlabel("Predicted", color=NAVY, fontweight="bold", fontsize=12)
    ax1.set_ylabel("True label", color=NAVY, fontweight="bold", fontsize=12)
    ax1.set_title("Confusion matrix (row %)",
                  color=NAVY, fontsize=13, pad=10)

    # --- Per-class F1 (right) ---
    f1 = [0.806, 0.837, 0.878, 0.898, 0.813]
    colors = [BLUE, BLUE, NAVY, AMBER, BLUE]

    bars = ax2.barh(labels, f1, color=colors, edgecolor=NAVY,
                    linewidth=1.2, height=0.65)
    bars[3].set_edgecolor(AMBER)
    bars[3].set_linewidth(3.0)

    for bar, v in zip(bars, f1):
        ax2.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{v:.3f}", va="center", fontsize=12,
                 color=NAVY, fontweight="bold")

    ax2.set_xlim(0.7, 1.0)
    ax2.set_xlabel("F1 score", color=DARK, fontsize=12)
    ax2.set_title("Per-class F1   ·   Macro-F1 = 0.846",
                  color=NAVY, fontsize=13, pad=10)
    ax2.grid(axis="x", alpha=0.25)
    ax2.set_axisbelow(True)
    ax2.invert_yaxis()
    ax2.tick_params(axis="y", labelsize=11)

    plt.tight_layout()
    fig.savefig(OUT / "fig_per_class.png", dpi=180, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    fig_training()
    fig_robustness()
    fig_suppression()
    fig_per_class()
    print(f"Figures saved to {OUT}/")
    for p in sorted(OUT.glob("*.png")):
        size_kb = p.stat().st_size // 1024
        print(f"  {p.name}  ({size_kb} KB)")


if __name__ == "__main__":
    main()
