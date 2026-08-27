"""Generate fig_6_3_robustness.png and fig_6_4_per_class.png in the
blue + orange palette that matches fig_6_1, fig_4_2 and the poster diagrams.

Both are saved directly to LateX/images/ so chapter 6's \\includegraphics
calls resolve on next compile.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

OUT_DIR = Path(__file__).resolve().parent / "LateX" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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


# ---------------------------------------------------------------------------
# fig_6_3 — Robustness under missing modalities
# ---------------------------------------------------------------------------

def fig_6_3_robustness():
    out = OUT_DIR / "fig_6_3_robustness.png"

    methods = ["Full\n(A + V + E)",
               "AV only\n(no EEG, demo path)",
               "VE only\n(no audio)",
               "AE only\n(no video)"]
    values = [84.7, 81.5, 79.7, 54.0]
    colors = [LIGHT_BLUE, ORANGE, LIGHT_BLUE, LIGHT_BLUE]
    edges = [LB_DEEP, OR_DEEP, LB_DEEP, LB_DEEP]

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200)
    fig.patch.set_facecolor(BG_PALE)
    ax.set_facecolor(BG_PALE)

    x_pos = np.arange(len(methods))
    bars = ax.bar(x_pos, values, color=colors, edgecolor=edges,
                  linewidth=1.7, width=0.62, zorder=3)
    bars[1].set_linewidth(3.5)  # highlight demo path

    # Value labels above bars
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.0,
                f"{v:.1f}%", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=DARK_TEXT)

    # Reference line at dropout-untrained cross-attention baseline
    ax.axhline(80.2, color=GREY, linestyle="--", linewidth=1.4, alpha=0.75)
    ax.text(3.55, 80.65, "cross-attn baseline\n(no dropout) = 80.2%",
            color=GREY, va="bottom", ha="right",
            fontsize=9.5, style="italic")

    # Demo-path annotation
    ax.annotate("DEMO PATH\nzero-EEG inference",
                xy=(1, 81.5), xytext=(1, 35),
                ha="center", fontsize=11, fontweight="bold", color=OR_DEEP,
                arrowprops=dict(arrowstyle="->", color=OR_DEEP, lw=2))

    # X-axis
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods, fontsize=11, color=DARK_TEXT)

    # Y-axis
    ax.set_ylim(0, 100)
    ax.set_ylabel("Mean test accuracy  (%, 42 subjects)",
                  color=LIGHT_BLUE, fontsize=12, fontweight="bold")

    ax.set_title("Robustness to Missing Modalities — Modality-Dropout Trained",
                 color=LIGHT_BLUE, fontsize=13.5, fontweight="bold", pad=12)

    ax.grid(axis="y", color=GREY, alpha=0.25, linewidth=0.7, zorder=1)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(LB_DEEP)
    ax.spines["bottom"].set_color(LB_DEEP)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", colors=DARK_TEXT, labelsize=11)

    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG_PALE)
    plt.close(fig)
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# fig_6_4 — Per-class metrics (confusion matrix + F1 bars side-by-side)
# ---------------------------------------------------------------------------

def fig_6_4_per_class():
    out = OUT_DIR / "fig_6_4_per_class.png"

    labels = ["Neutral", "Sadness", "Anger", "Happiness", "Calmness"]

    confusion = np.array([
        [78, 1,  1,  3, 14],
        [5, 78,  7,  3,  5],
        [2,  5, 87,  4,  1],
        [0,  0,  2, 94,  2],
        [7,  1,  0,  5, 85],
    ], dtype=float)

    f1 = [0.806, 0.837, 0.878, 0.898, 0.813]
    macro_f1 = float(np.mean(f1))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 5.8), dpi=200,
        gridspec_kw={"width_ratios": [1.15, 1.0]}
    )
    fig.patch.set_facecolor(BG_PALE)
    ax1.set_facecolor(BG_PALE)
    ax2.set_facecolor(BG_PALE)

    # --- Confusion matrix heatmap (blue colormap) ---
    cmap = LinearSegmentedColormap.from_list(
        "blue_cmap", [PANEL_WHITE, LB_SOFT, LIGHT_BLUE, LB_DEEP])

    im = ax1.imshow(confusion, cmap=cmap, vmin=0, vmax=100, aspect="equal")

    for i in range(5):
        for j in range(5):
            val = int(confusion[i, j])
            color = "white" if val > 55 else DARK_TEXT
            weight = "bold" if i == j else "normal"
            ax1.text(j, i, str(val), ha="center", va="center",
                     color=color, fontweight=weight, fontsize=13)

    ax1.set_xticks(range(5))
    ax1.set_yticks(range(5))
    ax1.set_xticklabels(labels, fontsize=11, color=DARK_TEXT)
    ax1.set_yticklabels(labels, fontsize=11, color=DARK_TEXT)
    ax1.set_xlabel("Predicted", color=LIGHT_BLUE,
                   fontweight="bold", fontsize=12)
    ax1.set_ylabel("True label", color=LIGHT_BLUE,
                   fontweight="bold", fontsize=12)
    ax1.set_title("Confusion matrix  (row %)",
                  color=LIGHT_BLUE, fontsize=13, fontweight="bold", pad=10)
    ax1.tick_params(axis="both", colors=DARK_TEXT)

    # --- F1 bar chart ---
    colors = [LIGHT_BLUE] * 5
    edges = [LB_DEEP] * 5
    colors[3] = ORANGE  # highlight Happiness (highest F1)
    edges[3] = OR_DEEP

    bars = ax2.barh(labels, f1, color=colors, edgecolor=edges,
                    linewidth=1.6, height=0.62, zorder=3)
    bars[3].set_linewidth(3.5)

    for bar, v in zip(bars, f1):
        ax2.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{v:.3f}", va="center", fontsize=12,
                 color=DARK_TEXT, fontweight="bold")

    ax2.set_xlim(0.7, 1.0)
    ax2.set_xlabel("$F_1$  score", color=ORANGE,
                   fontweight="bold", fontsize=12)
    ax2.set_title(f"Per-class $F_1$   ·   Macro-$F_1$ = {macro_f1:.3f}",
                  color=LIGHT_BLUE, fontsize=13, fontweight="bold", pad=10)
    ax2.invert_yaxis()
    ax2.grid(axis="x", color=GREY, alpha=0.25, linewidth=0.7, zorder=1)
    ax2.set_axisbelow(True)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_color(LB_DEEP)
    ax2.spines["bottom"].set_color(LB_DEEP)
    ax2.spines["left"].set_linewidth(1.2)
    ax2.spines["bottom"].set_linewidth(1.2)
    ax2.tick_params(axis="both", colors=DARK_TEXT, labelsize=11)

    plt.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=BG_PALE)
    plt.close(fig)
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    fig_6_3_robustness()
    fig_6_4_per_class()
