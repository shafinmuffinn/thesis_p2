"""Generate two diagram PNGs for the poster, to replace text tables.

Outputs (300 DPI for A1 poster print):
    figures/poster_training_results.png         8-method horizontal bar chart
    figures/poster_prior_fusion_comparison.png  Prior fusion methods on EAV
                                                vs this work

Styling matches fig_6_1_fusion_accuracy_distribution.png: pale background,
light-blue and orange accents, dark text.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# Palette (matches the fig_6_1 final styling)
BG_PALE      = "#E0E0E0"
PANEL_WHITE  = "#FFFFFF"
LIGHT_BLUE   = "#5DADE2"
LB_DEEP      = "#2E86C1"
LB_PALE      = "#AED6F1"
LB_SOFT      = "#D6EAF8"
ORANGE       = "#FF9933"
OR_DEEP      = "#E67E22"
DARK_TEXT    = "#202020"
GREY         = "#666666"
GRID_LIGHT   = "#BDBDBD"


# ---------------------------------------------------------------------------
# Diagram 1: Training results (8 methods)
# ---------------------------------------------------------------------------

def fig_training_results():
    """Horizontal bar chart, grouped: per-modality / fusion / final system."""
    methods = [
        # (label, accuracy %, category)
        ("Audio only (AST)",                  57.1, "per-modality"),
        ("Vision only (ViT)",                 74.6, "per-modality"),
        ("EEG only (EEGNet)",                 43.9, "per-modality"),
        ("Naive late fusion (mean-softmax)",  77.5, "fusion"),
        ("Cross-attention fusion",            80.2, "fusion"),
        ("Concat-MLP fusion",                 81.7, "fusion"),
        ("+ Softhard modality dropout",       84.7, "headline"),
        ("Demo path (AV-only, EEG=0)",        81.5, "headline"),
    ]

    cat_colors = {
        "per-modality": LB_PALE,
        "fusion":       LIGHT_BLUE,
        "headline":     ORANGE,
    }
    cat_edges = {
        "per-modality": LB_DEEP,
        "fusion":       LB_DEEP,
        "headline":     OR_DEEP,
    }

    fig, ax = plt.subplots(figsize=(13, 6.2), dpi=300)
    fig.patch.set_facecolor(BG_PALE)
    ax.set_facecolor(BG_PALE)

    y_pos = np.arange(len(methods))
    labels = [m[0] for m in methods]
    values = [m[1] for m in methods]
    cats = [m[2] for m in methods]

    colors = [cat_colors[c] for c in cats]
    edges = [cat_edges[c] for c in cats]

    bars = ax.barh(y_pos, values, color=colors, edgecolor=edges,
                   linewidth=1.6, height=0.72, zorder=3)
    bars[6].set_linewidth(3.5)  # highlight headline bar (modality dropout)

    # Value labels at bar ends
    for bar, v in zip(bars, values):
        ax.text(v + 1.0, bar.get_y() + bar.get_height() / 2,
                f"{v:.1f}%", va="center", ha="left",
                fontsize=13, fontweight="bold", color=DARK_TEXT)

    # Chance line at 20%
    ax.axvline(20, color=GREY, linestyle=":", linewidth=1.3, alpha=0.7,
               zorder=1)
    ax.text(20.6, len(methods) - 0.45, "chance = 20%",
            ha="left", va="center", fontsize=9.5, color=GREY, style="italic")

    # Y-axis labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12, color=DARK_TEXT)
    ax.invert_yaxis()

    # X-axis
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Mean test accuracy  (%, 42 subjects)",
                  color=ORANGE, fontsize=12.5, fontweight="bold", labelpad=8)

    # Title
    ax.set_title("Training Results  —  42-Subject Mean Accuracy by Method",
                 color=LIGHT_BLUE, fontsize=15, fontweight="bold", pad=14)

    # Grid
    ax.grid(axis="x", color=GREY, alpha=0.25, linewidth=0.7, zorder=1)
    ax.set_axisbelow(True)

    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(LB_DEEP)
    ax.spines["bottom"].set_color(LB_DEEP)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", colors=DARK_TEXT, labelsize=11)

    # Legend — placed at lower centre below the chart to avoid overlap
    # with the bottom-bar value labels.
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=LB_PALE, ec=LB_DEEP, lw=1.5,
                      label="Per-modality"),
        plt.Rectangle((0, 0), 1, 1, fc=LIGHT_BLUE, ec=LB_DEEP, lw=1.5,
                      label="Learned fusion baselines"),
        plt.Rectangle((0, 0), 1, 1, fc=ORANGE, ec=OR_DEEP, lw=3,
                      label="Final system / demo path"),
    ]
    leg = ax.legend(handles=handles, loc="upper center",
                    bbox_to_anchor=(0.5, -0.13), ncol=3,
                    framealpha=0.95, fontsize=10.5,
                    facecolor=PANEL_WHITE, edgecolor=LB_DEEP)
    for txt in leg.get_texts():
        txt.set_color(DARK_TEXT)

    plt.tight_layout()
    out = OUT / "poster_training_results.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG_PALE)
    plt.close(fig)
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")


# ---------------------------------------------------------------------------
# Diagram 2: Prior fusion methods on EAV vs this work
# ---------------------------------------------------------------------------

def fig_prior_fusion_comparison():
    """Vertical bar chart of fusion accuracy across 5 methods with year tags."""
    methods = [
        # (label, year, accuracy, category)
        ("AMERL\n(Yin et al.)",          2024, 70.86, "prior"),
        ("Hyper-MML\n(Kang et al.)",     2025, 76.65, "prior"),
        ("EEG-MoCE",                     2026, 75.88, "prior"),
        ("Ours\n(cross-attention)",      2026, 80.20, "ours"),
        ("Ours\n(+ modality dropout)",   2026, 84.70, "ours-headline"),
    ]

    cat_colors = {
        "prior":         LB_PALE,
        "ours":          LIGHT_BLUE,
        "ours-headline": ORANGE,
    }
    cat_edges = {
        "prior":         LB_DEEP,
        "ours":          LB_DEEP,
        "ours-headline": OR_DEEP,
    }

    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    fig.patch.set_facecolor(BG_PALE)
    ax.set_facecolor(BG_PALE)

    x_pos = np.arange(len(methods))
    labels = [m[0] for m in methods]
    years = [m[1] for m in methods]
    values = [m[2] for m in methods]
    cats = [m[3] for m in methods]

    colors = [cat_colors[c] for c in cats]
    edges = [cat_edges[c] for c in cats]

    bars = ax.bar(x_pos, values, color=colors, edgecolor=edges,
                  linewidth=1.6, width=0.6, zorder=3)
    bars[4].set_linewidth(3.5)  # highlight headline

    # Value labels above bars
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.7,
                f"{v:.2f}%", ha="center", va="bottom",
                fontsize=13, fontweight="bold", color=DARK_TEXT)

    # Reference line at prior SOTA (Hyper-MML 76.65)
    ax.axhline(76.65, color=GREY, linestyle="--", linewidth=1.4, alpha=0.7)
    ax.text(0.0, 76.65 + 0.4, "  prior SOTA  (Hyper-MML, 76.65%)",
            color=GREY, fontsize=10, style="italic", va="bottom")

    # X-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11, color=DARK_TEXT)

    # Year row below the method names
    for i, year in enumerate(years):
        ax.text(i, 60.5, str(year), ha="center", va="top",
                fontsize=10, style="italic", color=GREY)

    # Y-axis
    ax.set_ylim(60, 95)
    ax.set_ylabel("Fusion accuracy  (%, 42 subjects)",
                  color=LIGHT_BLUE, fontsize=12.5, fontweight="bold",
                  labelpad=8)

    # Title
    ax.set_title("Prior Trimodal Fusion Methods on EAV  —  This Work Sets New SOTA",
                 color=LIGHT_BLUE, fontsize=14.5, fontweight="bold", pad=14)

    # SOTA arrow + annotation
    ax.annotate(
        "New SOTA\n+8.05 pp",
        xy=(4, 84.7), xytext=(4, 92.5),
        ha="center", fontsize=11.5, fontweight="bold", color=ORANGE,
        arrowprops=dict(arrowstyle="->", color=ORANGE, lw=2)
    )

    # Grid
    ax.grid(axis="y", color=GREY, alpha=0.25, linewidth=0.7, zorder=1)
    ax.set_axisbelow(True)

    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(LB_DEEP)
    ax.spines["bottom"].set_color(LB_DEEP)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(axis="both", colors=DARK_TEXT, labelsize=11)

    # Legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=LB_PALE, ec=LB_DEEP, lw=1.5,
                      label="Prior published work on EAV"),
        plt.Rectangle((0, 0), 1, 1, fc=LIGHT_BLUE, ec=LB_DEEP, lw=1.5,
                      label="This thesis (cross-attention)"),
        plt.Rectangle((0, 0), 1, 1, fc=ORANGE, ec=OR_DEEP, lw=3,
                      label="This thesis (+ modality dropout, new SOTA)"),
    ]
    leg = ax.legend(handles=handles, loc="upper left",
                    framealpha=0.95, fontsize=10,
                    facecolor=PANEL_WHITE, edgecolor=LB_DEEP)
    for txt in leg.get_texts():
        txt.set_color(DARK_TEXT)

    plt.tight_layout()
    out = OUT / "poster_prior_fusion_comparison.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG_PALE)
    plt.close(fig)
    print(f"wrote {out}  ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    fig_training_results()
    fig_prior_fusion_comparison()
