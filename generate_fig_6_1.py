"""Generate fig_6_1_fusion_accuracy_distribution.png — the per-subject
cross-attention fusion accuracy distribution figure for Section 6.3.

Uses the real per-subject accuracies from results/day5_fusion.csv
(42-subject rollout). The plot is a histogram of the 42 values overlaid
with a normal-distribution density at the empirical mean and std, with
the mean line, mu +/- sigma band, and the names of the outlier subjects
(below 0.70 and above 0.95) annotated.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ---- Constants -----------------------------------------------------------
OUT_DIR = Path(__file__).resolve().parent / "LateX" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "fig_6_1_fusion_accuracy_distribution.png"

# Real per-subject test accuracies from results/day5_fusion.csv
# Index = subject ID (1-42), value = test accuracy
SUBJECT_ACC = {
     1: 0.725000,   2: 0.850000,   3: 0.900000,   4: 0.916667,
     5: 0.783333,   6: 0.891667,   7: 0.866667,   8: 0.683333,
     9: 0.800000,  10: 0.791667,  11: 0.750000,  12: 0.600000,
    13: 0.866667,  14: 0.666667,  15: 0.816667,  16: 0.608333,
    17: 0.916667,  18: 0.783333,  19: 0.783333,  20: 0.958333,
    21: 0.833333,  22: 0.825000,  23: 0.733333,  24: 0.825000,
    25: 0.808333,  26: 0.850000,  27: 0.883333,  28: 0.825000,
    29: 0.775000,  30: 0.766667,  31: 0.783333,  32: 0.708333,
    33: 0.883333,  34: 0.708333,  35: 0.808333,  36: 0.833333,
    37: 0.733333,  38: 0.841667,  39: 0.816667,  40: 0.675000,
    41: 0.958333,  42: 0.841667,
}
LOW_THRESHOLD = 0.70
HIGH_THRESHOLD = 0.95

# Palette — dark theme with light-blue + orange accents
BG_BLACK    = "#000000"     # pure-black figure background
PANEL_DARK  = "#0D0D0D"     # near-black panel for callout box fills
LIGHT_BLUE  = "#5DADE2"     # primary accent
LB_DEEP     = "#2E86C1"     # deeper blue (edges, spines)
LB_PALE     = "#85C1E9"     # paler tint
ORANGE      = "#FF9933"     # secondary accent
OR_DEEP     = "#E67E22"     # deeper orange
WHITE       = "#FFFFFF"
PALE        = "#E0E0E0"     # general light text (tick labels, footer)
GRID_DARK   = "#262626"     # subtle dark-grey grid


def make_figure(items: list[tuple[int, float]]) -> None:
    """Render the histogram + density figure from the real per-subject
    accuracies in `items` (list of (subject_id, accuracy) tuples).

    Dark theme: pure-black background with light-blue and orange accents.
    """
    accuracies = np.array([a for _, a in items])
    mean = float(accuracies.mean())
    std = float(accuracies.std(ddof=0))

    low_items = sorted([it for it in items if it[1] < LOW_THRESHOLD],
                       key=lambda it: it[1])
    high_items = sorted([it for it in items if it[1] >= HIGH_THRESHOLD],
                        key=lambda it: it[1])

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=180)

    # Dark backgrounds
    fig.patch.set_facecolor(PALE)
    ax.set_facecolor(PALE)

    # --- Histogram (light-blue bars) ---
    n_bins = 14
    bin_edges = np.linspace(0.55, 1.00, n_bins + 1)
    counts, _edges, _bars = ax.hist(
        accuracies, bins=bin_edges,
        color=LIGHT_BLUE, edgecolor=LB_DEEP, linewidth=1.3,
        alpha=0.85, zorder=3,
        label=f"{len(accuracies)} per-subject accuracies",
    )

    # --- Normal density overlay (ORANGE curve) ---
    xs = np.linspace(0.55, 1.00, 400)
    bin_w = bin_edges[1] - bin_edges[0]
    density = (
        (1 / (std * np.sqrt(2 * np.pi)))
        * np.exp(-0.5 * ((xs - mean) / std) ** 2)
    )
    density_scaled = density * len(accuracies) * bin_w
    ax.plot(xs, density_scaled,
            color=ORANGE, linewidth=2.4, alpha=0.95,
            label=rf"Normal($\mu$={mean:.3f}, $\sigma$={std:.3f})",
            zorder=4)

    # --- Mean line and +/- 1 sigma band ---
    ax.axvspan(mean - std, mean + std,
               alpha=0.13, color=LIGHT_BLUE, zorder=2,
               label=r"$\mu \pm \sigma$")
    ax.axvline(mean, color=WHITE, linestyle="--",
               linewidth=1.6, zorder=5, alpha=0.85)
    y_top = max(counts) + 2.0
    ax.text(mean, y_top * 0.95, f"$\\mu = {mean:.3f}$",
            ha="center", va="top",
            fontsize=11, fontweight="bold", color=ORANGE,
            bbox=dict(boxstyle="round,pad=0.28",
                      fc=PANEL_DARK, ec=ORANGE, lw=1.4))

    # --- Annotate low outliers (LIGHT BLUE text) ---
    low_ids = ", ".join(f"sub{s:02d}" for s, _ in low_items)
    low_label = (
        f"{len(low_items)} subjects below {LOW_THRESHOLD:.2f}\n"
        f"{low_ids}"
    )
    ax.annotate(low_label,
                xy=(low_items[-1][1], 1.2),
                xytext=(0.57, y_top * 0.55),
                fontsize=9.5, color=LIGHT_BLUE, fontweight="bold",
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.4",
                          fc=PANEL_DARK, ec=LIGHT_BLUE, lw=1.4),
                arrowprops=dict(arrowstyle="->",
                                color=LIGHT_BLUE, lw=1.6))

    # --- Annotate high outliers (ORANGE text) ---
    high_ids = ", ".join(f"sub{s:02d}" for s, _ in high_items)
    high_label = (
        f"{len(high_items)} subjects above {HIGH_THRESHOLD:.2f}\n"
        f"{high_ids}"
    )
    ax.annotate(high_label,
                xy=(high_items[0][1], 1.2),
                xytext=(0.86, y_top * 0.55),
                fontsize=9.5, color=ORANGE, fontweight="bold",
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.4",
                          fc=PANEL_DARK, ec=ORANGE, lw=1.4),
                arrowprops=dict(arrowstyle="->",
                                color=ORANGE, lw=1.6))

    # --- Axes and styling ---
    ax.set_xlim(0.55, 1.00)
    ax.set_ylim(0, y_top + 1.0)

    # X-axis label in ORANGE, Y-axis label in LIGHT BLUE
    ax.set_xlabel("Per-subject test accuracy",
                  color=ORANGE, fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of subjects",
                  color=LIGHT_BLUE, fontsize=12, fontweight="bold")

    # Title in LIGHT BLUE
    ax.set_title(
        "Distribution of per-subject test accuracy   "
        "(cross-attention fusion, 42 subjects)",
        color=LIGHT_BLUE, fontsize=13, fontweight="bold", pad=12)

    # Tick labels in pale grey for contrast on black
    ax.tick_params(axis="both", colors=PALE, labelsize=10)

    # Grid: subtle dark grey
    ax.grid(axis="y", color=GRID_DARK, linewidth=0.8,
            alpha=0.7, zorder=1)
    ax.set_axisbelow(True)

    # Spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(LB_DEEP)
    ax.spines["bottom"].set_color(LB_DEEP)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)

    # --- Legend (dark frame, light text) ---
    leg = ax.legend(loc="upper left", framealpha=0.95, fontsize=10,
                    facecolor=PANEL_DARK, edgecolor=LB_DEEP)
    for text in leg.get_texts():
        text.set_color(PALE)

    # --- Footer note (orange italic, so the second accent color
    # appears below the plot frame too) ---
    fig.text(0.5, -0.04,
             "Source: results/day5_fusion.csv  (42-subject rollout, "
             "TrimodalAttentionFusion).",
             ha="center", va="top", fontsize=8.5,
             color=ORANGE, style="italic")

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight",
                facecolor=BG_BLACK)
    plt.close(fig)


def main() -> None:
    items = sorted(SUBJECT_ACC.items())
    make_figure(items)
    accs = np.array([a for _, a in items])
    print(f"wrote {OUT_PATH}")
    print(f"  size: {OUT_PATH.stat().st_size // 1024} KB")
    print(f"  n:    {len(accs)}")
    print(f"  mean: {accs.mean():.4f}")
    print(f"  std:  {accs.std(ddof=0):.4f}")
    print(f"  min:  {accs.min():.4f}   max: {accs.max():.4f}")
    print(f"  below {LOW_THRESHOLD}: "
          f"{sorted(s for s, a in items if a < LOW_THRESHOLD)}")
    print(f"  above {HIGH_THRESHOLD}: "
          f"{sorted(s for s, a in items if a >= HIGH_THRESHOLD)}")


if __name__ == "__main__":
    main()
