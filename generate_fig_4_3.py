"""Generate fig_4_3_suppression_criteria.png — the decision-flow funnel
showing how the 5,040 EAV test trials are filtered through the three
suppression-event criteria to arrive at the 401 events that populate
the suppression matrix.

Styled in the pale + light-blue + orange palette that matches fig_6_1,
fig_4_2_attention_arch, and the poster diagrams.

Saves to LateX/images/fig_4_3_suppression_criteria.png.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Polygon

OUT_DIR = Path(__file__).resolve().parent / "LateX" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "fig_4_3_suppression_criteria.png"

# Palette
BG_PALE      = "#E0E0E0"
PANEL_WHITE  = "#FFFFFF"
LIGHT_BLUE   = "#5DADE2"
LB_DEEP      = "#2E86C1"
LB_PALE      = "#AED6F1"
LB_SOFT      = "#D6EAF8"
ORANGE       = "#FF9933"
OR_DEEP      = "#E67E22"
OR_SOFT      = "#FCE4C2"
DARK_TEXT    = "#202020"
GREY         = "#666666"


def trial_box(ax, cx, cy, w, h, count, pct_label, sublabel, *,
              fc=LB_SOFT, ec=LB_DEEP, lw=2.0):
    """A trial-count box: count + percentage on top line, sublabel below."""
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.18",
        fc=fc, ec=ec, lw=lw)
    ax.add_patch(box)
    # Main count + percentage
    ax.text(cx, cy + h / 2 - 0.55,
            f"{count}", ha="center", va="center",
            fontsize=20, fontweight="bold", color=DARK_TEXT)
    ax.text(cx + 2.4, cy + h / 2 - 0.55,
            f"({pct_label})", ha="left", va="center",
            fontsize=12, color=GREY)
    # Subtitle line
    ax.text(cx, cy + h / 2 - 1.25, sublabel,
            ha="center", va="center",
            fontsize=11, color=DARK_TEXT)


def filter_box(ax, cx, cy, w, h, title, criterion):
    """A filter / decision-criterion box (orange)."""
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.16",
        fc=OR_SOFT, ec=OR_DEEP, lw=2.0)
    ax.add_patch(box)
    ax.text(cx, cy + h / 4, title,
            ha="center", va="center",
            fontsize=11, fontweight="bold", color=OR_DEEP)
    ax.text(cx, cy - h / 4, criterion,
            ha="center", va="center",
            fontsize=10.5, color=DARK_TEXT)


def filter_arrow(ax, x_from, y_from, x_to, y_to):
    ax.annotate(
        "", xy=(x_to, y_to), xytext=(x_from, y_from),
        arrowprops=dict(arrowstyle="-|>,head_width=0.4,head_length=0.7",
                        color=LB_DEEP, lw=2.0, shrinkA=0, shrinkB=0))


def main():
    XLIM, YLIM = 20, 26
    fig, ax = plt.subplots(figsize=(11, 14), dpi=200)
    fig.patch.set_facecolor(BG_PALE)
    ax.set_facecolor(BG_PALE)
    ax.set_xlim(0, XLIM)
    ax.set_ylim(0, YLIM)
    ax.set_aspect("auto")
    ax.axis("off")

    cx = XLIM / 2

    # ---- Title ----
    ax.text(cx, 24.8,
            "Suppression Event Selection",
            ha="center", fontsize=18, fontweight="bold", color=LB_DEEP)
    ax.text(cx, 23.9,
            "Three sequential filters applied to all 5,040 EAV test trials",
            ha="center", fontsize=11.5, style="italic", color=OR_DEEP)

    # ---- Total trials ----
    trial_box(ax, cx, 22.0, 11, 1.9,
              "5,040", "100% of test trials",
              "all 42 subjects × 120 trials/subject",
              fc=LB_PALE, ec=LB_DEEP)

    filter_arrow(ax, cx, 21.05, cx, 19.85)

    # ---- Filter 1: AV consensus ----
    filter_box(ax, cx, 19.0, 14, 1.7,
               "FILTER 1   —   Audio-Vision consensus",
               r"pred_audio(s,t)  ==  pred_vision(s,t)")

    filter_arrow(ax, cx, 18.15, cx, 16.95)

    trial_box(ax, cx, 16.0, 11, 1.9,
              "2,426", "48.1% of all trials",
              "trials with external (AV) consensus",
              fc=LB_SOFT, ec=LB_DEEP)

    filter_arrow(ax, cx, 15.05, cx, 13.85)

    # ---- Filter 2: EEG disagreement ----
    filter_box(ax, cx, 13.0, 14, 1.7,
               "FILTER 2   —   EEG disagreement",
               r"pred_eeg(s,t)  ≠  pred_audio(s,t)")

    filter_arrow(ax, cx, 12.15, cx, 10.95)

    trial_box(ax, cx, 10.0, 11, 1.9,
              "1,252", "24.8% of all trials",
              "AV consensus  +  EEG disagrees",
              fc=LB_SOFT, ec=LB_DEEP)

    filter_arrow(ax, cx, 9.05, cx, 7.85)

    # ---- Filter 3: EEG confidence ----
    filter_box(ax, cx, 7.0, 14, 1.7,
               "FILTER 3   —   EEG confidence threshold",
               r"max(softmax(z_eeg(s,t)))  ≥  0.5")

    filter_arrow(ax, cx, 6.15, cx, 4.95)

    # ---- Final: suppression events (highlight) ----
    box = FancyBboxPatch(
        (cx - 6.5, 3.05), 13, 1.9,
        boxstyle="round,pad=0.05,rounding_size=0.18",
        fc=OR_SOFT, ec=OR_DEEP, lw=3.5)
    ax.add_patch(box)
    ax.text(cx, 4.4,
            "401", ha="center", va="center",
            fontsize=22, fontweight="bold", color=DARK_TEXT)
    ax.text(cx + 2.4, 4.4,
            "(8.0% base rate)", ha="left", va="center",
            fontsize=12.5, color=OR_DEEP, fontweight="bold")
    ax.text(cx, 3.55,
            "SUPPRESSION  EVENTS   →   populate the 5×5 matrix",
            ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=DARK_TEXT)

    # ---- Footer (rationale) ----
    ax.text(cx, 1.65,
            "Rationale:  Filter 1 demands a clear external signal.  "
            "Filter 2 demands an internal contradiction.\n"
            "Filter 3 demands that the internal contradiction be confident "
            "(majority probability mass on one class).",
            ha="center", va="center",
            fontsize=10, style="italic", color=GREY,
            linespacing=1.4)

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor=BG_PALE)
    plt.close(fig)
    print(f"wrote {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
