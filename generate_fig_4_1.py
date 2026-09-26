"""Generate fig_4_1_trimodal_pipeline.png — the high-level three-stage
trimodal pipeline diagram for Section 4.1.

This figure is the *system-level* view of the framework:
    Stage A  Per-modality encoders   (AST, ViT, EEGNet, trained per subject)
    Stage B  Pre-classifier feature extraction
    Stage C  Trimodal fusion + coherence analysis

Compare with fig_4_2_attention_arch.png, which is the *module-level*
zoom into just the TrimodalAttentionFusion block inside Stage C.

Styling matches the rest of the thesis figures: pale background with
light-blue and orange accents.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT_DIR = Path(__file__).resolve().parent / "LateX_P3" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "fig_4_1_trimodal_pipeline.png"

# Palette (matches generate_fig_6_1.py final styling)
BG_PALE      = "#E0E0E0"     # pale background
PANEL_WHITE  = "#FFFFFF"
LIGHT_BLUE   = "#5DADE2"
LB_DEEP      = "#2E86C1"
LB_PALE      = "#85C1E9"
LB_SOFT      = "#D6EAF8"
ORANGE       = "#FF9933"
OR_DEEP      = "#E67E22"
OR_SOFT      = "#FCE4C2"
DARK_TEXT    = "#202020"
GREY         = "#666666"
LIGHT_GREY   = "#BDBDBD"


def block(ax, cx, cy, w, h, text, *,
          fc=LB_SOFT, ec=LB_DEEP, lw=1.6,
          fontsize=10.5, weight="normal", color=DARK_TEXT, radius=0.12):
    """Rounded rectangle with centred multi-line text."""
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0.04,rounding_size={radius}",
        fc=fc, ec=ec, lw=lw)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=color,
            linespacing=1.3)


def sharp_block(ax, cx, cy, w, h, text, *,
                fc=OR_SOFT, ec=OR_DEEP, lw=2.0,
                fontsize=11, weight="bold", color=DARK_TEXT):
    """Rectangle (input/output emphasis)."""
    rect = Rectangle(
        (cx - w / 2, cy - h / 2), w, h,
        facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=color,
            linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, *, color=LB_DEEP, lw=1.6,
          shape_label=None, label_dy=0):
    """Vertical or diagonal arrow with optional small label."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>,head_width=0.32,head_length=0.6",
                        color=color, lw=lw, shrinkA=0, shrinkB=0))
    if shape_label is not None:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2 + label_dy
        ax.text(mx + 0.25, my, shape_label, ha="left", va="center",
                fontsize=8.5, style="italic", color=GREY)


def stage_label(ax, x, y, text):
    """Side-tab label for a pipeline stage."""
    box = FancyBboxPatch(
        (x - 1.4, y - 0.45), 2.6, 0.9,
        boxstyle="round,pad=0.03,rounding_size=0.18",
        fc=ORANGE, ec=OR_DEEP, lw=1.5)
    ax.add_patch(box)
    ax.text(x - 0.1, y, text, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="white")


def main():
    XLIM, YLIM = 22, 16
    fig, ax = plt.subplots(figsize=(13, 9.5), dpi=180)
    fig.patch.set_facecolor(BG_PALE)
    ax.set_facecolor(BG_PALE)
    ax.set_xlim(0, XLIM)
    ax.set_ylim(0, YLIM)
    ax.set_aspect("auto")
    ax.axis("off")

    # ---- Title ----
    ax.text(XLIM / 2, 15.5, "Three-Stage Trimodal Pipeline",
            ha="center", fontsize=16, fontweight="bold", color=LB_DEEP)
    ax.text(XLIM / 2, 14.9,
            "Per-modality encoders  →  cached features  →  fusion and evaluation",
            ha="center", fontsize=11, style="italic", color=OR_DEEP)

    # ---- EAV dataset header ----
    sharp_block(ax, XLIM / 2, 13.8, 18, 0.85,
                "EAV Dataset   ·   42 participants × 5 emotions   ·   400 five-second trials per participant and modality",
                fc=OR_SOFT, ec=OR_DEEP, fontsize=10.5, weight="bold")

    # ---- Three modality columns ----
    cx_a, cx_v, cx_e = 5.0, 11.0, 17.0
    col_w = 4.0

    # Stage A label
    stage_label(ax, 2.0, 12.3, "STAGE A")
    ax.text(0.7, 11.7, "Per-modality\nencoders",
            ha="left", va="top", fontsize=9.5, color=GREY,
            style="italic")

    # Input boxes (modality raw inputs)
    block(ax, cx_a, 12.3, col_w, 0.85,
          "AUDIO  ·  16 kHz, 5 s",
          fc=LB_SOFT, ec=LB_DEEP, fontsize=10, weight="bold")
    block(ax, cx_v, 12.3, col_w, 0.85,
          "VIDEO  ·  25 frames, 5 s",
          fc=LB_SOFT, ec=LB_DEEP, fontsize=10, weight="bold")
    block(ax, cx_e, 12.3, col_w, 0.85,
          "EEG  ·  30 ch, 500 Hz",
          fc=LB_SOFT, ec=LB_DEEP, fontsize=10, weight="bold")

    for cxn in (cx_a, cx_v, cx_e):
        arrow(ax, cxn, 11.85, cxn, 11.35)

    # Per-modality encoder boxes
    block(ax, cx_a, 10.8, col_w, 1.05,
          "AST\nMIT/ast-finetuned-audioset",
          fc=LIGHT_BLUE, ec=LB_DEEP, fontsize=10.5,
          weight="bold", color="white")
    block(ax, cx_v, 10.8, col_w, 1.05,
          "ViT\ndima806/facial_emotions",
          fc=LIGHT_BLUE, ec=LB_DEEP, fontsize=10.5,
          weight="bold", color="white")
    block(ax, cx_e, 10.8, col_w, 1.05,
          "EEGNet\nF1=8, D=8, F2=64",
          fc=LIGHT_BLUE, ec=LB_DEEP, fontsize=10.5,
          weight="bold", color="white")

    for cxn in (cx_a, cx_v, cx_e):
        arrow(ax, cxn, 10.3, cxn, 9.8)

    # Feature dimensions output
    block(ax, cx_a, 9.4, col_w, 0.7, "768-d audio feature",
          fc=PANEL_WHITE, ec=LB_DEEP, fontsize=9.5)
    block(ax, cx_v, 9.4, col_w, 0.7, "768-d vision feature",
          fc=PANEL_WHITE, ec=LB_DEEP, fontsize=9.5)
    block(ax, cx_e, 9.4, col_w, 0.7, "960-d EEG feature",
          fc=PANEL_WHITE, ec=LB_DEEP, fontsize=9.5)

    # ---- Stage A artefact (per-modality logits cache) ----
    # (small dashed annotation noting that logits are also saved)
    ax.text(20.3, 9.4, "+\nper-trial\ntest logits\nsaved",
            ha="center", va="center", fontsize=8, style="italic",
            color=GREY)

    # ---- Down arrows into Stage B ----
    for cxn in (cx_a, cx_v, cx_e):
        arrow(ax, cxn, 9.05, cxn, 8.5)

    # ---- Stage B band ----
    stage_label(ax, 2.0, 8.05, "STAGE B")
    ax.text(0.7, 7.5, "Pre-classifier\nfeature\nextraction",
            ha="left", va="top", fontsize=9.5, color=GREY,
            style="italic")

    block(ax, XLIM / 2, 8.05, 18, 0.85,
          "Strip per-modality classification heads   ·   "
          "cache penultimate-layer features per (subject, modality)",
          fc=LB_PALE, ec=LB_DEEP, fontsize=10.5, weight="bold")

    # ---- Converging arrows into Stage C fusion ----
    for cxn in (cx_a, cx_v, cx_e):
        arrow(ax, cxn, 7.6, XLIM / 2, 6.7)

    # ---- Stage C — Fusion ----
    stage_label(ax, 2.0, 6.0, "STAGE C")
    ax.text(0.7, 5.55, "Fusion +\nevaluation",
            ha="left", va="top", fontsize=9.5, color=GREY,
            style="italic")

    block(ax, XLIM / 2, 6.0, 18, 1.4,
          "TrimodalAttentionFusion  ·  2.29 M params  (also: naive averaging, concat-MLP)\n"
          "Linear projections → modality embed → softhard dropout (training only)\n"
          "→ Transformer encoder × 2 → mean-pool → MLP head",
          fc=LIGHT_BLUE, ec=LB_DEEP, fontsize=10.5,
          weight="bold", color="white")

    # Note: pointer to fig 4.2 zoom
    ax.text(20.5, 6.0, "→  see\nFigure 4.2\nfor module\ndetail",
            ha="center", va="center", fontsize=8.5,
            style="italic", color=OR_DEEP, fontweight="bold")

    # ---- Down arrows from fusion into two heads ----
    arrow(ax, XLIM / 2, 5.3, 7.0, 4.4)
    arrow(ax, XLIM / 2, 5.3, 15.0, 4.4)

    # ---- Two heads ----
    block(ax, 7.0, 3.85, 6.5, 1.1,
          "CLASSIFICATION  HEAD\n5-way softmax over emotion classes",
          fc=PANEL_WHITE, ec=LB_DEEP, fontsize=10, weight="bold")

    block(ax, 15.0, 3.85, 6.5, 1.1,
          "COHERENCE  AUDIT\nsuppression matrix vs permutation null",
          fc=PANEL_WHITE, ec=OR_DEEP, fontsize=10, weight="bold")

    arrow(ax, 7.0, 3.3, 7.0, 2.6)
    arrow(ax, 15.0, 3.3, 15.0, 2.6, color=OR_DEEP)

    # ---- Outputs ----
    sharp_block(ax, 7.0, 2.05, 7.0, 1.05,
                "EVALUATION  PROTOCOLS\n"
                "within-subject  ·  cross-subject  ·  calibrated  ·  few-shot",
                fc=LB_SOFT, ec=LB_DEEP, fontsize=9.5, weight="bold")

    sharp_block(ax, 15.0, 2.05, 7.0, 1.05,
                "AUDIT  RESULT\n"
                "401 events  ·  reproduces EEG errors (r = 0.989)",
                fc=OR_SOFT, ec=OR_DEEP, fontsize=9.5, weight="bold")

    # ---- Footer ----
    ax.text(XLIM / 2, 0.7,
            "Each stage writes cached artefacts to disk before the next stage "
            "begins   ·   pipeline is resume-safe across Colab sessions",
            ha="center", va="center", fontsize=9, style="italic", color=GREY)

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight",
                facecolor=BG_PALE)
    plt.close(fig)
    print(f"wrote {OUT_PATH}")
    print(f"  size: {OUT_PATH.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
