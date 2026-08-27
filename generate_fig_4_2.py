"""Generate fig_4_2_attention_arch.png — the TrimodalAttentionFusion module
internals diagram, restyled in the pale background + light-blue + orange
palette that matches fig_6_1 and the poster diagrams.

Saves to LateX/images/fig_4_2_attention_arch.png (overwrites the old
navy/amber version).
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT_DIR = Path(__file__).resolve().parent / "LateX" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "fig_4_2_attention_arch.png"

# Palette (matches fig_6_1 + poster diagrams)
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
GREY_LIGHT   = "#BDBDBD"


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def block(ax, cx, cy, w, h, text, *,
          fc=LB_SOFT, ec=LB_DEEP, lw=1.6,
          fontsize=10.5, weight="normal", color=DARK_TEXT, radius=0.12):
    """Rounded rectangle centered at (cx, cy) with centred multi-line text."""
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
    """Sharp-corner rectangle (used for inputs / outputs)."""
    rect = Rectangle(
        (cx - w / 2, cy - h / 2), w, h,
        facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=color,
            linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, *, color=LB_DEEP, lw=1.5):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>,head_width=0.28,head_length=0.55",
                        color=color, lw=lw, shrinkA=0, shrinkB=0))


def stage_tab(ax, x, y, text, *, fc=ORANGE, ec=OR_DEEP):
    """Side tab indicating the pipeline stage."""
    box = FancyBboxPatch(
        (x - 1.6, y - 0.45), 3.0, 0.9,
        boxstyle="round,pad=0.03,rounding_size=0.18",
        fc=fc, ec=ec, lw=1.5)
    ax.add_patch(box)
    ax.text(x - 0.1, y, text, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="white")


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------

def main():
    XLIM, YLIM = 30, 28
    fig, ax = plt.subplots(figsize=(14, 12), dpi=200)
    fig.patch.set_facecolor(BG_PALE)
    ax.set_facecolor(BG_PALE)
    ax.set_xlim(0, XLIM)
    ax.set_ylim(0, YLIM)
    ax.set_aspect("auto")
    ax.axis("off")

    # ---- Title ----
    ax.text(XLIM / 2, 27.0,
            "TrimodalAttentionFusion  —  Self-Attention over Modality Tokens",
            ha="center", fontsize=17, fontweight="bold", color=LB_DEEP)
    ax.text(XLIM / 2, 26.2,
            "Trimodal fusion module   ·   2.28 M trainable parameters",
            ha="center", fontsize=11.5, style="italic", color=OR_DEEP)

    # Three column centres for the parallel input streams
    cx_a, cx_v, cx_e = 7.0, 15.0, 23.0
    cx_main = 15.0  # main centre column
    col_w = 6.0

    # ============= INPUTS (Stage 1 inputs from encoders) =============
    y = 24.6
    for cxn, label in [
        (cx_a, "x_audio   from AST\n(768,)"),
        (cx_v, "x_vision   from ViT\n(768,)"),
        (cx_e, "x_eeg   from EEGNet\n(960,)"),
    ]:
        sharp_block(ax, cxn, y, col_w, 1.0,
                    label, fc=OR_SOFT, ec=OR_DEEP, lw=2.0,
                    fontsize=11, weight="bold", color=DARK_TEXT)
        arrow(ax, cxn, y - 0.55, cxn, y - 1.3)

    # ============= STAGE 1 — Projection =============
    y = 22.6
    stage_tab(ax, 2.0, y, "STAGE 1")
    ax.text(0.5, y - 0.75, "Projection",
            ha="left", va="center", fontsize=9.5,
            color=GREY, style="italic")
    for cxn, lbl in [
        (cx_a, "Linear ( 768  →  256 )"),
        (cx_v, "Linear ( 768  →  256 )"),
        (cx_e, "Linear ( 960  →  256 )"),
    ]:
        block(ax, cxn, y, col_w, 0.9, lbl,
              fc=LIGHT_BLUE, ec=LB_DEEP, lw=1.6,
              fontsize=11, weight="bold", color=DARK_TEXT)
        arrow(ax, cxn, y - 0.5, cxn, y - 1.25)

    # ============= STAGE 2 — Modality embedding =============
    y = 20.6
    stage_tab(ax, 2.0, y, "STAGE 2")
    ax.text(0.5, y - 0.85, "Modality\nembedding",
            ha="left", va="center", fontsize=9.5,
            color=GREY, style="italic")
    for cxn, idx in [(cx_a, 0), (cx_v, 1), (cx_e, 2)]:
        block(ax, cxn, y, col_w, 1.0,
              f"+ modality_embed[{idx}]\nLayerNorm",
              fc=LB_SOFT, ec=LB_DEEP, lw=1.4,
              fontsize=10.5, color=DARK_TEXT)

    # ---- Merge arrows from each column into centre ----
    for cxn in (cx_a, cx_v, cx_e):
        arrow(ax, cxn, y - 0.5, cx_main, y - 1.6)

    # ============= Stack into sequence =============
    y = 18.5
    block(ax, cx_main, y, 23.0, 0.9,
          "Stack into sequence   ·   tokens = [audio_tok, vision_tok, eeg_tok]   "
          "·   shape (B, 3, 256)",
          fc=LB_PALE, ec=LB_DEEP, lw=1.6,
          fontsize=11.5, weight="bold", color=DARK_TEXT)
    arrow(ax, cx_main, y - 0.45, cx_main, y - 1.25)

    # ============= STAGE 2.5 — Softhard Modality Dropout =============
    y = 16.7
    stage_tab(ax, 27.0, y, "STAGE 2.5", fc=ORANGE, ec=OR_DEEP)
    ax.text(28.5, y - 0.7, "robustness",
            ha="left", va="center", fontsize=9.5,
            color=OR_DEEP, style="italic")

    block(ax, cx_main, y, 23.0, 1.3,
          "Softhard Modality Dropout   ·   (training only)\n"
          "with p = 1/3   ·   pick m ∈ {audio, vision, eeg}   ·   "
          "zero its token",
          fc=OR_SOFT, ec=OR_DEEP, lw=2.0,
          fontsize=11, weight="bold", color=DARK_TEXT)
    arrow(ax, cx_main, y - 0.65, cx_main, y - 1.4)

    # ============= STAGE 3 — Transformer Encoder × 2 layers =============
    y_container_top = 14.8
    y_container_bot = 10.6
    container_h = y_container_top - y_container_bot

    stage_tab(ax, 2.0, (y_container_top + y_container_bot) / 2, "STAGE 3")
    ax.text(0.5, (y_container_top + y_container_bot) / 2 - 0.85,
            "Transformer\nEncoder × 2",
            ha="left", va="center", fontsize=9.5,
            color=GREY, style="italic")

    # Big container box
    container = FancyBboxPatch(
        (cx_main - 12.0, y_container_bot), 24.0, container_h,
        boxstyle="round,pad=0.05,rounding_size=0.2",
        fc=LB_SOFT, ec=LB_DEEP, lw=2.0)
    ax.add_patch(container)

    ax.text(cx_main, y_container_top - 0.5,
            "Transformer Encoder   ·   × 2 layers   (pre-LayerNorm)",
            ha="center", fontsize=12, fontweight="bold", color=LB_DEEP)

    # 4 horizontal sub-blocks of one encoder layer
    sub_y = (y_container_top + y_container_bot) / 2 - 0.4
    sub_w = 5.0
    sub_h = 1.4

    # Positions for 4 sub-blocks
    positions = [
        cx_main - 9.5,   # MHA
        cx_main - 3.5,   # + residual
        cx_main + 2.5,   # FFN
        cx_main + 8.5,   # + residual
    ]

    block(ax, positions[0], sub_y, sub_w, sub_h,
          "LayerNorm  →\nMulti-Head\nSelf-Attention\n8 heads, d=32",
          fc=LIGHT_BLUE, ec=LB_DEEP, lw=1.6,
          fontsize=9.5, weight="bold", color="white")

    block(ax, positions[1], sub_y, sub_w, sub_h,
          "+ residual",
          fc=PANEL_WHITE, ec=GREY, lw=1.2,
          fontsize=10, color=DARK_TEXT)

    block(ax, positions[2], sub_y, sub_w, sub_h,
          "LayerNorm  →\nFFN\n( 256 → 1024 → 256 )\nGELU",
          fc=LIGHT_BLUE, ec=LB_DEEP, lw=1.6,
          fontsize=9.5, weight="bold", color="white")

    block(ax, positions[3], sub_y, sub_w, sub_h,
          "+ residual",
          fc=PANEL_WHITE, ec=GREY, lw=1.2,
          fontsize=10, color=DARK_TEXT)

    # Connecting arrows between sub-blocks
    for i in range(3):
        x1 = positions[i] + sub_w / 2
        x2 = positions[i + 1] - sub_w / 2
        arrow(ax, x1, sub_y, x2, sub_y)

    # Exit arrow from the container
    arrow(ax, cx_main, y_container_bot, cx_main, y_container_bot - 0.85)

    # ============= STAGE 4 — Extract + Mean-pool =============
    y = 9.0
    stage_tab(ax, 2.0, y, "STAGE 4")
    ax.text(0.5, y - 0.85, "Extract +\nmean-pool",
            ha="left", va="center", fontsize=9.5,
            color=GREY, style="italic")
    block(ax, cx_main, y, 23.0, 1.1,
          "Extract per-modality embeddings  +  mean-pool fused embedding\n"
          "audio_embed  ·  vision_embed  ·  eeg_embed   →   fused_embed (B, 256)",
          fc=LB_PALE, ec=LB_DEEP, lw=1.6,
          fontsize=10.5, color=DARK_TEXT)
    arrow(ax, cx_main, y - 0.55, cx_main, y - 1.4)

    # ============= FUSED EMBEDDING (highlight) =============
    y = 7.0
    sharp_block(ax, cx_main, y, 15.0, 1.1,
                "FUSED   EMBEDDING   ·   256-dim",
                fc=OR_SOFT, ec=OR_DEEP, lw=3.0,
                fontsize=14, weight="bold", color=DARK_TEXT)
    arrow(ax, cx_main, y - 0.55, cx_main, y - 1.3)

    # ============= STAGE 5 — MLP head =============
    y = 5.0
    stage_tab(ax, 2.0, y, "STAGE 5")
    ax.text(0.5, y - 0.65, "MLP head",
            ha="left", va="center", fontsize=9.5,
            color=GREY, style="italic")
    block(ax, cx_main, y, 23.0, 1.1,
          "Linear ( 256  →  256 )   →   GELU   →   Dropout ( 0.1 )   "
          "→   Linear ( 256  →  5 )",
          fc=LIGHT_BLUE, ec=LB_DEEP, lw=1.6,
          fontsize=11.5, weight="bold", color="white")
    arrow(ax, cx_main, y - 0.55, cx_main, y - 1.3)

    # ============= OUTPUT =============
    y = 3.0
    sharp_block(ax, cx_main, y, 16.0, 1.1,
                "OUTPUT   ·   logits (B, 5)   —   "
                "Neutral · Anger · Happiness · Sadness · Calmness",
                fc=OR_SOFT, ec=OR_DEEP, lw=2.5,
                fontsize=12, weight="bold", color=DARK_TEXT)

    # ---- Footer note ----
    ax.text(XLIM / 2, 1.2,
            "Phase-2 commitment: forward() returns named per-modality "
            "embeddings; accepts x_eeg = zeros(B, 960) as a valid call.",
            ha="center", va="center",
            fontsize=9.5, style="italic", color=GREY)

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor=BG_PALE)
    plt.close(fig)
    print(f"wrote {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
