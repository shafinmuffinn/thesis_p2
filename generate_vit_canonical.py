"""Generate vit_canonical.png — a thesis-adapted version of the canonical
Vision Transformer diagram (Dosovitskiy et al., 2021).

Same layout and colour style as the original ViT paper / common explainer
diagrams, but with:
  - Class labels replaced by EAV's five emotions
  - Input image suggested as a face (simple icon)
  - A small annotation noting the per-frame + mean-pool usage
  - The 768-d CLS embedding flagged as feeding the trimodal fusion
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# Palette (matches the original style)
BG          = "#F4F4F2"
GOLD_BG     = "#FFE6B3"
PEACH       = "#FFD9B3"
MINT_BG     = "#C8E6C9"
ORANGE_NORM = "#FFB366"
BLUE_BG     = "#B3D9FF"
GREEN_MLP   = "#A5D6A7"
GREY_BAR    = "#BDBDBD"
DARK        = "#1A1A1A"
GREY_LINE   = "#9A9A9A"
NAVY_NOTE   = "#1F3A68"
GOLD_NOTE   = "#C8961B"


def main():
    fig, ax = plt.subplots(figsize=(16, 8), dpi=180)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.set_aspect("auto")
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 16, 8, fc=BG, ec="none"))

    # ===== Titles =====
    ax.text(5.5, 7.5, "Vision Transformer  (ViT)",
            ha="center", fontsize=15, fontweight="bold", color=DARK)
    ax.text(13.5, 7.5, "Transformer Encoder",
            ha="center", fontsize=14, fontweight="bold", color=DARK)

    # Vertical dashed separator
    for y in np.arange(0.4, 7.3, 0.22):
        ax.plot([11.2, 11.2], [y, y + 0.11],
                color=GREY_LINE, lw=1.2)

    # ===================================================================
    # LEFT PANEL — Vision Transformer overview
    # ===================================================================

    # --- Class box (top-left) ---
    ax.add_patch(Rectangle((0.4, 5.3), 1.7, 1.9, fc=GOLD_BG, ec="none"))
    ax.text(1.25, 7.0, "Class", ha="center",
            fontweight="bold", fontsize=11.5, color=DARK)
    classes = ["Neutral", "Anger", "Happiness", "Sadness", "Calmness"]
    for i, label in enumerate(classes):
        ax.text(1.25, 6.6 - i * 0.26, label,
                ha="center", fontsize=9.5, color=DARK)

    # --- MLP Head ---
    ax.add_patch(FancyBboxPatch((2.7, 5.85), 1.2, 0.85,
                                boxstyle="round,pad=0.05",
                                fc=PEACH, ec="none"))
    ax.text(3.3, 6.45, "MLP", ha="center", fontsize=11, fontweight="bold")
    ax.text(3.3, 6.10, "Head", ha="center", fontsize=11, fontweight="bold")

    # arrow MLP Head -> Class
    ax.annotate("", xy=(2.15, 6.27), xytext=(2.7, 6.27),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.8))

    # --- Transformer Encoder bar ---
    ax.add_patch(Rectangle((2.7, 4.2), 7.6, 0.85, fc=GREY_BAR, ec="none"))
    ax.text(6.5, 4.625, "Transformer Encoder",
            ha="center", va="center",
            fontsize=13, fontweight="bold", color="white")

    # arrow Transformer -> MLP Head
    ax.annotate("", xy=(3.3, 5.85), xytext=(3.3, 5.05),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.8))

    # --- Numbered tokens (CLS + patches) ---
    n_tokens = 10
    token_w, token_gap = 0.55, 0.2
    total_w = n_tokens * token_w + (n_tokens - 1) * token_gap
    start_x = 2.7 + (7.6 - total_w) / 2
    token_y = 3.2

    for i in range(n_tokens):
        x = start_x + i * (token_w + token_gap)
        ax.add_patch(Rectangle((x, token_y), token_w, 0.6,
                               fc=PEACH, ec="none"))
        ax.text(x + token_w / 2, token_y + 0.3, str(i),
                ha="center", va="center",
                fontsize=11, fontweight="bold", color=DARK)
        # Annotate token 0 as [CLS]
        if i == 0:
            ax.text(x + token_w / 2, token_y + 0.95, "[CLS]",
                    ha="center", va="bottom",
                    fontsize=8, fontweight="bold", color=GOLD_NOTE)

    # arrows tokens -> Transformer Encoder
    for i in range(n_tokens):
        x = start_x + i * (token_w + token_gap) + token_w / 2
        ax.annotate("", xy=(x, 4.2), xytext=(x, 3.85),
                    arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.2))

    # --- Linear projection bar ---
    ax.add_patch(Rectangle((start_x - 0.2, 2.25),
                           total_w + 0.4, 0.7,
                           fc=MINT_BG, ec="none"))
    ax.text(start_x + total_w / 2, 2.6,
            "Linear projection of Flattened Patches",
            ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=DARK)

    # arrows linear projection -> tokens (1..9)
    for i in range(1, n_tokens):
        x = start_x + i * (token_w + token_gap) + token_w / 2
        ax.annotate("", xy=(x, 3.2), xytext=(x, 2.95),
                    arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.1))

    # --- Image patches at the bottom (9 cells, suggesting a face) ---
    patch_y = 1.4
    patch_h = 0.65

    # Subtle face-tint colors (warm browns)
    face_colors = ["#F2C094", "#F0B380", "#EDA571", "#E8966C",
                   "#E58A60", "#D87A52", "#CC6E48", "#BD6440", "#A85638"]

    for i in range(1, n_tokens):
        x = start_x + i * (token_w + token_gap)
        ax.add_patch(Rectangle((x, patch_y), token_w, patch_h,
                               fc=face_colors[(i - 1) % len(face_colors)],
                               ec="#888", lw=0.4))

    # arrows patches -> linear projection
    for i in range(1, n_tokens):
        x = start_x + i * (token_w + token_gap) + token_w / 2
        ax.annotate("", xy=(x, 2.25), xytext=(x, 2.05),
                    arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.0))

    # --- Patch + Position Embedding caption (bottom left) ---
    ax.text(0.4, 1.85, "Patch + Position",
            ha="left", fontsize=11, fontweight="bold", color=DARK)
    ax.text(0.4, 1.55, "Embedding",
            ha="left", fontsize=11, fontweight="bold", color=DARK)
    ax.text(0.4, 1.25, "* Extra learnable",
            ha="left", fontsize=9, style="italic", color=DARK)
    ax.text(0.4, 1.0, "  [class] embedding",
            ha="left", fontsize=9, style="italic", color=DARK)

    # --- Face icon (suggests the input is a face image) ---
    face_x, face_y, face_size = 0.5, 0.05, 0.85
    # 3x3 patch grid background
    n_grid = 3
    cell = face_size / n_grid
    for i in range(n_grid):
        for j in range(n_grid):
            c = face_colors[(i * n_grid + j) % len(face_colors)]
            ax.add_patch(Rectangle((face_x + j * cell, face_y + i * cell),
                                   cell, cell, fc=c, ec="#888", lw=0.3))
    # Eyes
    eye_y = face_y + face_size * 0.65
    ax.add_patch(Circle((face_x + face_size * 0.32, eye_y),
                        0.045, fc=DARK, zorder=10))
    ax.add_patch(Circle((face_x + face_size * 0.68, eye_y),
                        0.045, fc=DARK, zorder=10))
    # Mouth (smile curve)
    theta = np.linspace(np.pi * 1.1, np.pi * 1.9, 30)
    cx, cy = face_x + face_size * 0.5, face_y + face_size * 0.35
    ax.plot(cx + 0.18 * np.cos(theta),
            cy + 0.15 * np.sin(theta + np.pi),
            color=DARK, lw=1.5, zorder=10)

    ax.text(face_x + face_size / 2, face_y - 0.18,
            "face crop",
            ha="center", fontsize=8, style="italic", color=GREY_LINE)

    # ===================================================================
    # RIGHT PANEL — Transformer Encoder zoom-in
    # ===================================================================
    rx_left = 12.0
    rx_right = 15.4
    rx_cx = (rx_left + rx_right) / 2

    # Outer rounded box
    ax.add_patch(FancyBboxPatch((11.85, 0.85), 3.6, 6.0,
                                boxstyle="round,pad=0.1",
                                fc="white", ec=DARK, lw=1.6))

    # Embedded Patches box (bottom)
    ax.add_patch(FancyBboxPatch((rx_left + 0.3, 0.4), 2.8, 0.55,
                                boxstyle="round,pad=0.05",
                                fc=PEACH, ec="none"))
    ax.text(rx_cx, 0.675, "Embedded Patches",
            ha="center", va="center",
            fontsize=10.5, fontweight="bold")

    # Helper layout (bottom → top)
    components = [
        ("Norm",                ORANGE_NORM, 0.55, False),
        ("Multi-Head\nAttention", BLUE_BG,   0.95, False),
        ("PLUS",                 None,        0.0,  True),   # residual junction
        ("Norm",                ORANGE_NORM, 0.55, False),
        ("MLP",                  GREEN_MLP,   0.95, False),
        ("PLUS",                 None,        0.0,  True),
    ]

    y = 0.95
    sep = 0.3
    plus_positions = []
    box_centers = []

    for label, fill, height, is_plus in components:
        # arrow from previous to this
        ax.annotate("", xy=(rx_cx, y + sep / 2),
                    xytext=(rx_cx, y - 0.05),
                    arrowprops=dict(arrowstyle="-|>",
                                    color=DARK, lw=1.5))

        if is_plus:
            ax.add_patch(Circle((rx_cx, y + sep / 2 + 0.18),
                                0.20, fc="white", ec=DARK, lw=1.5))
            ax.text(rx_cx, y + sep / 2 + 0.18, "+",
                    ha="center", va="center",
                    fontsize=15, fontweight="bold")
            plus_positions.append(y + sep / 2 + 0.18)
            y += sep + 0.36
        else:
            box_y = y + sep
            ax.add_patch(FancyBboxPatch(
                (rx_left + 0.4, box_y), 2.6, height,
                boxstyle="round,pad=0.05",
                fc=fill, ec="none"))
            for line_i, line in enumerate(label.split("\n")):
                ax.text(rx_cx,
                        box_y + height / 2 +
                        (0.18 if "\n" in label and line_i == 0 else
                         -0.18 if "\n" in label else 0),
                        line, ha="center", va="center",
                        fontsize=10.5, fontweight="bold")
            box_centers.append(box_y + height / 2)
            y = box_y + height

    # Residual skip lines (drawn on the left side of the inner column)
    skip_x = rx_left - 0.05
    # Skip 1: from Embedded Patches up to first +
    ax.plot([skip_x, skip_x], [0.95, plus_positions[0]],
            color=DARK, lw=1.5)
    ax.plot([skip_x, rx_cx - 0.20], [plus_positions[0], plus_positions[0]],
            color=DARK, lw=1.5)
    ax.annotate("", xy=(rx_cx - 0.20, plus_positions[0]),
                xytext=(rx_cx - 0.32, plus_positions[0]),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.5))

    # Skip 2: from after first + to second +
    skip2_x = rx_left - 0.05
    ax.plot([skip2_x, skip2_x],
            [plus_positions[0], plus_positions[1]],
            color=DARK, lw=1.5)
    ax.plot([skip2_x, rx_cx - 0.20],
            [plus_positions[1], plus_positions[1]],
            color=DARK, lw=1.5)
    ax.annotate("", xy=(rx_cx - 0.20, plus_positions[1]),
                xytext=(rx_cx - 0.32, plus_positions[1]),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.5))

    # Output arrow leaving the box
    ax.annotate("", xy=(rx_cx, 7.0), xytext=(rx_cx, plus_positions[1] + 0.2),
                arrowprops=dict(arrowstyle="-|>", color=DARK, lw=1.5))

    # ===================================================================
    # Bottom-banner annotation: "How this thesis uses it"
    # ===================================================================
    banner_x, banner_y = 0.4, -0.15
    # Skip — out of bounds. Place above the patch row at the right edge.

    # Right-edge annotation note about thesis usage
    note_x = 4.0
    note_y = 0.20
    ax.add_patch(FancyBboxPatch((note_x, note_y), 7.0, 0.95,
                                boxstyle="round,pad=0.05",
                                fc="#FFF3CD", ec=GOLD_NOTE, lw=1.5))
    ax.text(note_x + 3.5, note_y + 0.78,
            "How this thesis uses ViT",
            ha="center", fontsize=10.5, fontweight="bold", color=NAVY_NOTE)
    ax.text(note_x + 3.5, note_y + 0.4,
            "Run ViT independently per frame (×25 frames per 5-s clip)  →  mean-pool [CLS] across frames",
            ha="center", fontsize=9, color=DARK)
    ax.text(note_x + 3.5, note_y + 0.12,
            "The 768-d clip-level [CLS] embedding feeds the trimodal fusion module",
            ha="center", fontsize=9, color=DARK)

    # Source citation at the bottom right
    ax.text(15.4, 0.05,
            "adapted from Dosovitskiy et al. (2021)",
            ha="right", fontsize=7.5, style="italic", color=GREY_LINE)

    out_path = OUT / "vit_canonical.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"wrote {out_path}  ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
