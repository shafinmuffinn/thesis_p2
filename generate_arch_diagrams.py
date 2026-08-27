"""Generate the four architecture pipeline diagrams.

Four PNG figures, each a single per-model architecture diagram showing
inputs, internal blocks, shape transformations, and outputs:

  arch_ast.png       — AST (audio encoder)
  arch_vit.png       — ViT (vision encoder)
  arch_eegnet.png    — EEGNet (EEG encoder, CNN)
  arch_fusion.png    — TrimodalAttentionFusion (trimodal self-attention fusion)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

# --- Palette (matches thesis slides) --------------------------------------
NAVY        = "#1A3A5C"
DEEP_NAVY   = "#0F2540"
LIGHT_BLUE  = "#D9E7F5"
MID_BLUE    = "#2E7EB8"
PALE_BLUE   = "#EEF3FA"
WHITE       = "#FFFFFF"
GOLD        = "#C87F12"
GOLD_BG     = "#FDF1E0"
GREY        = "#666666"
LIGHT_GREY  = "#F2F2F2"
GREEN       = "#2E8B57"
GREEN_BG    = "#D4EDDA"
RED         = "#B01F1F"
RED_BG      = "#FAE4E4"
DARK_TEXT   = "#202020"


# --- Helpers --------------------------------------------------------------

def block(ax, cx, cy, w, h, text, *,
          fc=LIGHT_BLUE, ec=NAVY, lw=1.5,
          fontsize=10, weight="normal", color=DARK_TEXT,
          radius=0.15):
    """Rounded rectangle with centred multi-line text."""
    box = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        fc=fc, ec=ec, lw=lw)
    ax.add_patch(box)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=color,
            linespacing=1.3)


def sharp_block(ax, cx, cy, w, h, text, *,
                fc=LIGHT_BLUE, ec=NAVY, lw=1.5,
                fontsize=10, weight="normal", color=DARK_TEXT):
    """Sharp-corner rectangle (used for input/output emphasis)."""
    rect = Rectangle(
        (cx - w / 2, cy - h / 2), w, h,
        facecolor=fc, edgecolor=ec, linewidth=lw)
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, color=color,
            linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, *, color=NAVY, lw=1.4,
          shape_label=None, label_dx=0.4, label_dy=0):
    """Arrow with optional small italic shape label beside it."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>,head_width=0.25,head_length=0.5",
                        color=color, lw=lw, shrinkA=0, shrinkB=0))
    if shape_label is not None:
        mx, my = (x1 + x2) / 2 + label_dx, (y1 + y2) / 2 + label_dy
        ax.text(mx, my, shape_label, ha="left", va="center",
                fontsize=8.5, style="italic", color=GREY)


def side_note(ax, x, y, text, *, ha="left", color=GREY, fontsize=9):
    ax.text(x, y, text, ha=ha, va="center",
            fontsize=fontsize, color=color, style="italic",
            linespacing=1.3)


def container(ax, x, y, w, h, *, fc=PALE_BLUE, ec=NAVY, lw=2.0):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.25",
        fc=fc, ec=ec, lw=lw)
    ax.add_patch(box)


def title_block(ax, title, subtitle, *, xlim, top):
    ax.text(xlim / 2, top, title,
            ha="center", fontsize=17, fontweight="bold", color=NAVY)
    ax.text(xlim / 2, top - 0.55, subtitle,
            ha="center", fontsize=11, style="italic", color=GREY)


def footer(ax, lines, *, xlim, bottom_y, total_w=11):
    """Footer block with model card info."""
    container(ax, (xlim - total_w) / 2, bottom_y, total_w, 1.7,
              fc=LIGHT_GREY, ec=GREY, lw=1.0)
    n = len(lines)
    for i, (txt, weight) in enumerate(lines):
        y = bottom_y + 1.5 - (i + 0.5) * (1.4 / n)
        ax.text(xlim / 2, y, txt,
                ha="center", fontsize=9.5,
                fontweight=weight, color=DARK_TEXT)


# --- Diagram 1: AST -------------------------------------------------------

def fig_ast():
    XLIM, YLIM = 12, 26
    fig, ax = plt.subplots(figsize=(8.5, 13), dpi=180)
    ax.set_xlim(0, XLIM)
    ax.set_ylim(0, YLIM)
    ax.set_aspect("equal")
    ax.axis("off")

    title_block(ax, "AST  —  Audio Spectrogram Transformer",
                "Audio modality encoder", xlim=XLIM, top=YLIM - 0.6)

    cx = 6

    # INPUT
    sharp_block(ax, cx, 23.8, 7, 1.0,
                "INPUT  ·  5 s audio clip @ 16 kHz mono",
                fc=GOLD_BG, ec=GOLD, lw=2,
                fontsize=11, weight="bold", color=NAVY)
    side_note(ax, 10.0, 23.8, "shape\n(80 000,)")

    arrow(ax, cx, 23.3, cx, 22.5)

    # Feature extractor
    block(ax, cx, 22.0, 7, 1.0,
          "AST feature extractor  (Hugging Face)\nLog-mel spectrogram, 128-bin",
          fc=LIGHT_GREY, ec=GREY, fontsize=10)
    side_note(ax, 10.0, 22.0, "windowing\n+ mel filterbank")

    arrow(ax, cx, 21.5, cx, 20.7, shape_label="(T × 128)   T ≈ 500")

    # Patch embedding
    block(ax, cx, 20.2, 7, 1.0,
          "Patch embedding\n16 × 16 patches, stride (10, 10)",
          fc=LIGHT_BLUE, ec=MID_BLUE, fontsize=10)
    side_note(ax, 10.0, 20.2, "linear proj\n→ 768-d /patch")

    arrow(ax, cx, 19.7, cx, 18.9, shape_label="(N_patches, 768)")

    # [CLS] + position
    block(ax, cx, 18.4, 7, 1.0,
          "Prepend [CLS] token  +  position embeddings",
          fc=LIGHT_GREY, ec=GREY, fontsize=10)

    arrow(ax, cx, 17.9, cx, 17.1, shape_label="(1 + N_patches, 768)")

    # Transformer Encoder × 12 (big container)
    container(ax, 1.0, 7.7, 10, 9.0, fc=PALE_BLUE, ec=NAVY, lw=2)
    ax.text(cx, 16.4, "Transformer Encoder  ·  × 12 layers",
            ha="center", fontsize=12, fontweight="bold", color=NAVY)
    ax.text(cx, 15.85, "(per layer, pre-LN order)",
            ha="center", fontsize=9, style="italic", color=GREY)

    block(ax, cx, 15.0, 7, 0.85,
          "Multi-Head Self-Attention   ·   12 heads, d = 64/head",
          fc=DEEP_NAVY, ec=NAVY, fontsize=10, weight="bold", color=WHITE)
    arrow(ax, cx, 14.55, cx, 13.85)

    block(ax, cx, 13.45, 7, 0.7,
          "+ residual  →  LayerNorm",
          fc=LIGHT_GREY, ec=GREY, fontsize=9.5)
    arrow(ax, cx, 13.1, cx, 12.4)

    block(ax, cx, 12.0, 7, 0.85,
          "FFN ( 768 → 3072 → 768 )   ·   GELU",
          fc=MID_BLUE, ec=NAVY, fontsize=10, weight="bold", color=WHITE)
    arrow(ax, cx, 11.55, cx, 10.85)

    block(ax, cx, 10.45, 7, 0.7,
          "+ residual  →  LayerNorm",
          fc=LIGHT_GREY, ec=GREY, fontsize=9.5)

    ax.text(cx, 9.4, "(stack of 12 identical layers)",
            ha="center", fontsize=10, fontweight="bold", color=NAVY,
            style="italic")
    ax.text(cx, 8.7, "→ contextualised token sequence",
            ha="center", fontsize=9, color=GREY, style="italic")

    arrow(ax, cx, 7.6, cx, 6.85)

    # Extract CLS
    block(ax, cx, 6.4, 7, 0.85,
          "Extract  [CLS]  token  only",
          fc=LIGHT_GREY, ec=GREY, fontsize=10.5)
    side_note(ax, 10.0, 6.4, "token [0]")

    arrow(ax, cx, 5.95, cx, 4.95, shape_label="(768,)")

    # Audio embedding (fusion input)
    sharp_block(ax, cx, 4.4, 8, 1.1,
                "AUDIO  EMBEDDING   ·   768-dim",
                fc=GOLD_BG, ec=GOLD, lw=2.8,
                fontsize=12.5, weight="bold", color=NAVY)
    side_note(ax, 10.4, 4.4, "→ to fusion\nmodule", color=GOLD,
              fontsize=9.5)

    arrow(ax, cx, 3.85, cx, 3.0)

    # Per-modality head
    block(ax, cx, 2.55, 7, 0.85,
          "Linear ( 768 → 5 )  →  audio logits",
          fc=GREEN_BG, ec=GREEN, fontsize=10)
    side_note(ax, 10.0, 2.55, "per-modality\nCE loss only", fontsize=8.5)

    # Footer
    footer(ax, [
        ("Checkpoint: MIT/ast-finetuned-audioset-10-10-0.4593", "bold"),
        ("Parameters ≈ 87 M  ·  Pre-trained on AudioSet (2 M clips, 527 classes)", "normal"),
        ("Fine-tuned per subject: 5 frozen + 5 fine-tune epochs  ·  Adam, batch 8", "normal"),
    ], xlim=XLIM, bottom_y=0.1)

    fig.savefig(OUT / "arch_ast.png", dpi=180, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# --- Diagram 2: ViT -------------------------------------------------------

def fig_vit():
    XLIM, YLIM = 12, 26
    fig, ax = plt.subplots(figsize=(8.5, 13), dpi=180)
    ax.set_xlim(0, XLIM)
    ax.set_ylim(0, YLIM)
    ax.set_aspect("equal")
    ax.axis("off")

    title_block(ax, "ViT  —  Facial-Emotion Vision Transformer",
                "Vision modality encoder", xlim=XLIM, top=YLIM - 0.6)

    cx = 6

    # INPUT
    sharp_block(ax, cx, 23.8, 7.5, 1.0,
                "INPUT  ·  5 s video, 25 frames @ 5 fps,  224 × 224 × 3",
                fc=GOLD_BG, ec=GOLD, lw=2,
                fontsize=11, weight="bold", color=NAVY)
    side_note(ax, 10.2, 23.8, "shape\n(25, 3, 224, 224)")

    arrow(ax, cx, 23.3, cx, 22.4)

    # Per-frame loop indicator
    ax.text(cx, 21.95, "── per-frame loop  ( × 25 frames )  ──",
            ha="center", fontsize=10, fontweight="bold",
            color=NAVY, style="italic")

    arrow(ax, cx, 21.55, cx, 20.85)

    # HF processor
    block(ax, cx, 20.45, 7, 0.9,
          "HF image processor  ·  normalise (μ, σ) per channel",
          fc=LIGHT_GREY, ec=GREY, fontsize=10)

    arrow(ax, cx, 20.0, cx, 19.25, shape_label="(3, 224, 224)")

    # Patch embedding
    block(ax, cx, 18.85, 7, 0.9,
          "Patch embedding  ·  16 × 16 patches  →  196 patches/frame\n"
          "linear projection → 768-d per patch",
          fc=LIGHT_BLUE, ec=MID_BLUE, fontsize=9.5)
    arrow(ax, cx, 18.35, cx, 17.65, shape_label="(196, 768)")

    block(ax, cx, 17.25, 7, 0.85,
          "Prepend [CLS]  +  position embeddings",
          fc=LIGHT_GREY, ec=GREY, fontsize=10)
    arrow(ax, cx, 16.8, cx, 16.05, shape_label="(197, 768)")

    # Transformer Encoder × 12
    container(ax, 1.0, 7.45, 10, 8.5, fc=PALE_BLUE, ec=NAVY, lw=2)
    ax.text(cx, 15.6, "Transformer Encoder  ·  × 12 layers",
            ha="center", fontsize=12, fontweight="bold", color=NAVY)
    ax.text(cx, 15.05, "(per layer)", ha="center",
            fontsize=9, style="italic", color=GREY)

    block(ax, cx, 14.3, 7, 0.8,
          "Multi-Head Self-Attention   ·   12 heads",
          fc=DEEP_NAVY, ec=NAVY, fontsize=10, weight="bold", color=WHITE)
    arrow(ax, cx, 13.9, cx, 13.2)

    block(ax, cx, 12.8, 7, 0.65,
          "+ residual  →  LayerNorm",
          fc=LIGHT_GREY, ec=GREY, fontsize=9.5)
    arrow(ax, cx, 12.475, cx, 11.75)

    block(ax, cx, 11.35, 7, 0.8,
          "FFN ( 768 → 3072 → 768 )   ·   GELU",
          fc=MID_BLUE, ec=NAVY, fontsize=10, weight="bold", color=WHITE)
    arrow(ax, cx, 10.95, cx, 10.25)

    block(ax, cx, 9.85, 7, 0.65,
          "+ residual  →  LayerNorm",
          fc=LIGHT_GREY, ec=GREY, fontsize=9.5)

    ax.text(cx, 8.8, "(stack of 12 identical layers)",
            ha="center", fontsize=10, fontweight="bold",
            color=NAVY, style="italic")
    ax.text(cx, 8.1, "→ Extract [CLS] token per frame  ·  (768,)",
            ha="center", fontsize=9, color=GREY, style="italic")

    arrow(ax, cx, 7.35, cx, 6.6)

    # End per-frame loop / mean pool across 25 frames
    block(ax, cx, 6.1, 7.5, 0.95,
          "Mean-pool [CLS] tokens across  25 frames",
          fc=LIGHT_GREY, ec=GREY, fontsize=10.5, weight="bold")
    side_note(ax, 10.4, 6.1, "bag-of-frames\naveraging",
              fontsize=9)

    arrow(ax, cx, 5.6, cx, 4.85, shape_label="(768,)")

    # Vision embedding (fusion input)
    sharp_block(ax, cx, 4.3, 8, 1.1,
                "VISION  EMBEDDING   ·   768-dim",
                fc=GOLD_BG, ec=GOLD, lw=2.8,
                fontsize=12.5, weight="bold", color=NAVY)
    side_note(ax, 10.4, 4.3, "→ to fusion\nmodule", color=GOLD,
              fontsize=9.5)

    arrow(ax, cx, 3.75, cx, 2.9)

    block(ax, cx, 2.45, 7, 0.85,
          "Linear ( 768 → 5 )  →  vision logits",
          fc=GREEN_BG, ec=GREEN, fontsize=10)
    side_note(ax, 10.0, 2.45, "per-modality\nCE loss only", fontsize=8.5)

    footer(ax, [
        ("Checkpoint: dima806/facial_emotions_image_detection", "bold"),
        ("Parameters ≈ 86 M  ·  Pre-trained on 7-class facial expressions, re-head to 5", "normal"),
        ("Fine-tuned per subject: 3 frozen + 2 fine-tune epochs  ·  Adam, batch 8", "normal"),
    ], xlim=XLIM, bottom_y=0.05)

    fig.savefig(OUT / "arch_vit.png", dpi=180, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# --- Diagram 3: EEGNet ----------------------------------------------------

def fig_eegnet():
    XLIM, YLIM = 12, 26
    fig, ax = plt.subplots(figsize=(8.5, 13), dpi=180)
    ax.set_xlim(0, XLIM)
    ax.set_ylim(0, YLIM)
    ax.set_aspect("equal")
    ax.axis("off")

    title_block(ax, "EEGNet  —  Depthwise Separable CNN",
                "EEG modality encoder", xlim=XLIM, top=YLIM - 0.6)

    cx = 6

    # INPUT
    sharp_block(ax, cx, 23.8, 7.5, 1.0,
                "INPUT  ·  5 s EEG, 30 channels @ 100 Hz",
                fc=GOLD_BG, ec=GOLD, lw=2,
                fontsize=11, weight="bold", color=NAVY)
    side_note(ax, 10.4, 23.8, "shape\n(1, 30, 500)")

    arrow(ax, cx, 23.3, cx, 22.5)

    # BLOCK 1 container
    container(ax, 1.0, 14.8, 10, 7.5, fc=PALE_BLUE, ec=NAVY, lw=2)
    ax.text(cx, 21.95, "BLOCK 1  ·  Temporal + Spatial Filtering",
            ha="center", fontsize=11.5, fontweight="bold", color=NAVY)

    block(ax, cx, 21.2, 7.5, 0.85,
          "Conv2D ( 1 → 8,  kernel = (1, 300),  pad = same )",
          fc=LIGHT_BLUE, ec=MID_BLUE, fontsize=10)
    side_note(ax, 10.4, 21.2, "8 temporal\nfilters", fontsize=8.5)
    arrow(ax, cx, 20.775, cx, 20.15, shape_label="(8, 30, 500)")

    block(ax, cx, 19.75, 7.5, 0.65,
          "BatchNorm2D ( 8 )  →  ELU",
          fc=LIGHT_GREY, ec=GREY, fontsize=9.5)
    arrow(ax, cx, 19.425, cx, 18.7)

    block(ax, cx, 18.25, 7.5, 0.95,
          "DepthwiseConv2D ( 8 → 64,  kernel = (30, 1),  groups = 8 )\n"
          "spatial mixing across all 30 channels",
          fc=LIGHT_BLUE, ec=MID_BLUE, fontsize=9.5)
    side_note(ax, 10.4, 18.25, "L2 max-norm\nconstraint",
              color=RED, fontsize=8.5)
    arrow(ax, cx, 17.775, cx, 17.05, shape_label="(64, 1, 500)")

    block(ax, cx, 16.65, 7.5, 0.65,
          "BatchNorm2D ( 64 )  →  ELU",
          fc=LIGHT_GREY, ec=GREY, fontsize=9.5)
    arrow(ax, cx, 16.325, cx, 15.6)

    block(ax, cx, 15.2, 7.5, 0.65,
          "AvgPool2D ( (1, 4) )  →  Dropout ( 0.5 )",
          fc=LIGHT_GREY, ec=GREY, fontsize=9.5)

    arrow(ax, cx, 14.875, cx, 13.9, shape_label="(64, 1, 125)")

    # BLOCK 2 container
    container(ax, 1.0, 9.0, 10, 4.7, fc=PALE_BLUE, ec=NAVY, lw=2)
    ax.text(cx, 13.35, "BLOCK 2  ·  Pointwise Separable Conv",
            ha="center", fontsize=11.5, fontweight="bold", color=NAVY)

    block(ax, cx, 12.6, 7.5, 0.85,
          "Conv2D ( 64 → 64,  kernel = (1, 16),  pad = same )",
          fc=LIGHT_BLUE, ec=MID_BLUE, fontsize=10)
    side_note(ax, 10.4, 12.6, "L2 max-norm",
              color=RED, fontsize=8.5)
    arrow(ax, cx, 12.175, cx, 11.55)

    block(ax, cx, 11.15, 7.5, 0.65,
          "BatchNorm2D ( 64 )  →  ELU",
          fc=LIGHT_GREY, ec=GREY, fontsize=9.5)
    arrow(ax, cx, 10.825, cx, 10.1)

    block(ax, cx, 9.7, 7.5, 0.65,
          "AvgPool2D ( (1, 8) )  →  Dropout ( 0.5 )",
          fc=LIGHT_GREY, ec=GREY, fontsize=9.5)

    arrow(ax, cx, 9.375, cx, 8.45, shape_label="(64, 1, 15)")

    # Flatten
    block(ax, cx, 7.95, 7.5, 0.85,
          "Flatten   ·   64 × 1 × 15  →  960",
          fc=LIGHT_GREY, ec=GREY, fontsize=10.5, weight="bold")

    arrow(ax, cx, 7.5, cx, 6.55, shape_label="(960,)")

    # EEG embedding (fusion input)
    sharp_block(ax, cx, 6.0, 8, 1.1,
                "EEG  EMBEDDING   ·   960-dim",
                fc=GOLD_BG, ec=GOLD, lw=2.8,
                fontsize=12.5, weight="bold", color=NAVY)
    side_note(ax, 10.4, 6.0, "→ to fusion\nmodule", color=GOLD,
              fontsize=9.5)

    arrow(ax, cx, 5.45, cx, 4.55)

    block(ax, cx, 4.1, 7.5, 0.85,
          "Linear ( 960 → 5 )  →  EEG logits",
          fc=GREEN_BG, ec=GREEN, fontsize=10)
    side_note(ax, 10.4, 4.1, "per-modality\nCE loss only", fontsize=8.5)

    footer(ax, [
        ("EEGNet  ·  F1=8, D=8, F2=64, kernLength=300, Chans=30, Samples=500", "bold"),
        ("Parameters ≈ 2.2 M  ·  Trained from random init  ·  350 epochs per subject", "normal"),
        ("Adam, lr=1e-4, dropout 0.5  ·  L2 max-norm hooks on depthwise + dense weights", "normal"),
    ], xlim=XLIM, bottom_y=0.05)

    fig.savefig(OUT / "arch_eegnet.png", dpi=180, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# --- Diagram 4: Trimodal Fusion -------------------------------------------

def fig_fusion():
    XLIM, YLIM = 14, 26
    fig, ax = plt.subplots(figsize=(10, 13), dpi=180)
    ax.set_xlim(0, XLIM)
    ax.set_ylim(0, YLIM)
    ax.set_aspect("equal")
    ax.axis("off")

    title_block(ax, "TrimodalAttentionFusion  —  Self-Attention over Modality Tokens",
                "Trimodal fusion module  ·  2.28 M trainable parameters",
                xlim=XLIM, top=YLIM - 0.6)

    # Three input columns
    cx_a, cx_v, cx_e = 3, 7, 11  # column centres
    cx = 7  # main centre

    # INPUTS — three parallel streams
    sharp_block(ax, cx_a, 23.8, 3.4, 1.0,
                "x_audio   from AST\n(768,)",
                fc=GOLD_BG, ec=GOLD, lw=2,
                fontsize=10.5, weight="bold", color=NAVY)
    sharp_block(ax, cx_v, 23.8, 3.4, 1.0,
                "x_vision   from ViT\n(768,)",
                fc=GOLD_BG, ec=GOLD, lw=2,
                fontsize=10.5, weight="bold", color=NAVY)
    sharp_block(ax, cx_e, 23.8, 3.4, 1.0,
                "x_eeg   from EEGNet\n(960,)",
                fc=GOLD_BG, ec=GOLD, lw=2,
                fontsize=10.5, weight="bold", color=NAVY)

    for col_cx in (cx_a, cx_v, cx_e):
        arrow(ax, col_cx, 23.3, col_cx, 22.6)

    # STAGE 1: Per-modality projection — parallel
    block(ax, cx_a, 22.1, 3.4, 0.9,
          "Linear\n( 768 → 256 )",
          fc=LIGHT_BLUE, ec=MID_BLUE, fontsize=10)
    block(ax, cx_v, 22.1, 3.4, 0.9,
          "Linear\n( 768 → 256 )",
          fc=LIGHT_BLUE, ec=MID_BLUE, fontsize=10)
    block(ax, cx_e, 22.1, 3.4, 0.9,
          "Linear\n( 960 → 256 )",
          fc=LIGHT_BLUE, ec=MID_BLUE, fontsize=10)

    ax.text(0.3, 22.1, "Stage 1\nProjection", ha="left", va="center",
            fontsize=9.5, fontweight="bold", color=NAVY)

    for col_cx in (cx_a, cx_v, cx_e):
        arrow(ax, col_cx, 21.65, col_cx, 20.95)

    # STAGE 2: Add modality embedding — parallel
    block(ax, cx_a, 20.5, 3.4, 0.9,
          "+ modality_embed[0]\nLayerNorm",
          fc=LIGHT_GREY, ec=GREY, fontsize=9.5)
    block(ax, cx_v, 20.5, 3.4, 0.9,
          "+ modality_embed[1]\nLayerNorm",
          fc=LIGHT_GREY, ec=GREY, fontsize=9.5)
    block(ax, cx_e, 20.5, 3.4, 0.9,
          "+ modality_embed[2]\nLayerNorm",
          fc=LIGHT_GREY, ec=GREY, fontsize=9.5)

    ax.text(0.3, 20.5, "Stage 2\nModality\nembedding",
            ha="left", va="center",
            fontsize=9.5, fontweight="bold", color=NAVY)

    # Three streams converge into the stacked sequence
    for col_cx in (cx_a, cx_v, cx_e):
        arrow(ax, col_cx, 20.05, cx, 19.15, lw=1.4)

    # Stacked sequence label
    block(ax, cx, 18.7, 11, 0.85,
          "Stack into sequence   tokens = [audio_tok, vision_tok, eeg_tok]   ·   shape (B, 3, 256)",
          fc=LIGHT_GREY, ec=NAVY, fontsize=10.5, weight="bold")

    arrow(ax, cx, 18.25, cx, 17.6)

    # Stage 2.5: Modality dropout (training only)
    block(ax, cx, 17.15, 11, 1.0,
          "Softhard Modality Dropout   (training only)\n"
          "with p = 1/3  ·  pick m ∈ {audio, vision, eeg}  ·  zero its token",
          fc="#FFF3CD", ec=GOLD, fontsize=10, weight="bold")
    side_note(ax, 13.4, 17.15, "Stage 2.5\nrobustness", ha="left",
              color=GOLD, fontsize=9)

    arrow(ax, cx, 16.55, cx, 15.85)

    # STAGE 3: Transformer encoder
    container(ax, 1.0, 8.0, 12, 7.4, fc=PALE_BLUE, ec=NAVY, lw=2.0)
    ax.text(cx, 15.0, "Stage 3  ·  Transformer Encoder  ·  × 2 layers",
            ha="center", fontsize=12, fontweight="bold", color=NAVY)
    ax.text(cx, 14.5, "(per layer, pre-LN order)",
            ha="center", fontsize=9, style="italic", color=GREY)

    block(ax, cx, 13.7, 11, 0.8,
          "LayerNorm   →   Multi-Head Self-Attention   ·   8 heads, d = 32/head",
          fc=DEEP_NAVY, ec=NAVY, fontsize=10.5, weight="bold", color=WHITE)
    arrow(ax, cx, 13.3, cx, 12.65)

    block(ax, cx, 12.25, 11, 0.65,
          "+ residual",
          fc=LIGHT_GREY, ec=GREY, fontsize=9.5)
    arrow(ax, cx, 11.925, cx, 11.2)

    block(ax, cx, 10.8, 11, 0.8,
          "LayerNorm   →   FFN ( 256 → 1024 → 256 )   ·   GELU",
          fc=MID_BLUE, ec=NAVY, fontsize=10.5, weight="bold", color=WHITE)
    arrow(ax, cx, 10.4, cx, 9.75)

    block(ax, cx, 9.35, 11, 0.65,
          "+ residual",
          fc=LIGHT_GREY, ec=GREY, fontsize=9.5)

    ax.text(cx, 8.5, "(× 2 stacked layers)   →   post-attention sequence (B, 3, 256)",
            ha="center", fontsize=10, fontweight="bold",
            color=NAVY, style="italic")

    arrow(ax, cx, 7.7, cx, 7.0)

    # Stage 4: Extract per-modality embeddings + mean pool
    block(ax, cx, 6.55, 12, 1.1,
          "Stage 4   ·   Extract per-modality embeddings  +  mean-pool fused embedding\n"
          "audio_embed  =  tokens[:,0]   ·   vision_embed  =  tokens[:,1]   ·   eeg_embed  =  tokens[:,2]",
          fc=LIGHT_GREY, ec=NAVY, fontsize=9.5)

    arrow(ax, cx, 5.95, cx, 5.15, shape_label="fused_embed  (B, 256)",
          label_dx=0.6)

    # FUSED EMBEDDING (highlight)
    sharp_block(ax, cx, 4.55, 9, 1.0,
                "FUSED   EMBEDDING   ·   256-dim",
                fc=GOLD_BG, ec=GOLD, lw=2.8,
                fontsize=12.5, weight="bold", color=NAVY)

    arrow(ax, cx, 4.0, cx, 3.15)

    # MLP head
    block(ax, cx, 2.7, 12, 0.9,
          "Stage 5   ·   MLP head   Linear(256 → 256) → GELU → Dropout(0.1) → Linear(256 → 5)",
          fc=GREEN_BG, ec=GREEN, fontsize=10, weight="bold")

    arrow(ax, cx, 2.25, cx, 1.5, shape_label="logits (B, 5)",
          label_dx=0.6)

    sharp_block(ax, cx, 1.05, 9, 0.8,
                "OUTPUT   ·   5-class emotion logits",
                fc=DEEP_NAVY, ec=NAVY, lw=2,
                fontsize=11, weight="bold", color=WHITE)

    fig.savefig(OUT / "arch_fusion.png", dpi=180, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


# --- Driver ---------------------------------------------------------------

def main():
    fig_ast()
    fig_vit()
    fig_eegnet()
    fig_fusion()
    print(f"Figures saved to {OUT}/")
    for p in sorted(OUT.glob("arch_*.png")):
        kb = p.stat().st_size // 1024
        print(f"  {p.name}  ({kb} KB)")


if __name__ == "__main__":
    main()
