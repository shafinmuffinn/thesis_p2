"""Generate fig_5_1_workflow.png — block diagram of the three-component
development workflow described in Section 5.1:

    Local dev machine  ↔ GitHub (version control)
            ↓                       ↓
                Google Colab Pro (cloud GPU)
                          ↕
                Google Drive (cloud storage)

Styled in the pale + light-blue + orange palette matching fig_4_2,
fig_6_1, fig_6_3, fig_6_4 and the poster diagrams.

Saves to LateX/images/fig_5_1_workflow.png.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT_DIR = Path(__file__).resolve().parent / "LateX" / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "fig_5_1_workflow.png"

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


def block(ax, cx, cy, w, h, title, body, *,
          fc=LB_SOFT, ec=LB_DEEP, lw=2.0,
          title_color=LB_DEEP, title_size=13.5,
          body_color=DARK_TEXT, body_size=10.5):
    """Block with a bold title and a centred multi-line body."""
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.08,rounding_size=0.2",
        fc=fc, ec=ec, lw=lw)
    ax.add_patch(rect)
    ax.text(cx, cy + h / 2 - 0.55, title,
            ha="center", va="center",
            fontsize=title_size, fontweight="bold", color=title_color)
    ax.text(cx, cy - 0.3, body,
            ha="center", va="center",
            fontsize=body_size, color=body_color, linespacing=1.4)


def arrow(ax, x1, y1, x2, y2, label=None, *, color=LB_DEEP, lw=2.0,
          label_offset=(0, 0), label_color=None, fontsize=10):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>,head_width=0.35,head_length=0.7",
                        color=color, lw=lw, shrinkA=2, shrinkB=2))
    if label is not None:
        mx = (x1 + x2) / 2 + label_offset[0]
        my = (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label,
                ha="center", va="center",
                fontsize=fontsize, style="italic",
                color=label_color or color,
                bbox=dict(boxstyle="round,pad=0.25",
                          fc=PANEL_WHITE, ec="none"))


def main():
    XLIM, YLIM = 24, 14
    fig, ax = plt.subplots(figsize=(12, 7), dpi=200)
    fig.patch.set_facecolor(BG_PALE)
    ax.set_facecolor(BG_PALE)
    ax.set_xlim(0, XLIM)
    ax.set_ylim(0, YLIM)
    ax.set_aspect("auto")
    ax.axis("off")

    # Title
    ax.text(XLIM / 2, 13.4,
            "Development Workflow",
            ha="center", fontsize=15, fontweight="bold", color=LB_DEEP)
    ax.text(XLIM / 2, 12.7,
            "Local code authoring   ·   cloud GPU training   "
            "·   cloud storage persistence",
            ha="center", fontsize=11, style="italic", color=OR_DEEP)

    # Coordinates
    cx_local  = 5.0
    cx_github = 12.0
    cx_drive  = 19.0
    cx_colab  = 12.0   # main centre

    y_top    = 10.5     # local / github / drive top row
    y_colab  = 5.5      # Colab in the middle bottom

    # ---- TOP ROW: Local dev machine ----
    block(ax, cx_local, y_top, 7.0, 3.0,
          "Local Dev Machine  (CPU)",
          "Mac M1  →  Windows PC + PyCharm\n"
          "code authoring  ·  CPU analysis\n"
          "VS Code / PyCharm  ·  Python 3.10",
          fc=LB_SOFT, ec=LB_DEEP)

    # ---- TOP CENTRE: GitHub repository (the version-control mediator) ----
    block(ax, cx_github, y_top, 6.5, 3.0,
          "GitHub Repository",
          "shafinmuffinn/thesis_p2  (private)\n"
          "code  ·  commit history\n"
          "fine-grained PAT for Colab access",
          fc=OR_SOFT, ec=OR_DEEP)

    # ---- TOP RIGHT: Google Drive ----
    block(ax, cx_drive, y_top, 7.0, 3.0,
          "Google Drive  (Storage)",
          "MyDrive/Thesis_EAV/\n"
          "pickles  ·  checkpoints  ·  logits\n"
          "results  ·  logs  ·  HF cache",
          fc=OR_SOFT, ec=OR_DEEP)

    # ---- BOTTOM CENTRE: Colab Pro (compute hub) ----
    block(ax, cx_colab, y_colab, 18.5, 3.2,
          "Google Colab Pro   ·   Cloud GPU",
          "NVIDIA L4   ·   24 GB VRAM   ·   83 TFLOPS FP16\n"
          "training + inference   ·   ~12 h for the full 42-subject rollout",
          fc=LIGHT_BLUE, ec=LB_DEEP,
          title_color="white", body_color="white",
          title_size=13.5, body_size=11)

    # ---- Arrows ----

    # Local ↔ GitHub (git push/pull)
    arrow(ax, cx_local + 3.5, y_top + 0.7, cx_github - 3.3, y_top + 0.7,
          label="git push", lw=1.8,
          label_offset=(0, 0.4))
    arrow(ax, cx_github - 3.3, y_top - 0.7, cx_local + 3.5, y_top - 0.7,
          label="git pull", lw=1.8,
          label_offset=(0, -0.4))

    # GitHub → Colab (clone / pull on session start)
    arrow(ax, cx_github, y_top - 1.5, cx_github, y_colab + 1.6,
          label="git clone / pull\n(session start)", lw=2.0,
          label_offset=(2.0, 0))

    # Drive ↔ Colab (read pickles / write artefacts)
    arrow(ax, cx_drive - 0.5, y_top - 1.5, cx_colab + 8.0, y_colab + 1.4,
          color=OR_DEEP, lw=2.0)
    ax.text((cx_drive + cx_colab + 8) / 2 + 1.5,
            (y_top + y_colab) / 2 - 0.4,
            "read pickles  /\nwrite artefacts",
            ha="center", va="center",
            fontsize=10, style="italic", color=OR_DEEP,
            bbox=dict(boxstyle="round,pad=0.25",
                      fc=PANEL_WHITE, ec="none"))

    # ---- Footer note ----
    ax.text(XLIM / 2, 1.3,
            "Per-subject, per-modality logits cached in Drive so all "
            "fusion / coherence analysis is reproducible without re-running "
            "encoder training.",
            ha="center", va="center",
            fontsize=10, style="italic", color=GREY)

    plt.tight_layout()
    fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight", facecolor=BG_PALE)
    plt.close(fig)
    print(f"wrote {OUT_PATH}  ({OUT_PATH.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
