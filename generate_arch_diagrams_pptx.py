"""Generate thesis_arch_diagrams.pptx — four architecture diagrams as
native PowerPoint shapes (editable).

Four 16:9 widescreen slides, one per architecture:
  1. AST   (Audio Spectrogram Transformer)
  2. ViT   (facial-emotion Vision Transformer)
  3. EEGNet (depthwise separable CNN for EEG)
  4. TrimodalAttentionFusion (self-attention over 3 modality tokens)

Layout: left-column main vertical pipeline, right-column detail panel
(transformer per-layer structure or block expansions) + model card.
"""
from pptx import Presentation
from pptx.util import Cm, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

OUT_PATH = "thesis_arch_diagrams.pptx"

# 16:9 widescreen (matches thesis_slides.pptx)
SLIDE_W_CM = 33.87
SLIDE_H_CM = 19.05
SLIDE_W = Cm(SLIDE_W_CM)
SLIDE_H = Cm(SLIDE_H_CM)

# Palette
NAVY        = RGBColor(0x1A, 0x3A, 0x5C)
DEEP_NAVY   = RGBColor(0x0F, 0x25, 0x40)
LIGHT_BLUE  = RGBColor(0xD9, 0xE7, 0xF5)
MID_BLUE    = RGBColor(0x2E, 0x7E, 0xB8)
PALE_BLUE   = RGBColor(0xEE, 0xF3, 0xFA)
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
GOLD        = RGBColor(0xC8, 0x7F, 0x12)
GOLD_BG     = RGBColor(0xFD, 0xF1, 0xE0)
GREEN       = RGBColor(0x2E, 0x8B, 0x57)
GREEN_BG    = RGBColor(0xD4, 0xED, 0xDA)
GREY        = RGBColor(0x66, 0x66, 0x66)
LIGHT_GREY  = RGBColor(0xF2, 0xF2, 0xF2)
DARK        = RGBColor(0x20, 0x20, 0x20)
RED         = RGBColor(0xB0, 0x1F, 0x1F)
GOLD_PANEL  = RGBColor(0xFF, 0xF3, 0xCD)


# -------------------------------------------------------------------------
# Helpers (identical to previous version)
# -------------------------------------------------------------------------

def add_box(slide, x, y, w, h, text, *,
            shape=MSO_SHAPE.ROUNDED_RECTANGLE,
            fill=LIGHT_BLUE, line=NAVY, font_color=DARK,
            size=11, bold=False, italic=False, line_pt=1.25):
    box = slide.shapes.add_shape(shape, Cm(x), Cm(y), Cm(w), Cm(h))
    box.fill.solid()
    box.fill.fore_color.rgb = fill
    box.line.color.rgb = line
    box.line.width = Pt(line_pt)

    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Cm(0.12)
    tf.margin_right = Cm(0.12)
    tf.margin_top = Cm(0.05)
    tf.margin_bottom = Cm(0.05)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE

    for i, line in enumerate((text or " ").split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.color.rgb = font_color
        run.font.bold = bold
        run.font.italic = italic
    return box


def add_cbox(slide, cx, cy, w, h, text, **kwargs):
    return add_box(slide, cx - w / 2, cy - h / 2, w, h, text, **kwargs)


def add_text(slide, x, y, w, h, text, *,
             size=10, color=NAVY, bold=False, italic=False,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Cm(0.05)
    tf.margin_right = Cm(0.05)
    tf.margin_top = Cm(0.03)
    tf.margin_bottom = Cm(0.03)
    tf.vertical_anchor = anchor

    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.font.bold = bold
        run.font.italic = italic
    return tb


def _add_arrowhead(line_shape):
    sp = line_shape._element
    spPr = sp.find('.//' + qn('p:spPr'))
    if spPr is None:
        return
    ln = spPr.find(qn('a:ln'))
    if ln is None:
        ln = etree.SubElement(spPr, qn('a:ln'))
    for existing in ln.findall(qn('a:tailEnd')):
        ln.remove(existing)
    tailEnd = etree.SubElement(ln, qn('a:tailEnd'))
    tailEnd.set('type', 'triangle')
    tailEnd.set('w', 'med')
    tailEnd.set('len', 'med')


def add_arrow(slide, x1, y1, x2, y2, *, color=NAVY, width_pt=1.5):
    line = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Cm(x1), Cm(y1), Cm(x2), Cm(y2))
    line.line.color.rgb = color
    line.line.width = Pt(width_pt)
    _add_arrowhead(line)


def add_title_bar(slide, title, subtitle):
    add_box(slide, 0.3, 0.25, SLIDE_W_CM - 0.6, 1.0,
            title,
            shape=MSO_SHAPE.RECTANGLE,
            fill=NAVY, line=NAVY, font_color=WHITE,
            size=18, bold=True)
    add_text(slide, 0.3, 1.3, SLIDE_W_CM - 0.6, 0.45,
             subtitle, size=11, color=GREY, italic=True,
             align=PP_ALIGN.CENTER)


# -------------------------------------------------------------------------
# Layout constants for 16:9
# -------------------------------------------------------------------------

# Left column (main pipeline)
L_CX = 7.0           # left column centre x
L_W = 12.5           # left column block width
L_TOP = 2.0          # left column top y

# Right column (detail panel)
R_X = 14.5           # right column left edge
R_W = 18.7           # right column width
R_CX = R_X + R_W / 2 # right column centre x


def add_footer_strip(slide, lines):
    """Compact 3-line footer at the very bottom."""
    y = SLIDE_H_CM - 1.5
    add_box(slide, 0.3, y, SLIDE_W_CM - 0.6, 1.3, " ",
            shape=MSO_SHAPE.RECTANGLE,
            fill=LIGHT_GREY, line=GREY, line_pt=1.0)
    for i, (txt, bold) in enumerate(lines):
        yi = y + 0.1 + i * 0.4
        add_text(slide, 0.6, yi, SLIDE_W_CM - 1.2, 0.4, txt,
                 size=9.5, color=DARK, bold=bold,
                 align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)


# -------------------------------------------------------------------------
# Reusable: per-layer transformer detail panel (for AST/ViT)
# -------------------------------------------------------------------------

def add_transformer_detail_panel(slide, top_y, height, *,
                                 n_layers="12",
                                 heads="12",
                                 ffn_dims="768 → 3072 → 768"):
    """Right-column detail panel showing one transformer layer's structure."""
    add_box(slide, R_X, top_y, R_W, height, " ",
            fill=PALE_BLUE, line=NAVY, line_pt=2.0)

    add_text(slide, R_X, top_y + 0.15, R_W, 0.55,
             f"Transformer Encoder  ·  per-layer detail",
             size=13, color=NAVY, bold=True, align=PP_ALIGN.CENTER)
    add_text(slide, R_X, top_y + 0.7, R_W, 0.4,
             "(pre-LN order, repeated × " + n_layers + ")",
             size=10, color=GREY, italic=True, align=PP_ALIGN.CENTER)

    box_w = R_W - 2.5
    cx = R_CX

    y = top_y + 1.4
    add_cbox(slide, cx, y, box_w, 0.85,
             f"Multi-Head Self-Attention   ·   {heads} heads",
             fill=DEEP_NAVY, line=NAVY, font_color=WHITE,
             size=12, bold=True)
    add_arrow(slide, cx, y + 0.425, cx, y + 0.95)

    y += 1.15
    add_cbox(slide, cx, y, box_w, 0.7,
             "+ residual  →  LayerNorm",
             fill=LIGHT_GREY, line=GREY, size=10.5)
    add_arrow(slide, cx, y + 0.35, cx, y + 0.85)

    y += 1.0
    add_cbox(slide, cx, y, box_w, 0.85,
             f"FFN ( {ffn_dims} )   ·   GELU",
             fill=MID_BLUE, line=NAVY, font_color=WHITE,
             size=12, bold=True)
    add_arrow(slide, cx, y + 0.425, cx, y + 0.95)

    y += 1.15
    add_cbox(slide, cx, y, box_w, 0.7,
             "+ residual  →  LayerNorm",
             fill=LIGHT_GREY, line=GREY, size=10.5)

    add_text(slide, R_X, y + 1.0, R_W, 0.5,
             f"(stack of {n_layers} identical layers)",
             size=11, color=NAVY, bold=True, italic=True,
             align=PP_ALIGN.CENTER)


# -------------------------------------------------------------------------
# Slide 1: AST
# -------------------------------------------------------------------------

def build_ast(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, "AST  —  Audio Spectrogram Transformer",
                  "Audio modality encoder  ·  16:9 widescreen")

    # ---- Left column: main pipeline ----
    cx = L_CX

    y = L_TOP
    add_cbox(slide, cx, y + 0.45, L_W, 0.9,
             "INPUT  ·  5 s audio clip @ 16 kHz mono",
             shape=MSO_SHAPE.RECTANGLE,
             fill=GOLD_BG, line=GOLD, line_pt=2.5,
             size=12, bold=True, font_color=NAVY)
    add_arrow(slide, cx, y + 0.9, cx, y + 1.4)

    y += 1.55
    add_cbox(slide, cx, y + 0.5, L_W, 1.0,
             "AST feature extractor  (Hugging Face)\nLog-mel spectrogram, 128-bin",
             fill=LIGHT_GREY, line=GREY, size=10.5)
    add_arrow(slide, cx, y + 1.0, cx, y + 1.55)
    add_text(slide, cx + L_W / 2 - 4, y + 1.05, 4, 0.45,
             "(T × 128),  T ≈ 500",
             size=9, color=GREY, italic=True, align=PP_ALIGN.RIGHT)

    y += 1.7
    add_cbox(slide, cx, y + 0.5, L_W, 1.0,
             "Patch embedding\n16 × 16 patches, stride (10, 10)",
             fill=LIGHT_BLUE, line=MID_BLUE, size=10.5)
    add_arrow(slide, cx, y + 1.0, cx, y + 1.55)
    add_text(slide, cx + L_W / 2 - 4, y + 1.05, 4, 0.45,
             "(N_patches, 768)",
             size=9, color=GREY, italic=True, align=PP_ALIGN.RIGHT)

    y += 1.7
    add_cbox(slide, cx, y + 0.4, L_W, 0.8,
             "Prepend [CLS] token  +  position embeddings",
             fill=LIGHT_GREY, line=GREY, size=10.5)
    add_arrow(slide, cx, y + 0.8, cx, y + 1.35)
    add_text(slide, cx + L_W / 2 - 4, y + 0.85, 4, 0.45,
             "(1 + N_patches, 768)",
             size=9, color=GREY, italic=True, align=PP_ALIGN.RIGHT)

    y += 1.5
    add_cbox(slide, cx, y + 0.5, L_W, 1.0,
             "Transformer Encoder  ·  × 12 layers",
             fill=DEEP_NAVY, line=NAVY, font_color=WHITE,
             size=13, bold=True)
    add_text(slide, cx + L_W / 2 - 5, y + 1.05, 5, 0.45,
             "→ see right-side per-layer detail",
             size=9, color=GOLD, italic=True, bold=True, align=PP_ALIGN.RIGHT)
    add_arrow(slide, cx, y + 1.0, cx, y + 1.55)

    y += 1.7
    add_cbox(slide, cx, y + 0.4, L_W, 0.8,
             "Extract  [CLS]  token  only",
             fill=LIGHT_GREY, line=GREY, size=10.5)
    add_arrow(slide, cx, y + 0.8, cx, y + 1.35)
    add_text(slide, cx + L_W / 2 - 4, y + 0.85, 4, 0.45,
             "(768,)",
             size=9, color=GREY, italic=True, align=PP_ALIGN.RIGHT)

    y += 1.55
    add_cbox(slide, cx, y + 0.55, L_W + 1.0, 1.1,
             "AUDIO  EMBEDDING   ·   768-dim   →   to fusion",
             shape=MSO_SHAPE.RECTANGLE,
             fill=GOLD_BG, line=GOLD, line_pt=3.0,
             font_color=NAVY, size=14, bold=True)
    add_arrow(slide, cx, y + 1.1, cx, y + 1.65)

    y += 1.85
    add_cbox(slide, cx, y + 0.4, L_W, 0.8,
             "Linear ( 768 → 5 )  →  audio logits   (per-modality CE loss)",
             fill=GREEN_BG, line=GREEN, size=10.5)

    # ---- Right column: per-layer transformer detail ----
    add_transformer_detail_panel(slide, 2.0, 9.5,
                                 n_layers="12", heads="12",
                                 ffn_dims="768 → 3072 → 768")

    # ---- Right column model card (below detail panel) ----
    card_y = 12.0
    add_box(slide, R_X, card_y, R_W, 4.0, " ",
            shape=MSO_SHAPE.RECTANGLE,
            fill=LIGHT_GREY, line=GREY, line_pt=1.0)
    add_text(slide, R_X + 0.3, card_y + 0.2, R_W - 0.6, 0.5,
             "MODEL CARD", size=11, color=NAVY, bold=True,
             align=PP_ALIGN.CENTER)
    card_lines = [
        ("Checkpoint:", "MIT/ast-finetuned-audioset-10-10-0.4593"),
        ("Parameters:", "≈ 87 M"),
        ("Pre-training:", "AudioSet  (2 M clips, 527 classes)"),
        ("Fine-tune:", "5 frozen + 5 fine-tune epochs per subject"),
        ("Optimiser:", "Adam  ·  batch size 8"),
    ]
    for i, (label, value) in enumerate(card_lines):
        y_line = card_y + 0.8 + i * 0.55
        add_text(slide, R_X + 0.5, y_line, 4.0, 0.45, label,
                 size=10, color=NAVY, bold=True, align=PP_ALIGN.LEFT)
        add_text(slide, R_X + 4.5, y_line, R_W - 5.0, 0.45, value,
                 size=10, color=DARK, align=PP_ALIGN.LEFT)


# -------------------------------------------------------------------------
# Slide 2: ViT
# -------------------------------------------------------------------------

def build_vit(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, "ViT  —  Facial-Emotion Vision Transformer",
                  "Vision modality encoder  ·  16:9 widescreen")

    cx = L_CX

    y = L_TOP
    add_cbox(slide, cx, y + 0.45, L_W, 0.9,
             "INPUT  ·  5 s video, 25 frames @ 5 fps,  224 × 224 × 3",
             shape=MSO_SHAPE.RECTANGLE,
             fill=GOLD_BG, line=GOLD, line_pt=2.5,
             size=11.5, bold=True, font_color=NAVY)
    add_arrow(slide, cx, y + 0.9, cx, y + 1.4)
    add_text(slide, cx + L_W / 2 - 5, y + 0.95, 5, 0.45,
             "(25, 3, 224, 224)",
             size=9, color=GREY, italic=True, align=PP_ALIGN.RIGHT)

    y += 1.55
    add_text(slide, cx - L_W / 2, y + 0.25, L_W, 0.5,
             "── per-frame loop  ( × 25 frames ) ──",
             size=11, color=NAVY, bold=True, italic=True,
             align=PP_ALIGN.CENTER)
    add_arrow(slide, cx, y + 0.75, cx, y + 1.3)

    y += 1.45
    add_cbox(slide, cx, y + 0.4, L_W, 0.8,
             "HF image processor  ·  normalise (μ, σ) per channel",
             fill=LIGHT_GREY, line=GREY, size=10.5)
    add_arrow(slide, cx, y + 0.8, cx, y + 1.3)

    y += 1.45
    add_cbox(slide, cx, y + 0.5, L_W, 1.0,
             "Patch embedding  ·  16 × 16 patches  →  196 patches\n"
             "linear projection → 768-d per patch",
             fill=LIGHT_BLUE, line=MID_BLUE, size=10.5)
    add_arrow(slide, cx, y + 1.0, cx, y + 1.55)
    add_text(slide, cx + L_W / 2 - 4, y + 1.05, 4, 0.45,
             "(196, 768)",
             size=9, color=GREY, italic=True, align=PP_ALIGN.RIGHT)

    y += 1.7
    add_cbox(slide, cx, y + 0.4, L_W, 0.8,
             "Prepend [CLS]  +  position embeddings",
             fill=LIGHT_GREY, line=GREY, size=10.5)
    add_arrow(slide, cx, y + 0.8, cx, y + 1.35)

    y += 1.5
    add_cbox(slide, cx, y + 0.5, L_W, 1.0,
             "Transformer Encoder  ·  × 12 layers",
             fill=DEEP_NAVY, line=NAVY, font_color=WHITE,
             size=13, bold=True)
    add_text(slide, cx + L_W / 2 - 5, y + 1.05, 5, 0.45,
             "→ see right-side per-layer detail",
             size=9, color=GOLD, italic=True, bold=True, align=PP_ALIGN.RIGHT)
    add_arrow(slide, cx, y + 1.0, cx, y + 1.55)
    add_text(slide, cx + L_W / 2 - 4, y + 1.05, 4, 0.45,
             "[CLS] per frame",
             size=9, color=GREY, italic=True, align=PP_ALIGN.RIGHT)

    y += 1.7
    add_cbox(slide, cx, y + 0.4, L_W + 1.0, 0.85,
             "Mean-pool [CLS] tokens across  25 frames",
             fill=LIGHT_GREY, line=GREY, size=11, bold=True)
    add_arrow(slide, cx, y + 0.825, cx, y + 1.35)
    add_text(slide, cx + L_W / 2 - 4, y + 0.85, 4, 0.45,
             "(768,)",
             size=9, color=GREY, italic=True, align=PP_ALIGN.RIGHT)

    y += 1.5
    add_cbox(slide, cx, y + 0.55, L_W + 1.0, 1.1,
             "VISION  EMBEDDING   ·   768-dim   →   to fusion",
             shape=MSO_SHAPE.RECTANGLE,
             fill=GOLD_BG, line=GOLD, line_pt=3.0,
             font_color=NAVY, size=14, bold=True)

    # ---- Right column: per-layer transformer detail ----
    add_transformer_detail_panel(slide, 2.0, 9.5,
                                 n_layers="12", heads="12",
                                 ffn_dims="768 → 3072 → 768")

    # ---- Model card ----
    card_y = 12.0
    add_box(slide, R_X, card_y, R_W, 4.0, " ",
            shape=MSO_SHAPE.RECTANGLE,
            fill=LIGHT_GREY, line=GREY, line_pt=1.0)
    add_text(slide, R_X + 0.3, card_y + 0.2, R_W - 0.6, 0.5,
             "MODEL CARD", size=11, color=NAVY, bold=True,
             align=PP_ALIGN.CENTER)
    card_lines = [
        ("Checkpoint:", "dima806/facial_emotions_image_detection"),
        ("Parameters:", "≈ 86 M"),
        ("Pre-training:", "7-class facial expressions, re-head to 5"),
        ("Fine-tune:", "3 frozen + 2 fine-tune epochs per subject"),
        ("Per-clip:", "Bag-of-frames mean over 25 frames"),
    ]
    for i, (label, value) in enumerate(card_lines):
        y_line = card_y + 0.8 + i * 0.55
        add_text(slide, R_X + 0.5, y_line, 4.5, 0.45, label,
                 size=10, color=NAVY, bold=True, align=PP_ALIGN.LEFT)
        add_text(slide, R_X + 5.0, y_line, R_W - 5.5, 0.45, value,
                 size=10, color=DARK, align=PP_ALIGN.LEFT)


# -------------------------------------------------------------------------
# Slide 3: EEGNet
# -------------------------------------------------------------------------

def build_eegnet(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide, "EEGNet  —  Depthwise Separable CNN",
                  "EEG modality encoder  ·  16:9 widescreen")

    cx = L_CX

    # Left column: main flow with summary block 1 + block 2 boxes
    y = L_TOP
    add_cbox(slide, cx, y + 0.45, L_W, 0.9,
             "INPUT  ·  5 s EEG, 30 channels @ 100 Hz",
             shape=MSO_SHAPE.RECTANGLE,
             fill=GOLD_BG, line=GOLD, line_pt=2.5,
             size=12, bold=True, font_color=NAVY)
    add_text(slide, cx + L_W / 2 - 4, y + 0.95, 4, 0.45,
             "(1, 30, 500)",
             size=9, color=GREY, italic=True, align=PP_ALIGN.RIGHT)
    add_arrow(slide, cx, y + 0.9, cx, y + 1.4)

    y += 1.55
    add_cbox(slide, cx, y + 0.7, L_W, 1.4,
             "BLOCK 1   ·   Temporal + Spatial Filtering\n"
             "(see right-side expansion)",
             fill=DEEP_NAVY, line=NAVY, font_color=WHITE,
             size=12, bold=True)
    add_arrow(slide, cx, y + 1.4, cx, y + 1.9)
    add_text(slide, cx + L_W / 2 - 4, y + 1.4, 4, 0.45,
             "(64, 1, 125)",
             size=9, color=GREY, italic=True, align=PP_ALIGN.RIGHT)

    y += 2.05
    add_cbox(slide, cx, y + 0.7, L_W, 1.4,
             "BLOCK 2   ·   Pointwise Separable Conv\n"
             "(see right-side expansion)",
             fill=DEEP_NAVY, line=NAVY, font_color=WHITE,
             size=12, bold=True)
    add_arrow(slide, cx, y + 1.4, cx, y + 1.9)
    add_text(slide, cx + L_W / 2 - 4, y + 1.4, 4, 0.45,
             "(64, 1, 15)",
             size=9, color=GREY, italic=True, align=PP_ALIGN.RIGHT)

    y += 2.05
    add_cbox(slide, cx, y + 0.4, L_W, 0.85,
             "Flatten   ·   64 × 1 × 15  →  960",
             fill=LIGHT_GREY, line=GREY, size=11, bold=True)
    add_arrow(slide, cx, y + 0.825, cx, y + 1.35)
    add_text(slide, cx + L_W / 2 - 4, y + 0.85, 4, 0.45,
             "(960,)",
             size=9, color=GREY, italic=True, align=PP_ALIGN.RIGHT)

    y += 1.5
    add_cbox(slide, cx, y + 0.55, L_W + 1.0, 1.1,
             "EEG  EMBEDDING   ·   960-dim   →   to fusion",
             shape=MSO_SHAPE.RECTANGLE,
             fill=GOLD_BG, line=GOLD, line_pt=3.0,
             font_color=NAVY, size=14, bold=True)
    add_arrow(slide, cx, y + 1.1, cx, y + 1.65)

    y += 1.85
    add_cbox(slide, cx, y + 0.4, L_W, 0.8,
             "Linear ( 960 → 5 )  →  EEG logits",
             fill=GREEN_BG, line=GREEN, size=10.5)

    # ---- Right column: Block 1 expansion ----
    b1_top = 2.0
    b1_h = 6.2
    add_box(slide, R_X, b1_top, R_W, b1_h, " ",
            fill=PALE_BLUE, line=NAVY, line_pt=2.0)
    add_text(slide, R_X, b1_top + 0.15, R_W, 0.5,
             "BLOCK 1   ·   Temporal + Spatial Filtering",
             size=12, color=NAVY, bold=True, align=PP_ALIGN.CENTER)

    box_w = R_W - 2.0
    cxr = R_CX
    yr = b1_top + 0.85
    add_cbox(slide, cxr, yr, box_w, 0.65,
             "Conv2D ( 1 → 8,  kernel = (1, 300),  pad = same )",
             fill=LIGHT_BLUE, line=MID_BLUE, size=10)
    add_arrow(slide, cxr, yr + 0.325, cxr, yr + 0.75)
    yr += 0.9
    add_cbox(slide, cxr, yr, box_w, 0.55,
             "BatchNorm2D ( 8 )  →  ELU",
             fill=LIGHT_GREY, line=GREY, size=9.5)
    add_arrow(slide, cxr, yr + 0.275, cxr, yr + 0.75)
    yr += 0.9
    add_cbox(slide, cxr, yr, box_w, 0.65,
             "DepthwiseConv2D ( 8 → 64,  kernel = (30, 1),  groups = 8 )",
             fill=LIGHT_BLUE, line=MID_BLUE, size=10)
    add_arrow(slide, cxr, yr + 0.325, cxr, yr + 0.75)
    yr += 0.9
    add_cbox(slide, cxr, yr, box_w, 0.55,
             "BatchNorm2D ( 64 )  →  ELU",
             fill=LIGHT_GREY, line=GREY, size=9.5)
    add_arrow(slide, cxr, yr + 0.275, cxr, yr + 0.75)
    yr += 0.9
    add_cbox(slide, cxr, yr, box_w, 0.55,
             "AvgPool2D ( (1, 4) )  →  Dropout ( 0.5 )",
             fill=LIGHT_GREY, line=GREY, size=9.5)

    # ---- Right column: Block 2 expansion ----
    b2_top = 8.5
    b2_h = 4.4
    add_box(slide, R_X, b2_top, R_W, b2_h, " ",
            fill=PALE_BLUE, line=NAVY, line_pt=2.0)
    add_text(slide, R_X, b2_top + 0.15, R_W, 0.5,
             "BLOCK 2   ·   Pointwise Separable Conv",
             size=12, color=NAVY, bold=True, align=PP_ALIGN.CENTER)

    yr = b2_top + 0.85
    add_cbox(slide, cxr, yr, box_w, 0.65,
             "Conv2D ( 64 → 64,  kernel = (1, 16),  pad = same )",
             fill=LIGHT_BLUE, line=MID_BLUE, size=10)
    add_arrow(slide, cxr, yr + 0.325, cxr, yr + 0.75)
    yr += 0.9
    add_cbox(slide, cxr, yr, box_w, 0.55,
             "BatchNorm2D ( 64 )  →  ELU",
             fill=LIGHT_GREY, line=GREY, size=9.5)
    add_arrow(slide, cxr, yr + 0.275, cxr, yr + 0.75)
    yr += 0.9
    add_cbox(slide, cxr, yr, box_w, 0.55,
             "AvgPool2D ( (1, 8) )  →  Dropout ( 0.5 )",
             fill=LIGHT_GREY, line=GREY, size=9.5)

    # ---- Model card ----
    card_y = 13.4
    add_box(slide, R_X, card_y, R_W, 4.0, " ",
            shape=MSO_SHAPE.RECTANGLE,
            fill=LIGHT_GREY, line=GREY, line_pt=1.0)
    add_text(slide, R_X + 0.3, card_y + 0.15, R_W - 0.6, 0.45,
             "MODEL CARD", size=11, color=NAVY, bold=True,
             align=PP_ALIGN.CENTER)
    card_lines = [
        ("Architecture:", "EEGNet  ·  F1=8, D=8, F2=64, kernLength=300"),
        ("Parameters:", "≈ 2.2 M  ·  trained from random init"),
        ("Training:", "350 epochs per subject  ·  Adam, lr=1e-4"),
        ("Regularisation:", "Dropout 0.5  ·  L2 max-norm hooks"),
    ]
    for i, (label, value) in enumerate(card_lines):
        y_line = card_y + 0.7 + i * 0.5
        add_text(slide, R_X + 0.5, y_line, 4.5, 0.45, label,
                 size=10, color=NAVY, bold=True, align=PP_ALIGN.LEFT)
        add_text(slide, R_X + 5.0, y_line, R_W - 5.5, 0.45, value,
                 size=10, color=DARK, align=PP_ALIGN.LEFT)


# -------------------------------------------------------------------------
# Slide 4: TrimodalAttentionFusion
# -------------------------------------------------------------------------

def build_fusion(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title_bar(slide,
                  "TrimodalAttentionFusion  —  Self-Attention over Modality Tokens",
                  "Trimodal fusion module  ·  2.28 M trainable parameters")

    # Three input column centres
    cx_a = 5.5
    cx_v = 13.0
    cx_e = 20.5
    col_w = 6.5
    cx_main = (cx_a + cx_e) / 2  # = 13.0

    # ===== Top: 3 input streams =====
    y = 2.2
    for cxn, label in [
        (cx_a, "x_audio   from AST\n(768,)"),
        (cx_v, "x_vision   from ViT\n(768,)"),
        (cx_e, "x_eeg   from EEGNet\n(960,)")]:
        add_cbox(slide, cxn, y + 0.5, col_w, 1.0,
                 label,
                 shape=MSO_SHAPE.RECTANGLE,
                 fill=GOLD_BG, line=GOLD, line_pt=2,
                 size=11, bold=True, font_color=NAVY)
        add_arrow(slide, cxn, y + 1.0, cxn, y + 1.45)

    # Stage 1 — Projection
    y += 1.5
    for cxn, label in [
        (cx_a, "Linear ( 768 → 256 )"),
        (cx_v, "Linear ( 768 → 256 )"),
        (cx_e, "Linear ( 960 → 256 )")]:
        add_cbox(slide, cxn, y + 0.4, col_w, 0.8,
                 label, fill=LIGHT_BLUE, line=MID_BLUE,
                 size=11, bold=True)
        add_arrow(slide, cxn, y + 0.8, cxn, y + 1.25)
    add_text(slide, 0.4, y + 0.2, 4.6, 0.7,
             "Stage 1\nProjection",
             size=10, color=NAVY, bold=True,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE)

    # Stage 2 — Modality embedding
    y += 1.35
    for cxn, label in [
        (cx_a, "+ modality_embed[0]\nLayerNorm"),
        (cx_v, "+ modality_embed[1]\nLayerNorm"),
        (cx_e, "+ modality_embed[2]\nLayerNorm")]:
        add_cbox(slide, cxn, y + 0.5, col_w, 1.0,
                 label, fill=LIGHT_GREY, line=GREY, size=9.5)
    add_text(slide, 0.4, y + 0.2, 4.6, 0.85,
             "Stage 2\nModality\nembedding",
             size=10, color=NAVY, bold=True,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.MIDDLE)

    # Three converge into stacked sequence
    for cxn in (cx_a, cx_v, cx_e):
        add_arrow(slide, cxn, y + 1.0, cx_main, y + 1.7)

    # Stack into sequence
    y += 1.85
    add_cbox(slide, cx_main, y + 0.4, 26.0, 0.8,
             "Stack into sequence  ·  tokens = [audio_tok, vision_tok, eeg_tok]  ·  shape (B, 3, 256)",
             fill=LIGHT_GREY, line=NAVY, size=11, bold=True)
    add_arrow(slide, cx_main, y + 0.8, cx_main, y + 1.25)

    # Stage 2.5: Softhard Dropout
    y += 1.35
    add_cbox(slide, cx_main, y + 0.55, 26.0, 1.1,
             "Softhard Modality Dropout  ·  (training only)\n"
             "with p = 1/3   ·   pick m ∈ {audio, vision, eeg}   ·   zero its token",
             fill=GOLD_PANEL, line=GOLD, line_pt=2,
             size=11, bold=True)
    add_text(slide, cx_main + 13.5, y + 0.55, 4, 0.9,
             "Stage 2.5\nrobustness",
             size=10, color=GOLD, italic=True, bold=True,
             anchor=MSO_ANCHOR.MIDDLE)
    add_arrow(slide, cx_main, y + 1.1, cx_main, y + 1.55)

    # Stage 3: Transformer Encoder × 2 (compact, horizontal sub-layers)
    y += 1.65
    add_cbox(slide, cx_main, y + 1.05, 28.0, 2.1, " ",
             fill=PALE_BLUE, line=NAVY, line_pt=2.0)
    add_text(slide, 3.0, y + 0.15, 26.0, 0.5,
             "Stage 3   ·   Transformer Encoder   ·   × 2 layers   (pre-LN)",
             size=12, color=NAVY, bold=True, align=PP_ALIGN.CENTER)

    # Inside: a horizontal mini-pipeline
    sub_y = y + 0.85
    sub_x = 4.5
    sub_block_w = 5.2
    sub_h = 1.0
    sub_gap = 0.7

    add_cbox(slide, sub_x + sub_block_w / 2, sub_y + sub_h / 2,
             sub_block_w, sub_h,
             "LayerNorm  →\nMulti-Head SA\n8 heads, d = 32",
             fill=DEEP_NAVY, line=NAVY, font_color=WHITE,
             size=10, bold=True)
    sub_x += sub_block_w
    add_arrow(slide, sub_x, sub_y + sub_h / 2, sub_x + sub_gap,
              sub_y + sub_h / 2)
    sub_x += sub_gap

    add_cbox(slide, sub_x + sub_block_w / 2, sub_y + sub_h / 2,
             sub_block_w, sub_h,
             "+ residual",
             fill=LIGHT_GREY, line=GREY, size=10)
    sub_x += sub_block_w
    add_arrow(slide, sub_x, sub_y + sub_h / 2, sub_x + sub_gap,
              sub_y + sub_h / 2)
    sub_x += sub_gap

    add_cbox(slide, sub_x + sub_block_w / 2, sub_y + sub_h / 2,
             sub_block_w, sub_h,
             "LayerNorm  →\nFFN ( 256 → 1024 → 256 )\nGELU",
             fill=MID_BLUE, line=NAVY, font_color=WHITE,
             size=10, bold=True)
    sub_x += sub_block_w
    add_arrow(slide, sub_x, sub_y + sub_h / 2, sub_x + sub_gap,
              sub_y + sub_h / 2)
    sub_x += sub_gap

    add_cbox(slide, sub_x + sub_block_w / 2, sub_y + sub_h / 2,
             sub_block_w, sub_h,
             "+ residual",
             fill=LIGHT_GREY, line=GREY, size=10)

    add_arrow(slide, cx_main, y + 2.1, cx_main, y + 2.6)

    # Stage 4 + Stage 5 + Output (compact bottom row)
    y += 2.85
    add_cbox(slide, cx_main, y + 0.4, 28.0, 0.8,
             "Stage 4   ·   Extract per-modality embeddings  +  mean-pool fused embedding (B, 256)",
             fill=LIGHT_GREY, line=NAVY, size=10.5)
    add_arrow(slide, cx_main, y + 0.8, cx_main, y + 1.25)

    y += 1.4
    add_cbox(slide, cx_main, y + 0.5, 16.0, 1.0,
             "FUSED  EMBEDDING   ·   256-dim",
             shape=MSO_SHAPE.RECTANGLE,
             fill=GOLD_BG, line=GOLD, line_pt=3.0,
             font_color=NAVY, size=14, bold=True)
    add_arrow(slide, cx_main, y + 1.0, cx_main, y + 1.45)

    y += 1.55
    add_cbox(slide, cx_main, y + 0.4, 28.0, 0.8,
             "Stage 5   ·   MLP head   Linear(256 → 256) → GELU → Dropout(0.1) → Linear(256 → 5)   →   logits (B, 5)",
             fill=GREEN_BG, line=GREEN, size=11, bold=True)


# -------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------

def main():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    build_ast(prs)
    build_vit(prs)
    build_eegnet(prs)
    build_fusion(prs)

    prs.save(OUT_PATH)
    print(f"wrote {OUT_PATH}  ·  {len(prs.slides)} slides "
          f"({SLIDE_W_CM:.2f} × {SLIDE_H_CM:.2f} cm  16:9 widescreen)")
    for i, slide in enumerate(prs.slides, start=1):
        print(f"  Slide {i}: {len(slide.shapes)} shapes")


if __name__ == "__main__":
    main()
