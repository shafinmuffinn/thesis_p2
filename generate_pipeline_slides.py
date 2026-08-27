"""Generate thesis_pipeline.pptx — visual pipeline diagrams.

Three slides:
  1. Training pipeline (full system, headline figure)
  2. Fusion module detail (zoom into the cross-attention block)
  3. Inference / demo pipeline (zero-EEG path)

All shapes are native PowerPoint primitives so you can edit them in PPT.
"""
from pptx import Presentation
from pptx.util import Cm, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from lxml import etree

OUT_PATH = "thesis_pipeline.pptx"

# 16:9 widescreen
SLIDE_W = Cm(33.87)
SLIDE_H = Cm(19.05)

# Colour palette (matches the rest of the deck)
NAVY        = RGBColor(0x1F, 0x3A, 0x68)
LIGHT_BLUE  = RGBColor(0xDC, 0xE6, 0xF1)
MID_BLUE    = RGBColor(0x4B, 0x6E, 0xA8)
PALE_BLUE   = RGBColor(0xEE, 0xF3, 0xFA)
GOLD        = RGBColor(0xC8, 0x96, 0x1B)
GOLD_BG     = RGBColor(0xFF, 0xF3, 0xCD)
GREEN       = RGBColor(0x2E, 0x7D, 0x32)
GREEN_BG    = RGBColor(0xD4, 0xED, 0xDA)
GREY        = RGBColor(0x55, 0x55, 0x55)
LIGHT_GREY  = RGBColor(0xEC, 0xEC, 0xEC)
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
RED         = RGBColor(0xB0, 0x1F, 0x1F)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def add_box(slide, x, y, w, h, text, *,
            shape=MSO_SHAPE.ROUNDED_RECTANGLE,
            fill=LIGHT_BLUE, line=NAVY, font_color=NAVY,
            size=10, bold=False, italic=False, line_pt=1.25):
    """Add a labelled box. All coords in cm."""
    box = slide.shapes.add_shape(shape, Cm(x), Cm(y), Cm(w), Cm(h))
    box.fill.solid()
    box.fill.fore_color.rgb = fill
    box.line.color.rgb = line
    box.line.width = Pt(line_pt)

    tf = box.text_frame
    tf.word_wrap = True
    tf.margin_left = Cm(0.15)
    tf.margin_right = Cm(0.15)
    tf.margin_top = Cm(0.05)
    tf.margin_bottom = Cm(0.05)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE

    lines = (text or " ").split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = line
        run.font.size = Pt(size)
        run.font.color.rgb = font_color
        run.font.bold = bold
        run.font.italic = italic
    return box


def add_text(slide, x, y, w, h, text, *,
             size=10, color=NAVY, bold=False, italic=False,
             align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    """Add a floating text box."""
    tb = slide.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Cm(0.05)
    tf.margin_right = Cm(0.05)
    tf.margin_top = Cm(0.05)
    tf.margin_bottom = Cm(0.05)
    tf.vertical_anchor = anchor

    lines = text.split("\n")
    for i, line in enumerate(lines):
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
    existing = ln.find(qn('a:tailEnd'))
    if existing is not None:
        ln.remove(existing)
    tailEnd = etree.SubElement(ln, qn('a:tailEnd'))
    tailEnd.set('type', 'triangle')
    tailEnd.set('w', 'med')
    tailEnd.set('len', 'med')


def add_arrow(slide, x1, y1, x2, y2, *, color=NAVY, width_pt=1.75):
    """Arrow from (x1,y1) to (x2,y2) in cm, with a triangle head."""
    line = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, Cm(x1), Cm(y1), Cm(x2), Cm(y2))
    line.line.color.rgb = color
    line.line.width = Pt(width_pt)
    _add_arrowhead(line)
    return line


def add_title_bar(slide, text):
    add_box(slide, 0.4, 0.3, 33.07, 0.95, text,
            shape=MSO_SHAPE.RECTANGLE,
            fill=NAVY, line=NAVY, font_color=WHITE,
            size=16, bold=True)


# --------------------------------------------------------------------------
# Slide 1 — Training Pipeline
# --------------------------------------------------------------------------

def build_slide_training(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank

    add_title_bar(slide, "Training Pipeline — Trimodal Emotion Recognition on EAV")

    # Dataset header
    add_box(slide, 0.4, 1.45, 33.07, 0.85,
            "EAV Dataset   ·   42 subjects × 5 emotions × 200 cue-based interactions   ·   Trial-aligned across modalities",
            shape=MSO_SHAPE.RECTANGLE,
            fill=GOLD_BG, line=GOLD, font_color=NAVY,
            size=11, bold=True)

    # Column geometry: 3 columns, centered
    col_w = 8.5
    col_a_x = 2.4
    col_b_x = 12.69
    col_c_x = 22.97
    col_a_cx = col_a_x + col_w / 2
    col_b_cx = col_b_x + col_w / 2
    col_c_cx = col_c_x + col_w / 2

    # --- Input row (y=2.55) ---
    y = 2.55
    add_box(slide, col_a_x, y, col_w, 1.15, "AUDIO\n16 kHz  ·  5 s clip",
            fill=LIGHT_BLUE, size=11, bold=True)
    add_box(slide, col_b_x, y, col_w, 1.15, "VIDEO\n25 fps  ·  5 s clip",
            fill=LIGHT_BLUE, size=11, bold=True)
    add_box(slide, col_c_x, y, col_w, 1.15, "EEG\n30 channels  ·  500 Hz",
            fill=LIGHT_BLUE, size=11, bold=True)

    # arrows down
    for cx in (col_a_cx, col_b_cx, col_c_cx):
        add_arrow(slide, cx, y + 1.15, cx, y + 1.45)

    # --- Preprocessing row (y=4.0) ---
    y = 4.0
    add_box(slide, col_a_x, y, col_w, 1.2,
            "AST feature extractor\nlog-mel spectrogram",
            fill=PALE_BLUE, size=10)
    add_box(slide, col_b_x, y, col_w, 1.2,
            "MTCNN face crop  +  HF ViT processor\n→ 224×224 frames",
            fill=PALE_BLUE, size=10)
    add_box(slide, col_c_x, y, col_w, 1.2,
            "Downsample 100 Hz\n+ 0.5–45 Hz bandpass  ·  5 s windows",
            fill=PALE_BLUE, size=10)

    for cx in (col_a_cx, col_b_cx, col_c_cx):
        add_arrow(slide, cx, y + 1.2, cx, y + 1.5)

    # --- Encoder row (y=5.5) ---
    y = 5.5
    add_box(slide, col_a_x, y, col_w, 1.55,
            "AST  (Audio Spectrogram Transformer)\nMIT/ast-finetuned-audioset\n→ 768-d pooled embedding",
            fill=MID_BLUE, font_color=WHITE, size=10, bold=True)
    add_box(slide, col_b_x, y, col_w, 1.55,
            "ViT  (Vision Transformer)\ndima806/facial_emotions_image_detection\n→ 768-d pooled embedding",
            fill=MID_BLUE, font_color=WHITE, size=10, bold=True)
    add_box(slide, col_c_x, y, col_w, 1.55,
            "EEGNet  (F1=8, D=8, F2=64)\ntrained from random init\n→ 960-d pooled embedding",
            fill=MID_BLUE, font_color=WHITE, size=10, bold=True)

    for cx in (col_a_cx, col_b_cx, col_c_cx):
        add_arrow(slide, cx, y + 1.55, cx, y + 1.85)

    # --- Projection band (y=7.4, full width) ---
    y = 7.4
    add_box(slide, col_a_x, y, col_c_x + col_w - col_a_x, 0.8,
            "Linear projection → D = 256 per modality   →   modality tokens  [A_tok, V_tok, E_tok]",
            shape=MSO_SHAPE.RECTANGLE,
            fill=LIGHT_GREY, font_color=NAVY, size=11, bold=True)

    # Arrow into fusion
    add_arrow(slide, 16.94, y + 0.8, 16.94, y + 1.15)

    # --- Fusion box (y=8.4, full width, h=3.0) ---
    fy, fh = 8.4, 3.0
    add_box(slide, col_a_x, fy, col_c_x + col_w - col_a_x, fh, " ",
            shape=MSO_SHAPE.ROUNDED_RECTANGLE,
            fill=LIGHT_BLUE, line=NAVY, line_pt=2.0)

    # Header inside fusion
    add_text(slide, col_a_x + 0.2, fy + 0.1,
             col_c_x + col_w - col_a_x - 0.4, 0.55,
             "TRIMODAL ATTENTION FUSION   ·   2.28 M trainable parameters",
             size=12, bold=True, color=NAVY, align=PP_ALIGN.CENTER)

    # Two sub-blocks inside fusion: dropout + self-attention
    fusion_inner_y = fy + 0.75
    add_box(slide, col_a_x + 0.4, fusion_inner_y, 13.5, 1.1,
            "Modality Dropout  (softhard)\nrandom 1-of-3 zeroed   ·   training only",
            fill=GOLD_BG, line=GOLD, font_color=NAVY,
            size=10, bold=True, italic=True)

    add_box(slide, col_a_x + 14.4, fusion_inner_y, 13.5, 1.1,
            "Self-Attention  ×  2 layers\n8 heads   ·   GELU   ·   pre-norm LayerNorm",
            fill=WHITE, line=NAVY, font_color=NAVY,
            size=10, bold=True)

    add_text(slide, col_a_x + 0.4, fy + 2.05,
             col_c_x + col_w - col_a_x - 0.8, 0.85,
             "outputs:  per-modality contextualised embeddings + fused embedding (D = 256)",
             size=10, italic=True, color=GREY, align=PP_ALIGN.CENTER)

    # --- Branching arrows from fusion to heads ---
    add_arrow(slide, 11.0, fy + fh, 8.5, fy + fh + 0.7)
    add_arrow(slide, 22.87, fy + fh, 25.37, fy + fh + 0.7)

    # --- Heads row (y=12.1) ---
    hy = 12.1
    add_box(slide, 3.5, hy, 10.0, 1.5,
            "MLP CLASSIFIER\n5-way softmax\nNeutral · Anger · Happiness · Sadness · Calmness",
            fill=GREEN_BG, line=GREEN, font_color=NAVY,
            size=10, bold=True)

    add_box(slide, 20.37, hy, 10.0, 1.5,
            "COHERENCE HEAD\npairwise symmetric KL on softmaxes\n→ 5×5 suppression matrix",
            fill=GREEN_BG, line=GREEN, font_color=NAVY,
            size=10, bold=True)

    add_arrow(slide, 8.5, hy + 1.5, 8.5, hy + 1.8)
    add_arrow(slide, 25.37, hy + 1.5, 25.37, hy + 1.8)

    # --- Outputs (y=13.9) ---
    oy = 13.9
    add_box(slide, 3.5, oy, 10.0, 1.85,
            "CLASSIFICATION  RESULTS\n80.2% cross-attention\n84.7% + modality dropout\n81.5% AV-only (zero-EEG demo)",
            shape=MSO_SHAPE.RECTANGLE,
            fill=NAVY, font_color=WHITE, line=NAVY,
            size=11, bold=True)

    add_box(slide, 20.37, oy, 10.0, 1.85,
            "COHERENCE  RESULTS\n401 suppression events / 5,040 trials\nAnger ↔ Happiness  ·  Sad/Calm → Neutral",
            shape=MSO_SHAPE.RECTANGLE,
            fill=NAVY, font_color=WHITE, line=NAVY,
            size=11, bold=True)

    # Footer stage labels
    add_text(slide, 0.4, 16.1, 33.07, 0.6,
             "Stage 1  Preprocessing   ·   Stage 2  Per-modality encoders   ·   Stage 3  Trimodal fusion   ·   Stage 4  Classification & coherence heads",
             size=9, italic=True, color=GREY, align=PP_ALIGN.CENTER)

    # Bottom footer
    add_text(slide, 0.4, 18.4, 33.07, 0.4,
             "EAV dataset: Lee et al., Scientific Data 2024  ·  Fusion architecture: this thesis (2026)",
             size=8, italic=True, color=GREY, align=PP_ALIGN.CENTER)


# --------------------------------------------------------------------------
# Slide 2 — Fusion Module Detail
# --------------------------------------------------------------------------

def build_slide_fusion_detail(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    add_title_bar(slide, "Trimodal Attention Fusion — Module Detail")

    # Left column: vertical stack of fusion blocks
    cx = 11.0   # column center for the stack
    bw = 12.0   # box width
    bx = cx - bw / 2

    # Top: 3 modality tokens as a row of pills
    pill_w, pill_h = 3.2, 1.0
    pill_y = 1.85
    for i, (label, tx) in enumerate([
        ("A_tok\n(256-d)", cx - 4.5),
        ("V_tok\n(256-d)", cx - pill_w / 2),
        ("E_tok\n(256-d)", cx + 1.3),
    ]):
        add_box(slide, tx, pill_y, pill_w, pill_h, label,
                fill=LIGHT_BLUE, line=NAVY,
                size=10, bold=True)

    # Three arrows merging downward into dropout
    add_arrow(slide, cx - 3.0, pill_y + pill_h, cx, pill_y + pill_h + 0.85)
    add_arrow(slide, cx,        pill_y + pill_h, cx, pill_y + pill_h + 0.85)
    add_arrow(slide, cx + 3.0, pill_y + pill_h, cx, pill_y + pill_h + 0.85)

    # Stack — vertical layout
    sy = pill_y + pill_h + 0.85
    stack = [
        ("Modality Dropout  (softhard, training only)\np_drop = 1/3   ·   one token zeroed per step", GOLD_BG, GOLD, 1.2, True),
        ("+ learnable modality embedding   ·   LayerNorm", LIGHT_GREY, NAVY, 0.9, False),
        ("Transformer Encoder Block  ×  1\nMulti-head SA (h=8)  →  residual + LN  →  FFN (D → 4D → D)  →  residual + LN", PALE_BLUE, NAVY, 1.4, False),
        ("Transformer Encoder Block  ×  2\nidentical structure", PALE_BLUE, NAVY, 1.0, False),
        ("Token-wise mean pool over 3 modalities", LIGHT_GREY, NAVY, 0.8, False),
        ("Fused embedding   ·   D = 256", NAVY, NAVY, 1.0, True),
    ]
    for text, fill, line, h, italic in stack:
        font_color = WHITE if fill == NAVY else NAVY
        add_box(slide, bx, sy, bw, h, text,
                fill=fill, line=line, font_color=font_color,
                size=10, bold=True, italic=italic)
        add_arrow(slide, cx, sy + h, cx, sy + h + 0.3)
        sy += h + 0.3

    # Remove the final arrow that points to nothing — we replace by output
    # (just overdraw with a label at the end)

    # Right side: design rationale callouts
    rx = 22.0  # right-column x
    rw = 11.4

    add_box(slide, rx, 1.85, rw, 0.65,
            "DESIGN RATIONALE",
            shape=MSO_SHAPE.RECTANGLE,
            fill=NAVY, font_color=WHITE,
            size=12, bold=True)

    callouts = [
        ("Why self-attention over 3 tokens?",
         "Per-modality encoders produce single pooled features. Pairwise cross-attention (6-way Lee/Kim/Kim CMA) "
         "degenerates to a learned projection at single-token resolution. Self-attention does the same work with "
         "fewer parameters."),
        ("Why softhard modality dropout?",
         "Trains the fusion to remain calibrated when one modality is zeroed at inference. Enables a working "
         "zero-EEG demo path on a recorded MP4 clip and adds +4.5 pp accuracy."),
        ("Why expose per-modality embeddings?",
         "Phase-2 temporal heads (LSTM / TCN / temporal transformer) can plug onto the per-modality outputs "
         "without architectural rewrite. forward(x_eeg=zeros) is a first-class call."),
    ]
    cy = 2.7
    for title, body in callouts:
        add_box(slide, rx, cy, rw, 0.6, title,
                shape=MSO_SHAPE.RECTANGLE,
                fill=GOLD_BG, line=GOLD, font_color=NAVY,
                size=10, bold=True)
        cy += 0.6
        add_box(slide, rx, cy, rw, 2.0, body,
                shape=MSO_SHAPE.RECTANGLE,
                fill=WHITE, line=GREY, font_color=NAVY,
                size=9.5)
        cy += 2.0 + 0.4

    # Footer
    add_text(slide, 0.4, 18.4, 33.07, 0.4,
             "TrimodalAttentionFusion — fusion/trimodal_attention.py  ·  D = 256  ·  2.28 M parameters",
             size=8, italic=True, color=GREY, align=PP_ALIGN.CENTER)


# --------------------------------------------------------------------------
# Slide 3 — Inference / Demo Pipeline
# --------------------------------------------------------------------------

def build_slide_inference(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    add_title_bar(slide, "Inference Pipeline — Zero-EEG Demo Path")

    # Top: input MP4
    add_box(slide, 12.0, 1.6, 9.87, 1.0,
            "Input  ·  recorded MP4 video clip",
            fill=GOLD_BG, line=GOLD, font_color=NAVY,
            size=12, bold=True, shape=MSO_SHAPE.RECTANGLE)
    add_arrow(slide, 16.94, 2.6, 16.94, 3.0)

    # Demuxing into 3 streams (one is empty)
    add_box(slide, 1.5, 3.0, 9.0, 1.1,
            "ffmpeg demux\n→ Audio stream (16 kHz)",
            fill=LIGHT_BLUE, size=10, bold=True)
    add_box(slide, 12.44, 3.0, 9.0, 1.1,
            "OpenCV frame read\n→ Video frames (25 fps)",
            fill=LIGHT_BLUE, size=10, bold=True)
    add_box(slide, 23.37, 3.0, 9.0, 1.1,
            "EEG SENSOR ABSENT\n→ zero tensor (B, 960)",
            fill=GOLD_BG, line=RED, font_color=RED,
            size=10, bold=True, italic=True)

    add_arrow(slide, 6.0, 4.1, 6.0, 4.5)
    add_arrow(slide, 16.94, 4.1, 16.94, 4.5)
    add_arrow(slide, 27.87, 4.1, 27.87, 4.5)

    # Sliding window band
    add_box(slide, 1.5, 4.5, 30.87, 0.85,
            "5-second sliding window  ·  1-second stride  ·  same preprocessing as training",
            shape=MSO_SHAPE.RECTANGLE,
            fill=LIGHT_GREY, font_color=NAVY, size=11, bold=True)

    add_arrow(slide, 16.94, 5.35, 16.94, 5.75)

    # Encoders (frozen)
    y = 5.75
    add_box(slide, 1.5, y, 9.0, 1.4,
            "AST  (frozen)\n→ 768-d audio embedding",
            fill=MID_BLUE, font_color=WHITE, size=10, bold=True)
    add_box(slide, 12.44, y, 9.0, 1.4,
            "ViT  (frozen)\n→ 768-d vision embedding",
            fill=MID_BLUE, font_color=WHITE, size=10, bold=True)
    add_box(slide, 23.37, y, 9.0, 1.4,
            "EEGNet  (frozen, receives zeros)\n→ 960-d zero-driven embedding",
            fill=GREY, font_color=WHITE, size=10, bold=True, italic=True)

    for cx in (6.0, 16.94, 27.87):
        add_arrow(slide, cx, y + 1.4, cx, y + 1.7)

    # Fusion (dropout-trained)
    add_box(slide, 1.5, 7.6, 30.87, 1.4,
            "TRIMODAL  ATTENTION  FUSION   ·   dropout-trained checkpoint   ·   zero-EEG token is in-distribution",
            shape=MSO_SHAPE.ROUNDED_RECTANGLE,
            fill=LIGHT_BLUE, line=NAVY, font_color=NAVY,
            size=12, bold=True, line_pt=2.0)

    add_arrow(slide, 16.94, 9.0, 16.94, 9.4)

    # Per-window prediction
    add_box(slide, 4.5, 9.4, 24.87, 1.2,
            "Per-window prediction  +  confidence  +  3-window median smoothing",
            shape=MSO_SHAPE.RECTANGLE,
            fill=GREEN_BG, line=GREEN, font_color=NAVY,
            size=11, bold=True)

    add_arrow(slide, 16.94, 10.6, 16.94, 11.0)

    # Overlay rendering
    add_box(slide, 4.5, 11.0, 24.87, 1.4,
            "OpenCV overlay  (emotion pill + confidence bar + timestamp)\n+  ffmpeg audio re-mux",
            fill=PALE_BLUE, font_color=NAVY, size=11, bold=True)

    add_arrow(slide, 16.94, 12.4, 16.94, 12.8)

    # Output
    add_box(slide, 4.5, 12.8, 24.87, 1.6,
            "Output  ·  MP4 with frame-level emotion overlay  ·  caption: \"EEG unavailable; prediction uses audio + video only\"",
            shape=MSO_SHAPE.RECTANGLE,
            fill=NAVY, font_color=WHITE,
            size=12, bold=True)

    # Side annotation: empirical numbers
    add_box(slide, 1.5, 14.7, 30.87, 1.6,
            "Empirical: dropout-trained AV-only inference reaches 81.5% on EAV's held-out trials, "
            "exceeding the dropout-untrained full-modality baseline (80.2%). "
            "Modality dropout is therefore not a robustness ablation — it is a first-class training requirement for the demo.",
            shape=MSO_SHAPE.RECTANGLE,
            fill=GOLD_BG, line=GOLD, font_color=NAVY,
            size=11, italic=True)

    # Footer
    add_text(slide, 0.4, 18.4, 33.07, 0.4,
             "demo_inference.py  →  emits list of (start_time, emotion, confidence)   ·   demo_overlay.py  →  renders the MP4",
             size=8, italic=True, color=GREY, align=PP_ALIGN.CENTER)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def main():
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    build_slide_training(prs)
    build_slide_fusion_detail(prs)
    build_slide_inference(prs)

    prs.save(OUT_PATH)
    print(f"Wrote {OUT_PATH}  ·  {len(prs.slides)} slides")
    for i, slide in enumerate(prs.slides, start=1):
        n_shapes = len(slide.shapes)
        print(f"  Slide {i}: {n_shapes} shapes")


if __name__ == "__main__":
    main()
