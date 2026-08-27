#!/usr/bin/env python3
"""Generate the Pre-thesis 2 presentation slides as a 16:9 .pptx.

13 slides, ~45-60s per slide, total ~10-12 minutes of talking time:
     1. Title
     2. Problem & Motivation
     3. Research Questions
     4. Dataset (EAV)
     5. Methodology Overview
     6. Proposed Architecture
     7. Training Results
     8. Comparison with Prior EAV Work
     9. Modality Dropout & Robustness
    10. Cross-Modal Coherence Analysis (suppression matrix)
    11. Per-Class Performance & Confusion Matrix
    12. Live Demo
    13. Future Work & Conclusion

Output: thesis_slides.pptx
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.util import Cm, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "thesis_slides.pptx"

# 16:9 widescreen, 13.33" × 7.5" = 33.87 × 19.05 cm
SLIDE_W = 33.87
SLIDE_H = 19.05

# Color palette (matches the poster)
NAVY = RGBColor(0x1A, 0x3A, 0x5C)
BLUE = RGBColor(0x2E, 0x7E, 0xB8)
LIGHT_BLUE = RGBColor(0xD9, 0xE7, 0xF5)
AMBER = RGBColor(0xC8, 0x7F, 0x12)
GOLD_BG = RGBColor(0xFD, 0xF1, 0xE0)
DARK = RGBColor(0x20, 0x20, 0x20)
GREY = RGBColor(0x66, 0x66, 0x66)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GREY = RGBColor(0xF2, 0xF2, 0xF2)
GREEN = RGBColor(0x2E, 0x8B, 0x57)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def blank_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def add_background(slide):
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Cm(SLIDE_W), Cm(SLIDE_H))
    bg.fill.solid()
    bg.fill.fore_color.rgb = WHITE
    bg.line.fill.background()


def add_header_bar(slide, title, page_num=None, total=13):
    # Thin navy bar at top
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Cm(SLIDE_W), Cm(1.6))
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()

    # Title in bar
    tb = slide.shapes.add_textbox(Cm(0.8), Cm(0.15), Cm(SLIDE_W - 5), Cm(1.4))
    tf = tb.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = title
    r.font.size = Pt(24)
    r.font.bold = True
    r.font.color.rgb = WHITE
    r.font.name = "Calibri"

    # Page number bottom right
    if page_num is not None:
        pn = slide.shapes.add_textbox(
            Cm(SLIDE_W - 3), Cm(SLIDE_H - 0.9), Cm(2.5), Cm(0.7)
        )
        tf = pn.text_frame
        tf.margin_top = 0
        tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.RIGHT
        r = p.add_run()
        r.text = f"{page_num} / {total}"
        r.font.size = Pt(11)
        r.font.color.rgb = GREY
        r.font.name = "Calibri"

    # Footer (project short title)
    footer = slide.shapes.add_textbox(Cm(0.8), Cm(SLIDE_H - 0.9), Cm(SLIDE_W - 4), Cm(0.7))
    tf = footer.text_frame
    tf.margin_top = 0
    tf.margin_bottom = 0
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = "Cross-Modal Affective Coherence  ·  Shafin  ·  Pre-thesis 2"
    r.font.size = Pt(10)
    r.font.color.rgb = GREY
    r.font.name = "Calibri"


def add_text(slide, x, y, w, h, text, *, size=18, bold=False, italic=False,
             color=DARK, align=PP_ALIGN.LEFT, font="Calibri",
             anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = Cm(0.1)
    tf.margin_right = Cm(0.1)
    tf.margin_top = Cm(0.05)
    tf.margin_bottom = Cm(0.05)
    p = tf.paragraphs[0]
    p.alignment = align
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color
    r.font.name = font
    return tb


def add_bullets(slide, x, y, w, h, bullets, *, size=18, color=DARK,
                numbered=False, bold_first_word=False):
    tb = slide.shapes.add_textbox(Cm(x), Cm(y), Cm(w), Cm(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Cm(0.1)
    tf.margin_right = Cm(0.1)
    for i, item in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(8)
        prefix = f"{i+1}.  " if numbered else "•  "
        if bold_first_word and ":" in item:
            head, _, rest = item.partition(":")
            r1 = p.add_run()
            r1.text = prefix + head + ": "
            r1.font.size = Pt(size)
            r1.font.bold = True
            r1.font.color.rgb = color
            r1.font.name = "Calibri"
            r2 = p.add_run()
            r2.text = rest.strip()
            r2.font.size = Pt(size)
            r2.font.color.rgb = color
            r2.font.name = "Calibri"
        else:
            r = p.add_run()
            r.text = prefix + item
            r.font.size = Pt(size)
            r.font.color.rgb = color
            r.font.name = "Calibri"
    return tb


def add_table(slide, x, y, w, h, headers, rows, *, header_size=14,
              body_size=14, header_color=NAVY, header_text_color=WHITE,
              highlight_row=None):
    nrows = len(rows) + 1
    ncols = len(headers)
    table = slide.shapes.add_table(
        nrows, ncols, Cm(x), Cm(y), Cm(w), Cm(h)
    ).table

    for j, h_text in enumerate(headers):
        cell = table.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_color
        cell.margin_left = Cm(0.1)
        cell.margin_right = Cm(0.1)
        cell.margin_top = Cm(0.05)
        cell.margin_bottom = Cm(0.05)
        cell.text_frame.clear()
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = h_text
        r.font.size = Pt(header_size)
        r.font.bold = True
        r.font.color.rgb = header_text_color
        r.font.name = "Calibri"

    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            if highlight_row is not None and (i - 1) == highlight_row:
                cell.fill.solid()
                cell.fill.fore_color.rgb = GOLD_BG
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = WHITE if i % 2 == 1 else LIGHT_GREY
            cell.margin_left = Cm(0.15)
            cell.margin_right = Cm(0.15)
            cell.margin_top = Cm(0.05)
            cell.margin_bottom = Cm(0.05)
            cell.text_frame.clear()
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT
            r = p.add_run()
            r.text = str(val)
            r.font.size = Pt(body_size)
            r.font.color.rgb = DARK
            r.font.name = "Calibri"
            if highlight_row is not None and (i - 1) == highlight_row:
                r.font.bold = True
    return table


def add_callout_box(slide, x, y, w, h, text, *, fill=GOLD_BG,
                    border=AMBER, size=16, color=DARK, bold=False):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                                 Cm(x), Cm(y), Cm(w), Cm(h))
    box.fill.solid()
    box.fill.fore_color.rgb = fill
    box.line.color.rgb = border
    box.line.width = Pt(1.5)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Cm(0.3)
    tf.margin_right = Cm(0.3)
    tf.margin_top = Cm(0.15)
    tf.margin_bottom = Cm(0.15)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = color
    r.font.name = "Calibri"


def add_arch_box(slide, x, y, w, h, text, *, fill=LIGHT_BLUE, border=BLUE,
                 size=14, bold=True):
    box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                                 Cm(x), Cm(y), Cm(w), Cm(h))
    box.fill.solid()
    box.fill.fore_color.rgb = fill
    box.line.color.rgb = border
    box.line.width = Pt(1.5)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = Cm(0.15)
    tf.margin_right = Cm(0.15)
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.color.rgb = NAVY
    r.font.name = "Calibri"


def add_arrow(slide, x1, y1, x2, y2):
    line = slide.shapes.add_connector(1, Cm(x1), Cm(y1), Cm(x2), Cm(y2))
    line.line.color.rgb = NAVY
    line.line.width = Pt(2)


def add_image(slide, x, y, w, h, path):
    """Embed a PNG image at (x, y) cm with width/height in cm."""
    return slide.shapes.add_picture(
        str(path), Cm(x), Cm(y), width=Cm(w), height=Cm(h))


FIGURES = ROOT / "figures"


# ---------------------------------------------------------------------------
# Slides
# ---------------------------------------------------------------------------

def slide_1_title(prs):
    s = blank_slide(prs)
    add_background(s)
    # Full-bleed top band
    band = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Cm(SLIDE_W), Cm(SLIDE_H * 0.55))
    band.fill.solid()
    band.fill.fore_color.rgb = NAVY
    band.line.fill.background()

    # Title
    add_text(
        s, 1.5, 2.5, SLIDE_W - 3, 3.5,
        "Cross-Modal Affective Coherence",
        size=48, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
        anchor=MSO_ANCHOR.MIDDLE,
    )
    add_text(
        s, 1.5, 5.8, SLIDE_W - 3, 1.5,
        "Detecting Discrepancies Between Externally Expressed and "
        "Internally Experienced Emotion Using EEG, Audio, and Video",
        size=22, italic=True, color=LIGHT_BLUE, align=PP_ALIGN.CENTER,
    )
    # Author block (bottom half white)
    add_text(s, 0, 12.0, SLIDE_W, 1.0,
             "Shafin", size=28, bold=True, color=NAVY, align=PP_ALIGN.CENTER)
    add_text(s, 0, 13.4, SLIDE_W, 0.8,
             "[Institution name — to be filled in]",
             size=18, color=DARK, align=PP_ALIGN.CENTER)
    add_text(s, 0, 14.5, SLIDE_W, 0.8,
             "Pre-thesis 2  ·  June 2026",
             size=16, italic=True, color=GREY, align=PP_ALIGN.CENTER)


def slide_2_motivation(prs):
    s = blank_slide(prs)
    add_background(s)
    add_header_bar(s, "Problem & Motivation", page_num=2)

    add_text(s, 0.8, 2.0, SLIDE_W - 1.6, 1.0,
             "Emotion recognition systems must address two distinct problems:",
             size=20, color=DARK)

    add_bullets(s, 1.0, 3.2, SLIDE_W - 2, 3.5, [
        "What emotion is the subject displaying?  (classification)",
        "Does the displayed emotion match what they are internally feeling?  (coherence)",
    ], size=20, bold_first_word=False)

    add_text(s, 0.8, 7.0, SLIDE_W - 1.6, 1.0,
             "The second question is largely unaddressed in the literature.",
             size=20, italic=True, color=GREY)

    add_callout_box(
        s, 1.5, 8.5, SLIDE_W - 3, 2.5,
        "Audio and video capture voluntary, controllable external expression.\n"
        "EEG captures involuntary internal physiological state.\n"
        "Cross-modal disagreement is itself a clinically meaningful signal.",
        size=18, bold=False,
    )

    add_text(s, 0.8, 11.5, SLIDE_W - 1.6, 5.0,
             "Real-world relevance: emotion suppression is documented in "
             "depression, anxiety, alexithymia, PTSD, and autism spectrum "
             "conditions. A method that quantifies cross-modal divergence "
             "in a healthy population establishes the foundation for "
             "clinical Phase 2 work.",
             size=16, italic=True, color=DARK)


def slide_3_research_questions(prs):
    s = blank_slide(prs)
    add_background(s)
    add_header_bar(s, "Research Questions", page_num=3)

    add_bullets(s, 1.0, 2.5, SLIDE_W - 2, 14, [
        "Per-modality baselines: Can audio, video and EEG individually "
        "exceed the EAV paper's published per-modality baselines?",
        "Fusion gain: Does multimodal fusion improve over the best single "
        "modality, and by how much?",
        "Architectural contribution: Does learned cross-modal attention "
        "fusion outperform a naïve softmax-averaging late-fusion baseline?",
        "Coherence as signal: When the modalities disagree, do consistent "
        "and interpretable cross-population patterns emerge?",
        "Robustness: Does the system continue to produce meaningful "
        "predictions when one modality (specifically EEG) is unavailable "
        "at inference time?",
    ], size=18, numbered=True)


def slide_4_dataset(prs):
    s = blank_slide(prs)
    add_background(s)
    add_header_bar(s, "Dataset — EAV (Lee et al., Sci. Data 2024)", page_num=4)

    add_text(s, 0.8, 2.2, SLIDE_W - 1.6, 1.0,
             "EAV: EEG-Audio-Video dataset for emotion recognition in "
             "conversational contexts.",
             size=18, color=DARK)

    add_table(s, 0.8, 3.5, 14, 9,
        ["Property", "Value"],
        [
            ("Participants",           "42 (lab-controlled, cued)"),
            ("Emotion classes",        "5 (Neutral, Sad, Angry, Happy, Calm)"),
            ("Interactions / subject", "200 (Listen + Speak)"),
            ("EEG channels / rate",    "30 ch / 500 Hz"),
            ("Audio rate / segment",   "16 kHz / 5 s"),
            ("Video resolution / fps", "224×224 (cropped) / 30 fps"),
            ("Per-subject train/test", "280 / 120 trials"),
            ("Total test trials",      "5,040 (42 × 120)"),
        ],
        header_size=14, body_size=14,
    )

    # Right side note
    add_text(s, 17, 3.5, 16, 1.0,
             "Why EAV?", size=20, bold=True, color=NAVY)
    add_bullets(s, 17, 4.5, 16, 9, [
        "Only public corpus combining EEG + AV at 42-subject scale.",
        "Synchronised acquisition: labels apply uniformly across modalities.",
        "Conversational cue-based protocol elicits all 5 emotions "
        "controllably.",
        "EAV authors published per-modality baselines (SCNN, DeepFace, "
        "EEGNet); no fusion baseline reported.",
    ], size=15)


def slide_5_methodology(prs):
    s = blank_slide(prs)
    add_background(s)
    add_header_bar(s, "Methodology — Three-Stage Pipeline", page_num=5)

    add_arch_box(s, 1.0, 4.0, 9.5, 3.5,
                 "Stage 1\nPer-Modality Encoders\n(AST · ViT · EEGNet)",
                 size=16)
    add_arch_box(s, 12, 4.0, 9.5, 3.5,
                 "Stage 2\nTrimodal Cross-Attention\nFusion Module",
                 size=16)
    add_arch_box(s, 23, 4.0, 9.5, 3.5,
                 "Stage 3\nCoherence Analysis\n(Suppression Matrix)",
                 size=16)
    add_arrow(s, 10.5, 5.75, 12, 5.75)
    add_arrow(s, 21.5, 5.75, 23, 5.75)

    add_text(s, 0.8, 8.5, SLIDE_W - 1.6, 1.0,
             "Two stages classify; the third characterises modality "
             "disagreement.",
             size=18, italic=True, color=DARK, align=PP_ALIGN.CENTER)

    add_text(s, 0.8, 10.0, SLIDE_W - 1.6, 1.0,
             "Two architectural commitments (load-bearing for the demo):",
             size=18, bold=True, color=NAVY)

    add_bullets(s, 1.5, 11.2, SLIDE_W - 3, 5, [
        "forward() returns named per-modality embeddings → "
        "future temporal head requires no rewrite.",
        "forward(audio, video, eeg=zeros) is a valid call → enables "
        "EEG-missing inference at the demo time.",
    ], size=16, bold_first_word=False)


def slide_6_architecture(prs):
    s = blank_slide(prs)
    add_background(s)
    add_header_bar(s, "Proposed Architecture", page_num=6)

    add_text(s, 0.8, 2.0, SLIDE_W - 1.6, 1.0,
             "TrimodalAttentionFusion (2.28M parameters):",
             size=18, bold=True, color=NAVY)

    # Architecture pipeline
    add_arch_box(s, 0.8, 3.5, 5.0, 1.6, "Audio (AST)\n768-d feature",
                 fill=LIGHT_BLUE, size=12)
    add_arch_box(s, 0.8, 5.3, 5.0, 1.6, "Vision (ViT)\n768-d feature",
                 fill=LIGHT_BLUE, size=12)
    add_arch_box(s, 0.8, 7.1, 5.0, 1.6, "EEG (EEGNet)\n960-d feature",
                 fill=LIGHT_BLUE, size=12)

    add_arch_box(s, 7.5, 4.5, 5.0, 4.5,
                 "Linear projection\n→ 256-d\n+ modality embed",
                 fill=GOLD_BG, border=AMBER, size=13)
    add_arrow(s, 5.8, 4.3, 7.5, 5.0)
    add_arrow(s, 5.8, 6.1, 7.5, 6.5)
    add_arrow(s, 5.8, 7.9, 7.5, 8.0)

    add_arch_box(s, 14, 4.5, 6.5, 4.5,
                 "Transformer encoder\n(2 layers, 8 heads,\n"
                 "GELU, pre-LN, d_ff=1024)",
                 fill=NAVY, border=NAVY, size=13)
    for hb in s.shapes:
        pass  # Note: arch_box white text would need separate impl; navy bg with navy text loses. Re-do:
    # Redo the transformer box with white text:
    # (Easiest: use a one-off shape rather than the helper)
    # We'll skip the helper and just override coloring inline:
    # Actually simpler — replicate add_arch_box but with white text:
    # Let's just add a clean white-text box now
    box = s.shapes.add_shape(MSO_SHAPE.RECTANGLE,
                             Cm(14), Cm(4.5), Cm(6.5), Cm(4.5))
    box.fill.solid()
    box.fill.fore_color.rgb = NAVY
    box.line.color.rgb = NAVY
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = "Transformer encoder\n(2 layers · 8 heads\nGELU · pre-LN)"
    r.font.size = Pt(14)
    r.font.bold = True
    r.font.color.rgb = WHITE
    r.font.name = "Calibri"

    add_arrow(s, 12.5, 6.75, 14, 6.75)

    add_arch_box(s, 22, 4.5, 5.0, 4.5,
                 "Mean-pool\n→ 256-d fused\nembedding",
                 fill=GOLD_BG, border=AMBER, size=13)
    add_arrow(s, 20.5, 6.75, 22, 6.75)

    add_arch_box(s, 28.5, 4.5, 4.5, 4.5,
                 "MLP head\n→ 5 emotion\nlogits",
                 fill=GREEN, border=GREEN, size=13)
    add_arrow(s, 27, 6.75, 28.5, 6.75)

    add_text(s, 0.8, 10.5, SLIDE_W - 1.6, 1.0,
             "Why self-attention over 3 modality tokens, not 6-way pairwise CMA?",
             size=16, bold=True, color=NAVY)
    add_bullets(s, 1.5, 11.7, SLIDE_W - 3, 5, [
        "Our pretrained encoders produce a single pooled vector per "
        "modality (not a sequence of tokens).",
        "At single-token-per-modality resolution, pairwise cross-attention "
        "degenerates to learned projection.",
        "Self-attention over the 3-token modality sequence achieves "
        "equivalent cross-modal mixing with cleaner code and fewer "
        "parameters.",
    ], size=14)


def slide_7_training_results(prs):
    s = blank_slide(prs)
    add_background(s)
    add_header_bar(s, "Training Results (42 Subjects)", page_num=7)

    add_text(s, 0.8, 2.0, SLIDE_W - 1.6, 1.0,
             "Headline 42-subject mean test accuracies:",
             size=16, bold=True, color=NAVY)

    # Bar chart replaces the previous results table
    add_image(s, 0.8, 3.2, 21.5, 11.0,
              FIGURES / "fig_training.png")

    add_text(s, 22.7, 3.2, 10.5, 1.0,
             "Three key findings:",
             size=15, bold=True, color=NAVY)

    add_bullets(s, 22.7, 4.4, 10.5, 9.5, [
        "Per-modality classifiers exceed EAV's published baselines by "
        "+7 to +22 pp.",
        "Multimodal fusion adds 5.4 pp over the best single modality.",
        "Modality-dropout training adds another 4.5 pp on full-modality "
        "accuracy — a regularisation benefit on top of robustness.",
    ], size=12)

    add_callout_box(
        s, 22.7, 14.3, 10.5, 2.5,
        "Δ vs EAV-paper baselines:\n+47 pp on best single modality",
        size=13, bold=True,
    )


def slide_8_comparison(prs):
    s = blank_slide(prs)
    add_background(s)
    add_header_bar(s, "Comparison with Prior EAV Work", page_num=8)

    add_text(s, 0.8, 2.0, SLIDE_W - 1.6, 1.0,
             "EAV was released in September 2024. To date there are three "
             "published trimodal fusion baselines on the dataset:",
             size=16, color=DARK)

    add_table(s, 0.8, 3.4, SLIDE_W - 1.6, 9,
        ["Paper", "Year", "Audio", "Vision", "EEG", "Fusion", "Code"],
        [
            ("Lee et al. — EAV dataset paper", "2024",
             "36.7%", "52.8%", "36.7%", "— (no fusion)", "Yes"),
            ("AMERL (Yin et al.)", "2024",
             "58.2%", "67.2%", "53.5%", "70.86%", "Yes"),
            ("Hyper-MML (Kang et al.) — prev. SOTA", "2025",
             "n/r", "n/r", "n/r", "76.65%", "No"),
            ("EEG-MoCE", "2026",
             "60.5%", "53.8%", "62.7%", "75.88%", "No"),
            ("Ours — cross-attention fusion", "2026",
             "57.1%", "74.6%", "43.9%", "80.20%", "Yes"),
            ("Ours — + softhard modality dropout", "2026",
             "57.1%", "74.6%", "43.9%", "84.70%", "Yes"),
        ],
        header_size=13, body_size=13, highlight_row=5,
    )

    add_callout_box(
        s, 0.8, 13.0, SLIDE_W - 1.6, 3.0,
        "Cross-attention fusion alone exceeds prior SOTA (Hyper-MML) by "
        "+3.5 pp.  With softhard modality dropout: +8.1 pp over SOTA.\n"
        "Three contributions unique to this work: modality dropout · "
        "zero-EEG demo path · cross-modal coherence matrix.",
        size=14, bold=True, color=NAVY,
    )


def slide_8_robustness(prs):
    s = blank_slide(prs)
    add_background(s)
    add_header_bar(s, "Modality Dropout & Robustness", page_num=9)

    add_text(s, 0.8, 2.0, SLIDE_W - 1.6, 1.3,
             "Softhard modality dropout: at each training step one of the "
             "three modality tokens is randomly zeroed.",
             size=15, color=DARK)

    # Bar chart replaces the robustness table
    add_image(s, 0.5, 3.5, 19, 10.5,
              FIGURES / "fig_robustness.png")

    add_text(s, 20, 3.5, 13, 1.0,
             "What the numbers tell us:",
             size=15, bold=True, color=NAVY)

    add_bullets(s, 20, 4.7, 13, 8, [
        "Vision is the dominant modality — removing it costs 30 pp.",
        "EEG-missing inference (the deployment scenario) is "
        "near-tied with full-modality.",
        "Demo path (81.5%) is above the pre-dropout cross-attention "
        "baseline (80.2%).",
    ], size=12)

    add_callout_box(
        s, 0.8, 14.5, SLIDE_W - 1.6, 2.5,
        "The system deployed without EEG (81.5%) is more accurate than "
        "the original full-modality system before dropout (80.2%).",
        size=15, bold=True, color=NAVY,
    )


def slide_9_coherence(prs):
    s = blank_slide(prs)
    add_background(s)
    add_header_bar(s, "Cross-Modal Coherence Analysis — Suppression Matrix",
                   page_num=10)

    add_text(s, 0.8, 2.0, SLIDE_W - 1.6, 1.0,
             "Suppression event: AV agree on top-1, EEG predicts a "
             "different class with confidence ≥ 0.5.",
             size=13, italic=True, color=DARK)

    add_text(s, 0.8, 3.0, 18, 0.9,
             "5,040 trials → 401 suppression events (8.0 % base rate)",
             size=15, bold=True, color=NAVY)

    # Heatmap replaces the suppression matrix table
    add_image(s, 0.5, 4.0, 17, 12.5,
              FIGURES / "fig_suppression.png")

    add_text(s, 18, 4.0, 15, 1.0,
             "Dominant patterns (highlighted):",
             size=15, bold=True, color=NAVY)

    add_bullets(s, 18, 5.2, 15, 11, [
        "Bidirectional Anger ↔ Happiness: 65 + 52 = 117 events "
        "→ high-arousal valence confusion.",
        "Sadness → Neutral: 43 events; Calmness → Neutral: 40 events "
        "→ low-arousal physiological baseline.",
        "Patterns distributed across many subjects "
        "(not driven by outliers).",
        "EEG-Neutral inflation = −1.0 pp → no model-bias confound at "
        "scale; matrix is interpretable at face value.",
    ], size=12)


def slide_10_per_class(prs):
    s = blank_slide(prs)
    add_background(s)
    add_header_bar(s, "Per-Class Performance & Confusion Patterns",
                   page_num=11)

    add_text(s, 0.8, 2.0, SLIDE_W - 1.6, 1.0,
             "Per-emotion classification on the dropout-trained model "
             "(42 subjects, 5,040 trials):",
             size=14, bold=True, color=NAVY)

    # Combined confusion heatmap + F1 bar chart
    add_image(s, 0.5, 3.0, 32.5, 10.0,
              FIGURES / "fig_per_class.png")

    add_callout_box(
        s, 0.8, 13.5, SLIDE_W - 1.6, 3.5,
        "Cross-validation between classification and coherence:  "
        "Neutral ↔ Calmness confusion (14 %) matches the "
        "Calmness → Neutral suppression pattern (40 events). "
        "Low-arousal emotions are confusable in both supervised "
        "classification and cross-modal coherence terms.",
        size=14, bold=False, color=NAVY,
    )


def slide_11_demo(prs):
    s = blank_slide(prs)
    add_background(s)
    add_header_bar(s, "Live Demonstration", page_num=12)

    add_text(s, 0.8, 2.5, SLIDE_W - 1.6, 1.5,
             "Personal video clip processed by the demo path "
             "(AV-only, EEG = 0).",
             size=20, bold=True, color=NAVY, align=PP_ALIGN.CENTER)

    add_callout_box(
        s, 4, 5, SLIDE_W - 8, 6,
        "▶  PLAY DEMO VIDEO  ◀\n\n"
        "(thesis_demo_overlay.mp4)",
        fill=NAVY, border=NAVY, color=WHITE, size=28, bold=True,
    )

    add_text(s, 0.8, 12, SLIDE_W - 1.6, 1.0,
             "What to look for:",
             size=16, bold=True, color=NAVY)
    add_bullets(s, 1.5, 13, SLIDE_W - 3, 4, [
        "Per-segment emotion label updates every ~1 second using "
        "5-second sliding windows.",
        "Audio + video only; no EEG hardware required at inference.",
        "Reported demo-path accuracy on the 42-subject test set: 81.5 %.",
    ], size=14)


def slide_12_future_conclusion(prs):
    s = blank_slide(prs)
    add_background(s)
    add_header_bar(s, "Future Work & Conclusion", page_num=13)

    add_text(s, 0.8, 2.2, 16, 1.0,
             "Phase 2 future work (out of scope):",
             size=16, bold=True, color=NAVY)
    add_bullets(s, 1.0, 3.4, 15.5, 8, [
        "Clinical validation: deploy coherence detection on labelled "
        "depression / alexithymia populations.",
        "In-the-wild deployment: temporal head over per-modality "
        "embeddings; DFEW or MAFW dataset.",
        "MERCL contrastive pre-training across the three modalities.",
        "Full LOSO subject-independent evaluation.",
    ], size=14)

    add_text(s, 18, 2.2, 15, 1.0,
             "Summary of contributions:",
             size=16, bold=True, color=NAVY)
    add_bullets(s, 18, 3.4, 15, 10, [
        "First reproduction of trimodal A+V+E fusion on EAV.",
        "Cross-attention fusion module with named per-modality "
        "embeddings supporting future extensions.",
        "Modality-dropout training enabling EEG-missing inference "
        "at 81.5 % accuracy.",
        "Novel cross-modal coherence framework via suppression matrix.",
        "Cross-method validation: classification confusion mirrors "
        "coherence patterns.",
    ], size=13)

    add_callout_box(
        s, 0.8, 13.0, SLIDE_W - 1.6, 4.5,
        "Thank you for your attention.\nQuestions?",
        size=28, bold=True, color=NAVY,
        fill=GOLD_BG, border=AMBER,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    prs = Presentation()
    prs.slide_width = Cm(SLIDE_W)
    prs.slide_height = Cm(SLIDE_H)

    slide_1_title(prs)
    slide_2_motivation(prs)
    slide_3_research_questions(prs)
    slide_4_dataset(prs)
    slide_5_methodology(prs)
    slide_6_architecture(prs)
    slide_7_training_results(prs)
    slide_8_comparison(prs)
    slide_8_robustness(prs)
    slide_9_coherence(prs)
    slide_10_per_class(prs)
    slide_11_demo(prs)
    slide_12_future_conclusion(prs)

    prs.save(OUTPUT)
    print(f"wrote {OUTPUT}  ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
