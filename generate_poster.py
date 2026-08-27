#!/usr/bin/env python3
"""Generate the Pre-thesis 2 poster as a PowerPoint (.pptx) file.

Layout: A1 landscape, 3 columns.

Column 1: Abstract, Research Objectives, Methodology, Data Preprocessing, Data Specs
Column 2: Proposed Model Architecture, Training Results, Performance Comparison
Column 3: Classwise Performance Metrics, Research Outcomes, Future Work, References

Output: thesis_poster.pptx
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.util import Cm, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "thesis_poster.pptx"

# A1 landscape: 84.1 cm × 59.4 cm
SLIDE_W_CM = 84.1
SLIDE_H_CM = 59.4

# Layout constants
MARGIN = 0.7
HEADER_H = 7.0
COL_TOP = MARGIN + HEADER_H + 0.5
COL_W = (SLIDE_W_CM - 2 * MARGIN - 1.0) / 3   # three columns with gaps
COL_GAP = 0.5
COL_H = SLIDE_H_CM - COL_TOP - MARGIN

# Colors
NAVY = RGBColor(0x1A, 0x3A, 0x5C)
BLUE = RGBColor(0x2E, 0x7E, 0xB8)
LIGHT_BLUE = RGBColor(0xD9, 0xE7, 0xF5)
AMBER = RGBColor(0xC8, 0x7F, 0x12)
DARK = RGBColor(0x20, 0x20, 0x20)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GREY = RGBColor(0xF0, 0xF0, 0xF0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def add_text(slide, x_cm, y_cm, w_cm, h_cm, text, *,
             size=18, bold=False, color=DARK, align=PP_ALIGN.LEFT,
             font="Calibri", anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(Cm(x_cm), Cm(y_cm), Cm(w_cm), Cm(h_cm))
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
    r.font.color.rgb = color
    r.font.name = font
    return tb


def add_section_header(slide, x_cm, y_cm, w_cm, text, *, size=28):
    """A coloured bar with a white section title."""
    bar_h = 1.4
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Cm(x_cm), Cm(y_cm), Cm(w_cm), Cm(bar_h),
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY
    bar.line.fill.background()
    tf = bar.text_frame
    tf.margin_left = Cm(0.2)
    tf.margin_top = Cm(0.1)
    tf.margin_bottom = Cm(0.1)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = text
    r.font.size = Pt(size)
    r.font.bold = True
    r.font.color.rgb = WHITE
    r.font.name = "Calibri"
    return y_cm + bar_h + 0.2


def add_body_paragraphs(slide, x_cm, y_cm, w_cm, h_cm, paragraphs, *, size=16,
                        color=DARK):
    tb = slide.shapes.add_textbox(Cm(x_cm), Cm(y_cm), Cm(w_cm), Cm(h_cm))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Cm(0.1)
    tf.margin_right = Cm(0.1)
    for i, para in enumerate(paragraphs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.JUSTIFY
        p.space_after = Pt(6)
        r = p.add_run()
        r.text = para
        r.font.size = Pt(size)
        r.font.color.rgb = color
        r.font.name = "Calibri"
    return tb


def add_bullets(slide, x_cm, y_cm, w_cm, h_cm, bullets, *, size=15,
                color=DARK, numbered=False):
    tb = slide.shapes.add_textbox(Cm(x_cm), Cm(y_cm), Cm(w_cm), Cm(h_cm))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = Cm(0.1)
    tf.margin_right = Cm(0.1)
    for i, text in enumerate(bullets):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(4)
        bullet = f"{i+1}. " if numbered else "•  "
        r = p.add_run()
        r.text = bullet + text
        r.font.size = Pt(size)
        r.font.color.rgb = color
        r.font.name = "Calibri"
    return tb


def add_table(slide, x_cm, y_cm, w_cm, h_cm, headers, rows, *,
              header_size=14, body_size=13, header_color=NAVY,
              header_text_color=WHITE):
    nrows = len(rows) + 1
    ncols = len(headers)
    table = slide.shapes.add_table(
        nrows, ncols, Cm(x_cm), Cm(y_cm), Cm(w_cm), Cm(h_cm)
    ).table

    # Header row
    for j, h in enumerate(headers):
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
        r.text = h
        r.font.size = Pt(header_size)
        r.font.bold = True
        r.font.color.rgb = header_text_color
        r.font.name = "Calibri"

    # Body rows
    for i, row in enumerate(rows, start=1):
        for j, val in enumerate(row):
            cell = table.cell(i, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if i % 2 == 1 else LIGHT_GREY
            cell.margin_left = Cm(0.1)
            cell.margin_right = Cm(0.1)
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
    return table


# ---------------------------------------------------------------------------
# Build the poster
# ---------------------------------------------------------------------------

def build_poster():
    prs = Presentation()
    prs.slide_width = Cm(SLIDE_W_CM)
    prs.slide_height = Cm(SLIDE_H_CM)

    blank = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank)

    # White background
    bg = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, Cm(SLIDE_W_CM), Cm(SLIDE_H_CM)
    )
    bg.fill.solid()
    bg.fill.fore_color.rgb = WHITE
    bg.line.fill.background()

    # ---- Header banner ----
    header = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Cm(MARGIN), Cm(MARGIN), Cm(SLIDE_W_CM - 2 * MARGIN), Cm(HEADER_H),
    )
    header.fill.solid()
    header.fill.fore_color.rgb = NAVY
    header.line.fill.background()

    add_text(
        slide, MARGIN + 0.5, MARGIN + 0.5, SLIDE_W_CM - 2 * MARGIN - 1, 2.2,
        "Cross-Modal Affective Coherence: Detecting Discrepancies Between "
        "Externally Expressed and Internally Experienced Emotion",
        size=42, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
        anchor=MSO_ANCHOR.MIDDLE,
    )
    add_text(
        slide, MARGIN + 0.5, MARGIN + 2.8, SLIDE_W_CM - 2 * MARGIN - 1, 1.2,
        "A Multimodal Framework Using EEG, Audio and Video on the EAV Dataset",
        size=24, bold=False, color=WHITE,
        align=PP_ALIGN.CENTER,
    )
    add_text(
        slide, MARGIN + 0.5, MARGIN + 4.2, SLIDE_W_CM - 2 * MARGIN - 1, 1.0,
        "Shafin  ·  [Institution name — to be filled in]  ·  June 2026",
        size=20, bold=True, color=WHITE, align=PP_ALIGN.CENTER,
    )

    # Column positions
    col1_x = MARGIN
    col2_x = MARGIN + COL_W + COL_GAP
    col3_x = MARGIN + 2 * (COL_W + COL_GAP)

    # ===================== COLUMN 1 =====================
    y = COL_TOP

    # Abstract
    y = add_section_header(slide, col1_x, y, COL_W, "ABSTRACT")
    add_body_paragraphs(
        slide, col1_x, y, COL_W, 8.5,
        [
            "Emotion recognition systems that rely on audio and video alone "
            "capture only externally expressed affect, missing the internal "
            "physiological state. We present a trimodal framework on the "
            "EAV dataset that addresses two complementary goals: (1) "
            "accurate trimodal emotion classification, and (2) novel "
            "detection of cross-modal incoherence — events where the "
            "displayed emotion diverges from the internal state indicated "
            "by EEG. Our modality-dropout-trained system reaches 84.7% "
            "mean test accuracy across 42 subjects, exceeding the prior "
            "published state of the art on EAV (Hyper-MML, 76.65%) by "
            "+8.05 percentage points, and reaches 81.5% with EEG zeroed "
            "at inference, demonstrating graceful degradation. All four "
            "method comparisons are statistically significant (paired "
            "Wilcoxon, p<0.05). On 5,040 test trials, 8.0% exhibit "
            "confident cross-modal incoherence, with dominant patterns "
            "being bidirectional Anger↔Happiness confusion (117 events) "
            "and low-arousal-to-Neutral suppression (83 events)."
        ],
        size=14,
    )
    y += 9.6

    # Research Objectives
    y = add_section_header(slide, col1_x, y, COL_W, "RESEARCH OBJECTIVES")
    add_bullets(
        slide, col1_x, y, COL_W, 8.5,
        [
            "Per-modality baselines: Can audio, video and EEG individually "
            "exceed the EAV paper's per-modality baselines?",
            "Fusion gain: Does multimodal fusion improve over the best "
            "single modality?",
            "Architectural contribution: Does learned cross-modal attention "
            "outperform naïve late fusion?",
            "Coherence as signal: When modalities disagree, do consistent "
            "cross-population patterns emerge?",
            "Robustness: Does the system degrade gracefully when EEG is "
            "unavailable at inference time?",
        ],
        size=13, numbered=True,
    )
    y += 8.7

    # Methodology
    y = add_section_header(slide, col1_x, y, COL_W, "METHODOLOGY")
    add_bullets(
        slide, col1_x, y, COL_W, 8.0,
        [
            "Per-modality encoders: AST (audio), ViT (video), EEGNet (EEG).",
            "Cross-attention fusion module over three modality tokens with "
            "learnable modality embeddings and 2-layer transformer encoder.",
            "Concat-MLP fusion baseline (1.34M params) on concatenated "
            "features as a learned-fusion comparator.",
            "Softhard modality dropout training to enable missing-modality "
            "inference and provide architecture-agnostic regularisation.",
            "Coherence analysis: pairwise KL divergence + 5×5 suppression "
            "matrix counting confident cross-modal divergence events.",
        ],
        size=13,
    )
    y += 8.2

    # Data Preprocessing
    y = add_section_header(slide, col1_x, y, COL_W, "DATA PREPROCESSING")
    add_bullets(
        slide, col1_x, y, COL_W, 6.0,
        [
            "Audio: resample to 16 kHz; 5-second segmentation; AST feature "
            "extractor (log-mel spectrogram).",
            "Video: 25 frames per 5-second clip; MTCNN face detection; "
            "224×224 face crop; ViT image processor.",
            "EEG: 500→100 Hz downsampling; fifth-order Butterworth bandpass "
            "[0.5, 45] Hz; 5-second non-overlapping windows.",
            "Per-subject 70/30 split (h_idx=56): 280 train / 120 test trials.",
        ],
        size=13,
    )
    y += 6.2

    # Data Specs
    y = add_section_header(slide, col1_x, y, COL_W, "DATASET SPECIFICATIONS")
    add_table(
        slide, col1_x, y, COL_W, 4.2,
        ["Property", "Value"],
        [
            ("Participants",           "42 (lab-controlled)"),
            ("Emotion classes",        "5 (Neutral, Sad, Angry, Happy, Calm)"),
            ("Interactions/subject",   "200 (Listen + Speak)"),
            ("EEG channels / rate",    "30 / 500 Hz"),
            ("Audio rate / duration",  "16 kHz / 5-s segments"),
            ("Video resolution / fps", "224×224 / 25 fps post-pre"),
        ],
        header_size=14, body_size=12,
    )

    # ===================== COLUMN 2 =====================
    y = COL_TOP

    # Proposed Model Architecture
    y = add_section_header(slide, col2_x, y, COL_W, "PROPOSED MODEL ARCHITECTURE")
    add_body_paragraphs(
        slide, col2_x, y, COL_W, 8.0,
        [
            "The TrimodalAttentionFusion module receives three pooled "
            "per-modality features (audio: 768-d AST output, video: 768-d "
            "ViT output, EEG: 960-d EEGNet output) and produces a 5-class "
            "emotion prediction together with named per-modality embeddings.",
            "",
            "Architectural pipeline (2.28M trainable parameters):",
        ],
        size=13,
    )
    y += 4.0

    add_bullets(
        slide, col2_x, y, COL_W, 5.0,
        [
            "Linear projection of each modality to common d=256 dimension.",
            "Add learnable modality embedding (3×256) to distinguish tokens.",
            "Stack as 3-token sequence; apply 2-layer transformer encoder "
            "(8 heads, GELU, pre-layer-norm, dim_ff=1024).",
            "Mean-pool 3 post-attention tokens to fused 256-d vector.",
            "MLP head (Linear→GELU→Dropout→Linear) to 5-class logits.",
        ],
        size=13,
    )
    y += 5.5

    add_body_paragraphs(
        slide, col2_x, y, COL_W, 3.0,
        [
            "Two architectural commitments: (a) forward() returns named "
            "per-modality embeddings to support a future temporal head; "
            "(b) forward(audio, video, x_eeg=zeros) is a valid call, enabling "
            "missing-EEG inference at deployment.",
        ],
        size=13,
    )
    y += 3.2

    # Training Results
    y = add_section_header(slide, col2_x, y, COL_W, "TRAINING RESULTS (42 SUBJECTS)")
    add_table(
        slide, col2_x, y, COL_W, 8.4,
        ["Method", "Mean Acc", "Std"],
        [
            ("Audio only (AST)",               "57.1%", "0.116"),
            ("Vision only (ViT)",              "74.6%", "0.102"),
            ("EEG only (EEGNet)",              "43.9%", "0.094"),
            ("Naïve late fusion (mean-softmax)", "77.5%", "0.091"),
            ("Cross-attention fusion",           "80.2%", "0.083"),
            ("Concat-MLP fusion",                "81.7%", "0.076"),
            ("+ Softhard modality dropout",      "84.7%", "0.079"),
            ("Demo path (AV-only, EEG=0)",      "81.5%", "0.088"),
        ],
        header_size=14, body_size=12,
    )
    y += 8.6

    # Performance Comparison: new SOTA against prior fusion methods on EAV
    y = add_section_header(slide, col2_x, y, COL_W,
                           "PRIOR FUSION METHODS ON EAV")
    add_table(
        slide, col2_x, y, COL_W, 5.5,
        ["Method", "Year", "Fusion Acc"],
        [
            ("AMERL (Yin et al.)",       "2024", "70.86%"),
            ("Hyper-MML (Kang et al.)",  "2025", "76.65%"),
            ("EEG-MoCE",                 "2026", "75.88%"),
            ("Ours (cross-attention)",   "2026", "80.20%"),
            ("Ours (+ modality dropout)", "2026", "84.70%"),
        ],
        header_size=13, body_size=12,
    )
    y += 5.7

    # Robustness mini-table
    add_body_paragraphs(
        slide, col2_x, y, COL_W, 1.5,
        ["Degraded-modality accuracy (with modality dropout):"],
        size=13,
    )
    y += 1.0
    add_table(
        slide, col2_x, y, COL_W, 3.5,
        ["Configuration", "Mean", "Drop"],
        [
            ("Full (A+V+E)",        "84.7%", "—"),
            ("AV only (no EEG)",    "81.5%", "−3.2 pp"),
            ("VE only (no audio)",  "79.7%", "−5.0 pp"),
            ("AE only (no video)",  "54.0%", "−30.7 pp"),
        ],
        header_size=13, body_size=12,
    )

    # ===================== COLUMN 3 =====================
    y = COL_TOP

    # Classwise Performance Metrics
    y = add_section_header(slide, col3_x, y, COL_W, "CLASSWISE PERFORMANCE METRICS")
    add_table(
        slide, col3_x, y, COL_W, 6.0,
        ["Emotion", "Precision", "Recall", "F1"],
        [
            ("Neutral",   "0.831", "0.783", "0.806"),
            ("Sadness",   "0.902", "0.782", "0.837"),
            ("Anger",     "0.885", "0.871", "0.878"),
            ("Happiness", "0.852", "0.949", "0.898"),
            ("Calmness",  "0.778", "0.850", "0.813"),
        ],
        header_size=13, body_size=12,
    )
    y += 6.2

    add_body_paragraphs(
        slide, col3_x, y, COL_W, 1.5,
        ["Aggregate confusion matrix (row %; predicted per true class):"],
        size=13, color=DARK,
    )
    y += 1.1
    add_table(
        slide, col3_x, y, COL_W, 6.0,
        ["True \\ Pred", "Neu", "Sad", "Ang", "Hap", "Cal"],
        [
            ("Neutral",   "78", "1",  "1",  "3",  "14"),
            ("Sadness",   "5",  "78", "7",  "3",  "5"),
            ("Anger",     "2",  "5",  "87", "4",  "1"),
            ("Happiness", "0",  "0",  "2",  "94", "2"),
            ("Calmness",  "7",  "1",  "0",  "5",  "85"),
        ],
        header_size=12, body_size=11,
    )
    y += 6.2

    # Suppression matrix highlight
    add_body_paragraphs(
        slide, col3_x, y, COL_W, 1.5,
        ["Cross-modal suppression patterns (42 subj., 401 events):"],
        size=13,
    )
    y += 1.1
    add_table(
        slide, col3_x, y, COL_W, 4.0,
        ["External → Internal", "Events", "%"],
        [
            ("Anger → Happiness",   "65", "16.2%"),
            ("Happiness → Anger",   "52", "13.0%"),
            ("Sadness → Neutral",   "43", "10.7%"),
            ("Calmness → Neutral",  "40", "10.0%"),
        ],
        header_size=12, body_size=12,
    )
    y += 4.2

    # Research Outcomes
    y = add_section_header(slide, col3_x, y, COL_W, "RESEARCH OUTCOMES")
    add_bullets(
        slide, col3_x, y, COL_W, 8.5,
        [
            "New SOTA on EAV: 84.7% (modality dropout) exceeds prior "
            "best (Hyper-MML, 76.65%) by +8.05 pp.",
            "Per-modality classifiers exceed EAV's published baselines "
            "by 7-22 pp.",
            "Learned fusion significantly beats naive late fusion: "
            "cross-attention +2.7 pp (p=0.023), concat-MLP +4.2 pp "
            "(p<10⁻⁴).",
            "Modality dropout is the dominant gain: +4.5 pp over "
            "cross-attention (p=1.3×10⁻⁷), architecture-agnostic.",
            "8.0% of 5,040 trials show confident cross-modal incoherence; "
            "patterns cross-population, not outlier-driven.",
            "Demo path (AV-only) at 81.5% — significantly exceeds the "
            "dropout-untrained full-modality model (p=0.034).",
        ],
        size=12, numbered=True,
    )
    y += 9.0

    # Future Work
    y = add_section_header(slide, col3_x, y, COL_W, "FUTURE WORK (PHASE 2)")
    add_bullets(
        slide, col3_x, y, COL_W, 5.5,
        [
            "Clinical validation: coherence detection on labelled "
            "depression / alexithymia populations.",
            "Primary AV dataset (smartphone-recorded) merged with EAV "
            "via two-stage fine-tuning; Likert self-report as EEG proxy.",
            "In-the-wild deployment: temporal head over per-modality "
            "embeddings; DFEW or MAFW datasets.",
            "MERCL contrastive pre-training across the three modalities.",
            "5-fold subject-independent evaluation for generalisation.",
        ],
        size=12,
    )
    y += 5.7

    # References
    y = add_section_header(slide, col3_x, y, COL_W, "REFERENCES")
    refs = [
        "[1] Lee, Shomanov et al. EAV: EEG-Audio-Video Dataset. "
        "Sci. Data, 2024.",
        "[2] Yin et al. AMERL: Attention-based Multimodal Emotion "
        "Recognition. arXiv:2411.00822, 2024.",
        "[3] Kang et al. Hypergraph Multi-Modal Learning for EEG-based "
        "Emotion Recognition. arXiv:2502.21154, 2025.",
        "[4] EEG-MoCE: Hyperbolic Mixture-of-Curvature Experts. "
        "arXiv:2604.12579, 2026.",
        "[5] Lee, Kim, Kim. Emotion Recognition Using EEG and Audiovisual "
        "Features with Contrastive Learning. Bioengineering, 2024.",
        "[6] Chumachenko et al. Self-Attention Fusion for Audiovisual "
        "Emotion Recognition with Incomplete Data. ICPR, 2022.",
        "[7] Vaswani et al. Attention Is All You Need. NeurIPS, 2017.",
        "[8] Gong, Chung, Glass. AST: Audio Spectrogram Transformer. "
        "Interspeech, 2021.",
        "[9] Lawhern et al. EEGNet: A Compact Convolutional Network "
        "for EEG-based BCIs. J. Neural Eng., 2018.",
    ]
    tb = slide.shapes.add_textbox(Cm(col3_x), Cm(y), Cm(COL_W), Cm(COL_H - y))
    tf = tb.text_frame
    tf.word_wrap = True
    for i, ref in enumerate(refs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(2)
        r = p.add_run()
        r.text = ref
        r.font.size = Pt(10)
        r.font.name = "Calibri"
        r.font.color.rgb = DARK

    prs.save(OUTPUT)
    print(f"wrote {OUTPUT}")


# Helper to handle italic argument cleanly (PP_ALIGN doesn't have italic)
def _patch_add_text():
    """The signature of add_text in this script doesn't take italic_color.
    The add_text call above passed italic_color=None — that's a bug in
    the call; restate add_text correctly here to avoid runtime error."""
    pass


if __name__ == "__main__":
    build_poster()
