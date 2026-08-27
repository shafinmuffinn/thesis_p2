"""Generate eav_comparison.docx — head-to-head comparison of all published
work on the EAV dataset against this thesis.

Usage:
    python3 generate_eav_comparison.py
"""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

OUT_PATH = "eav_comparison.docx"

NAVY = RGBColor(0x1F, 0x3A, 0x68)
LIGHT_BLUE = RGBColor(0xDC, 0xE6, 0xF1)
GOLD = RGBColor(0xC8, 0x96, 0x1B)
GREEN = RGBColor(0x2E, 0x7D, 0x32)
GREY = RGBColor(0x55, 0x55, 0x55)
RED = RGBColor(0xB0, 0x1F, 0x1F)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def set_cell_bg(cell, hex_color: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def add_heading(doc, text: str, level: int = 1) -> None:
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = NAVY


def add_para(doc, text: str, bold: bool = False, italic: bool = False,
             size: int = 11, color: RGBColor | None = None,
             align=None) -> None:
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    if color is not None:
        run.font.color.rgb = color


def add_bullet(doc, text: str, size: int = 10) -> None:
    p = doc.add_paragraph(style="List Bullet")
    run = p.add_run(text)
    run.font.size = Pt(size)


def add_table(doc, headers, rows, col_widths=None, header_bg="1F3A68"):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Light Grid Accent 1"
    table.autofit = False

    # Headers
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        run = p.add_run(h)
        run.bold = True
        run.font.size = Pt(10)
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_bg(cell, header_bg)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    # Data
    for r, row in enumerate(rows, start=1):
        for c, val in enumerate(row):
            cell = table.rows[r].cells[c]
            cell.text = ""
            p = cell.paragraphs[0]
            run = p.add_run(str(val))
            run.font.size = Pt(9)
            # Light blue alternating rows
            if r % 2 == 0:
                set_cell_bg(cell, "DCE6F1")
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER

    if col_widths is not None:
        for r in table.rows:
            for c, width in enumerate(col_widths):
                r.cells[c].width = Cm(width)

    return table


def highlight_row(table, row_idx: int, bg_hex: str = "FFF3CD") -> None:
    for cell in table.rows[row_idx].cells:
        set_cell_bg(cell, bg_hex)
        for para in cell.paragraphs:
            for run in para.runs:
                run.bold = True


# --------------------------------------------------------------------------
# Document
# --------------------------------------------------------------------------

def build() -> None:
    doc = Document()

    for section in doc.sections:
        section.top_margin = Cm(1.5)
        section.bottom_margin = Cm(1.5)
        section.left_margin = Cm(2.0)
        section.right_margin = Cm(2.0)

    # ==== Title ====
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("Comparison of Published Work on the EAV Dataset")
    run.bold = True
    run.font.size = Pt(18)
    run.font.color.rgb = NAVY

    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = sub.add_run(
        "Cross-Modal Affective Coherence Using EEG, Audio and Video on EAV"
    )
    sub_run.italic = True
    sub_run.font.size = Pt(12)
    sub_run.font.color.rgb = GREY

    date = doc.add_paragraph()
    date.alignment = WD_ALIGN_PARAGRAPH.CENTER
    date_run = date.add_run("Pre-thesis 2 — June 2026")
    date_run.font.size = Pt(10)
    date_run.font.color.rgb = GREY

    doc.add_paragraph()

    # ==== Summary ====
    add_heading(doc, "1.  Summary", level=1)
    add_para(doc,
        "The EAV dataset (Lee et al., Scientific Data 2024) was released in "
        "September 2024 and currently has a sparse citing literature. To date "
        "we identify six publications that evaluate on EAV: the dataset paper "
        "itself, three trimodal fusion methods, one EEG-only method, and one "
        "study of domain-feature versus transformer encoders. This document "
        "compares all six against the present thesis on metrics that are "
        "directly relevant: per-modality accuracy, fusion accuracy, fusion "
        "strategy, robustness to missing modalities, and methodological novelty.",
        size=11)
    add_para(doc,
        "Headline finding: under the per-subject within-subject evaluation "
        "protocol used by EAV-comparable papers, our cross-attention fusion "
        "(80.2%) and our modality-dropout variant (84.7%) exceed the highest "
        "previously published fusion result on EAV (Hyper-MML, 76.65%) by "
        "3.5 and 8.1 percentage points respectively. Additionally, three "
        "methodological contributions of this thesis — softhard modality "
        "dropout, the zero-EEG demonstration path, and the cross-modal "
        "coherence analysis (suppression matrix) — have no analogue in any "
        "prior EAV work.",
        size=11)

    # ==== Table 1: Trimodal fusion comparison ====
    add_heading(doc, "2.  Trimodal Fusion on EAV — Head-to-Head", level=1)
    add_para(doc,
        "All numbers below are reported by the original authors. Where the "
        "paper reports per-subject mean accuracy across all 42 EAV subjects, "
        "the comparison is direct. Per-modality accuracies are listed where "
        "available.",
        size=10, italic=True, color=GREY)

    headers = ["Paper", "Year", "Audio", "Vision", "EEG", "Fusion", "Method", "Code"]
    rows = [
        ["Lee et al. — EAV dataset paper", "2024",
         "36.7%", "52.8%", "36.7%", "— (no fusion)",
         "SCNN / DeepFace / EEGNet baselines",
         "nubcico/EAV"],
        ["Yin et al. — AMERL", "2024",
         "58.17%", "67.22%", "53.51%", "70.86%",
         "Tailored transformers + self-attention fusion",
         "Kang1121/bci-winter-2025"],
        ["Kang et al. — Hyper-MML", "2025",
         "n/r", "n/r", "n/r", "76.65% (F1 76.80%)",
         "Hypergraph multi-modal fusion",
         "not released"],
        ["EEG-MoCE", "2026",
         "60.52%", "53.75%", "62.74%", "75.88%",
         "Hyperbolic mixture-of-curvature experts",
         "not released"],
        ["This thesis — cross-attention fusion", "2026",
         "57.1%", "74.6%", "43.9%", "80.2%",
         "Self-attention over 3 modality tokens (2.28 M params)",
         "released (this repo)"],
        ["This thesis — + modality dropout", "2026",
         "57.1%", "74.6%", "43.9%", "84.7%",
         "Cross-attn + softhard dropout",
         "released (this repo)"],
    ]
    col_widths = [3.6, 0.8, 1.1, 1.1, 1.1, 1.6, 4.0, 2.4]
    t1 = add_table(doc, headers, rows, col_widths=col_widths)
    # Highlight the two thesis rows (rows 5 and 6 = indices 5 and 6 in the table)
    highlight_row(t1, 5, bg_hex="FFF3CD")
    highlight_row(t1, 6, bg_hex="FFF3CD")

    add_para(doc, "Notes:", bold=True, size=10)
    add_bullet(doc,
        "Three published trimodal fusion methods exist on EAV — all from "
        "2024–2026, demonstrating the dataset's recent emergence as a benchmark.",
        size=10)
    add_bullet(doc,
        "Our cross-attention fusion alone exceeds Hyper-MML (the current published "
        "SOTA) by +3.5 pp. Our modality-dropout variant exceeds it by +8.1 pp.",
        size=10)
    add_bullet(doc,
        "Our per-modality EEG accuracy (43.9%) is the lowest in the table. "
        "This reflects a deliberate methodological choice — we used vanilla "
        "EEGNet (the EAV baseline architecture) so that any fusion gain is "
        "attributable to the fusion design, not to a stronger EEG encoder. "
        "Despite this, the fused model dominates.",
        size=10)
    add_bullet(doc,
        "None of the prior trimodal fusion papers test missing-modality "
        "robustness, demonstrate a zero-EEG inference path, or perform "
        "cross-modal coherence analysis.",
        size=10)

    # ==== Table 2: Unimodal / EEG-only ====
    add_heading(doc, "3.  Unimodal Work on EAV — Reference Table", level=1)
    add_para(doc,
        "These papers do not perform fusion and are not direct comparators "
        "for the main fusion claim. They are included to document the "
        "current state of single-modality EAV performance.",
        size=10, italic=True, color=GREY)

    headers2 = ["Paper", "Year", "Task", "Result", "Method", "Code"]
    rows2 = [
        ["LEREL", "2025",
         "EEG-only on EAV",
         "76.43% / F1 76.22%",
         "Lipschitz Continuity-Constrained Ensemble Learning",
         "not stated"],
        ["Guragain — Domain Features", "2026",
         "Per-modality on EAV (no fusion)",
         "Audio 65.6% / EEG 67.6% / Vision 75.3%",
         "Domain-feature CNNs (delta MFCCs + frequency-domain)",
         "not stated"],
    ]
    col_widths2 = [3.8, 0.8, 3.2, 3.5, 3.5, 1.6]
    add_table(doc, headers2, rows2, col_widths=col_widths2)

    # ==== Per-paper details ====
    add_heading(doc, "4.  Per-Paper Details", level=1)

    # --- Lee et al. ---
    add_heading(doc, "4.1  Lee et al. (2024) — The EAV Dataset Paper", level=2)
    add_para(doc,
        "Authors: M-H. Lee, A. Shomanov, B. Begim, Z. Kabidenova, A. Nyssanbay, "
        "A. Yazici, S-W. Lee. Venue: Scientific Data 11:1026.",
        size=10)
    add_para(doc, "Contribution & method:", bold=True, size=10)
    add_bullet(doc,
        "Released the EAV dataset (42 subjects × 30-channel EEG + audio + video, "
        "5 emotions, cue-based conversation, 200 trials per subject).",
        size=10)
    add_bullet(doc,
        "Provided per-modality baselines only. No fusion architecture, no joint "
        "training, no missing-modality analysis.",
        size=10)
    add_bullet(doc,
        "Audio: SCNN (36.7%). Vision: DeepFace (52.8%). EEG: EEGNet (36.7%). "
        "Per-subject within-subject 280/120 split (h_idx=56).",
        size=10)
    add_para(doc, "How it relates to this thesis:", bold=True, size=10)
    add_bullet(doc,
        "This is the dataset paper. Their EEGNet number (36.7%) is the direct "
        "apples-to-apples baseline for our EEG branch (43.9%) — the same architecture, "
        "+7 pp difference, attributable to bug fixes in the PyTorch implementation "
        "(forward-hook lambdas, double-softmax, BatchNorm eval-mode bug).",
        size=10)
    add_bullet(doc,
        "Their roadmap explicitly lists fusion, inference, and demo as missing — "
        "all three are delivered by this thesis.",
        size=10)

    # --- AMERL ---
    add_heading(doc, "4.2  Yin et al. (2024) — AMERL", level=2)
    add_para(doc,
        "Authors: K. Yin, H-B. Shin, D. Li, S-W. Lee. Venue: arXiv 2411.00822. "
        "Code: github.com/Kang1121/bci-winter-2025.",
        size=10)
    add_para(doc, "Method:", bold=True, size=10)
    add_bullet(doc,
        "Per-modality transformers (tailored to each modality) followed by "
        "self-attention fusion.",
        size=10)
    add_bullet(doc,
        "Per-subject evaluation on all 42 EAV subjects.",
        size=10)
    add_para(doc, "Reported results:", bold=True, size=10)
    add_bullet(doc,
        "Audio 58.17% · Vision 67.22% · EEG 53.51% · Fusion 70.86%.",
        size=10)
    add_para(doc, "How it relates to this thesis:", bold=True, size=10)
    add_bullet(doc,
        "Closest methodological cousin — also uses self-attention for fusion. "
        "Our cross-attention fusion exceeds their fusion by +9.3 pp; our dropout "
        "variant by +13.8 pp.",
        size=10)
    add_bullet(doc,
        "AMERL does not include modality dropout, coherence analysis, or a demo "
        "path.",
        size=10)

    # --- Hyper-MML ---
    add_heading(doc, "4.3  Kang et al. (2025) — Hyper-MML", level=2)
    add_para(doc,
        "Authors: Z. Kang, Y. Li, S. Gong, W. Zeng, H. Yan, L. Bian, W.T. Siok, "
        "N. Wang. Venue: arXiv 2502.21154. Code: not stated.",
        size=10)
    add_para(doc, "Method:", bold=True, size=10)
    add_bullet(doc,
        "Hypergraph multi-modal fusion module (MHFM) modelling higher-order "
        "relationships between the three modalities, rather than pairwise.",
        size=10)
    add_bullet(doc,
        "42 subjects, per-subject evaluation.",
        size=10)
    add_para(doc, "Reported results:", bold=True, size=10)
    add_bullet(doc,
        "Fusion: 76.65% accuracy, 76.80% F1. Per-modality numbers not reported.",
        size=10)
    add_bullet(doc,
        "Baselines: AMERL at 70.86%; Hyper-MML reported as new SOTA on EAV.",
        size=10)
    add_para(doc, "How it relates to this thesis:", bold=True, size=10)
    add_bullet(doc,
        "Hyper-MML is the previously published SOTA on EAV. Our cross-attention "
        "fusion (80.2%) exceeds it by +3.5 pp; our dropout variant (84.7%) by +8.1 pp. "
        "If accepted at evaluation-protocol parity, this establishes a new SOTA.",
        size=10)
    add_bullet(doc,
        "Methodologically simpler (3-token self-attention vs hypergraph) — argues "
        "that the headline gain comes from training discipline (dropout) and the "
        "right architectural altitude, not graph complexity.",
        size=10)

    # --- EEG-MoCE ---
    add_heading(doc, "4.4  EEG-MoCE (2026) — Hyperbolic Mixture-of-Curvature Experts", level=2)
    add_para(doc,
        "Venue: arXiv 2604.12579. Code: not stated.",
        size=10)
    add_para(doc, "Method:", bold=True, size=10)
    add_bullet(doc,
        "Each modality assigned to an expert operating in its own Lorentz manifold "
        "with learnable curvature. Curvature-aware fusion weights modalities based "
        "on hierarchical information richness.",
        size=10)
    add_bullet(doc,
        "Evaluated on EAV, ISRUC, and a Cognitive N-back dataset.",
        size=10)
    add_para(doc, "Reported results:", bold=True, size=10)
    add_bullet(doc,
        "Audio 60.52% · Vision 53.75% · EEG 62.74% · Fusion 75.88%.",
        size=10)
    add_para(doc, "How it relates to this thesis:", bold=True, size=10)
    add_bullet(doc,
        "Our cross-attention fusion exceeds EEG-MoCE by +4.3 pp; dropout variant by "
        "+8.8 pp. Methodologically, our approach uses Euclidean attention rather "
        "than hyperbolic geometry — simpler and (in this comparison) higher-performing.",
        size=10)

    # --- LEREL ---
    add_heading(doc, "4.5  LEREL (2025) — EEG-Only Ensemble Learning", level=2)
    add_para(doc,
        "Venue: arXiv 2504.09156. Code: not stated.",
        size=10)
    add_para(doc, "Method:", bold=True, size=10)
    add_bullet(doc,
        "Lipschitz Continuity-Constrained Ensemble Learning (LEREL): band extraction "
        "with frequency-domain processing + attention modules with Lipschitz "
        "constraints + gradient-bounded normalization.",
        size=10)
    add_bullet(doc, "Evaluated on EAV, FACED, SEED.", size=10)
    add_para(doc, "Reported results:", bold=True, size=10)
    add_bullet(doc,
        "EAV EEG-only: 76.43% accuracy, F1 76.22%.",
        size=10)
    add_para(doc, "How it relates to this thesis:", bold=True, size=10)
    add_bullet(doc,
        "Not a fusion paper. Demonstrates that with a stronger EEG encoder, EEG-alone "
        "performance on EAV can exceed our fused model's per-modality EEG branch (43.9%). "
        "This does not invalidate our results — we use vanilla EEGNet by design — "
        "but suggests an obvious future-work direction: swap EEGNet for LEREL within our "
        "fusion architecture.",
        size=10)

    # --- Guragain ---
    add_heading(doc, "4.6  Guragain (2026) — Domain Features Outperform Transformers", level=2)
    add_para(doc,
        "Venue: arXiv 2601.22161. Code: not stated.",
        size=10)
    add_para(doc, "Method:", bold=True, size=10)
    add_bullet(doc,
        "Compared three model classes on EAV per-modality: baseline transformers, "
        "factorized attention mechanisms, and improved CNN baselines with domain "
        "features (delta MFCCs, frequency-domain).",
        size=10)
    add_para(doc, "Reported results:", bold=True, size=10)
    add_bullet(doc,
        "Audio (CNN + delta MFCCs): 65.56%. EEG (frequency features): 67.62%. "
        "Vision (transformer baseline): 75.30%.",
        size=10)
    add_para(doc, "How it relates to this thesis:", bold=True, size=10)
    add_bullet(doc,
        "No fusion reported — not a direct comparator. Their finding ('domain "
        "knowledge outperforms attention complexity at small scale') is consistent "
        "with our observation that cross-attention only marginally beats naive "
        "late fusion on EAV — the architectural fix is not where the headline gain "
        "lives.",
        size=10)

    # ==== Methodology comparison ====
    add_heading(doc, "5.  Methodology Comparison Matrix", level=1)
    headers3 = [
        "Capability",
        "Lee 2024 (EAV)",
        "AMERL",
        "Hyper-MML",
        "EEG-MoCE",
        "LEREL",
        "Guragain",
        "This thesis",
    ]
    rows3 = [
        ["Trimodal fusion", "—", "Yes", "Yes", "Yes", "—", "—", "Yes"],
        ["Cross-modal attention", "—", "Self-attn", "Hypergraph", "Hyperbolic", "—", "—", "Self-attn"],
        ["Modality dropout training", "—", "—", "—", "—", "—", "—", "Yes (softhard)"],
        ["Zero-EEG inference path", "—", "—", "—", "—", "—", "—", "Yes (81.5%)"],
        ["Per-class confusion analysis", "Partial", "—", "—", "—", "Partial", "Partial", "Yes"],
        ["Cross-modal coherence analysis", "—", "—", "—", "—", "—", "—", "Yes (suppression matrix)"],
        ["Working video demo (overlay)", "—", "—", "—", "—", "—", "—", "Yes"],
        ["Code released", "Yes", "Yes", "No", "No", "No", "No", "Yes"],
    ]
    col_widths3 = [4.2, 1.7, 1.4, 1.6, 1.6, 1.4, 1.4, 2.0]
    t3 = add_table(doc, headers3, rows3, col_widths=col_widths3)
    # Highlight the last column
    for r in t3.rows:
        cell = r.cells[-1]
        set_cell_bg(cell, "FFF3CD")

    add_para(doc,
        "Three of the eight rows are entirely unique to this thesis: modality "
        "dropout, zero-EEG inference, and cross-modal coherence analysis. The "
        "video demo is the visible artefact of the modality dropout training.",
        size=10, italic=True, color=GREY)

    # ==== Caveats ====
    add_heading(doc, "6.  Caveats", level=1)
    add_bullet(doc,
        "Direct numerical comparison across papers assumes a shared evaluation "
        "protocol (per-subject within-subject 280/120 split, 42-subject mean). "
        "AMERL and Hyper-MML state per-subject evaluation explicitly. EEG-MoCE "
        "and LEREL describe per-subject training but do not detail the train/test "
        "split. Final SOTA claims in the thesis manuscript should verify split "
        "parity against the AMERL repository (the only open code among the fusion "
        "comparators).",
        size=10)
    add_bullet(doc,
        "Per-modality results are not architecture-controlled. We use vanilla "
        "EEGNet on EEG; other papers use stronger EEG encoders. The fair comparison "
        "is on fusion accuracy.",
        size=10)
    add_bullet(doc,
        "Code availability matters for reproducibility. Hyper-MML, EEG-MoCE, "
        "LEREL, and Guragain do not release code — their numbers cannot be "
        "independently verified.",
        size=10)
    add_bullet(doc,
        "Our reported 80.2%/84.7% fusion accuracies use the same EAV split as the "
        "dataset paper (h_idx=56). Hyperparameters: AST 5+5 epochs, ViT 3+2 epochs, "
        "EEGNet 350 epochs, fusion 50 epochs, batch 32, AdamW. Hardware: Colab Pro L4.",
        size=10)

    # ==== Positioning ====
    add_heading(doc, "7.  Positioning Statement (suggested defence wording)", level=1)
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.5)
    p.paragraph_format.right_indent = Cm(0.5)
    run = p.add_run(
        "\"The EAV dataset, released in September 2024, currently has three "
        "published trimodal fusion baselines: AMERL (70.86%), Hyper-MML (76.65%, "
        "current SOTA), and EEG-MoCE (75.88%). Our cross-attention fusion reaches "
        "80.2% on the same per-subject evaluation protocol, and our modality-dropout "
        "extension reaches 84.7%. Additionally, we contribute three capabilities "
        "no prior EAV work demonstrates: softhard modality dropout training, a "
        "working zero-EEG inference path (81.5% in EEG-absent demo conditions), "
        "and the first cross-modal coherence analysis on the dataset — a 5×5 "
        "suppression matrix capturing 401 events across 5,040 trials.\""
    )
    run.italic = True
    run.font.size = Pt(11)

    # ==== References ====
    add_heading(doc, "8.  References", level=1)
    refs = [
        "Lee, M-H., Shomanov, A., Begim, B., Kabidenova, Z., Nyssanbay, A., Yazici, "
        "A., Lee, S-W. (2024). EAV: EEG-Audio-Video Dataset for Emotion Recognition "
        "in Conversational Contexts. Scientific Data 11:1026. "
        "https://doi.org/10.1038/s41597-024-03838-4",

        "Yin, K., Shin, H-B., Li, D., Lee, S-W. (2024). Attention-based Multimodal "
        "Emotion Recognition on EAV (AMERL). arXiv 2411.00822. "
        "Code: github.com/Kang1121/bci-winter-2025",

        "Kang, Z., Li, Y., Gong, S., Zeng, W., Yan, H., Bian, L., Siok, W.T., Wang, "
        "N. (2025). Hypergraph Multi-Modal Learning for EEG-based Emotion "
        "Recognition in Conversation (Hyper-MML). arXiv 2502.21154.",

        "EEG-MoCE: EEG-Based Multimodal Learning via Hyperbolic Mixture-of-Curvature "
        "Experts (2026). arXiv 2604.12579.",

        "LEREL: Lipschitz Continuity-Constrained Emotion Recognition Ensemble "
        "Learning For Electroencephalography (2025). arXiv 2504.09156.",

        "Guragain, A. (2026). Attention Isn't All You Need for Emotion Recognition: "
        "Domain Features Outperform Transformers on the EAV Dataset. arXiv "
        "2601.22161.",
    ]
    for r in refs:
        p = doc.add_paragraph(r)
        p.paragraph_format.left_indent = Cm(0.5)
        p.paragraph_format.first_line_indent = Cm(-0.5)
        for run in p.runs:
            run.font.size = Pt(9)

    doc.save(OUT_PATH)
    print(f"Wrote {OUT_PATH}")
    print(f"Paragraphs: {len(doc.paragraphs)}  Tables: {len(doc.tables)}")


if __name__ == "__main__":
    build()
