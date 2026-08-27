#!/usr/bin/env python3
"""Generate the extended-deadline master plan as a formatted .docx file.

Outputs thesis_extended_plan.docx in the project root.

Contents:
    1. Reformulated thesis framework (post-pivot to coherence detection)
    2. Current project status (Days 1-5 summary, outstanding gaps)
    3. Necessary remaining steps
    4. 10-day plan (table + per-day elaboration), running 2026-05-24 to 2026-06-02
    5. Risk and contingency analysis
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

from generate_report import (
    _para, _bullet, _numbered, _format_table, _insert_toc,
)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "thesis_extended_plan.docx"

START_DATE = date(2026, 5, 24)
END_DATE = START_DATE + timedelta(days=9)   # 10-day window, inclusive


# ---------------------------------------------------------------------------
# Title page + TOC
# ---------------------------------------------------------------------------

def build_title_page(doc: Document) -> None:
    for _ in range(6):
        doc.add_paragraph()
    _para(doc, "Master Plan — Extended Schedule",
          bold=True, size=24, align=WD_ALIGN_PARAGRAPH.CENTER)
    _para(doc, "Cross-Modal Affective Coherence Using EEG, Audio and Video",
          italic=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER)
    for _ in range(6):
        doc.add_paragraph()
    _para(doc, "Author", bold=True, size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    _para(doc, "Shafin", size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()
    _para(doc, "Institution", bold=True, size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    _para(doc, "[Institution name — to be filled in]",
          size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()
    _para(doc, "Date Issued", bold=True, size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    _para(doc, START_DATE.strftime("%B %d, %Y"),
          size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    _para(doc, "Plan Window", bold=True, size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    _para(doc, f"{START_DATE.strftime('%B %d')} – {END_DATE.strftime('%B %d, %Y')}",
          size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_page_break()


def build_toc(doc: Document) -> None:
    _para(doc, "Table of Contents", bold=True, size=18)
    doc.add_paragraph()
    _insert_toc(doc)
    doc.add_page_break()


# ---------------------------------------------------------------------------
# Section 1 — Reformulated framework
# ---------------------------------------------------------------------------

def build_section_1(doc: Document) -> None:
    doc.add_heading("1. Reformulated Thesis Framework", level=1)

    doc.add_heading("1.1 Context and Reason for Reformulation", level=2)
    doc.add_paragraph(
        "The project was originally scoped on Day 1 as a reproduction of the "
        "trimodal emotion-recognition methodology of Lee, Kim and Kim "
        "(Bioengineering, 2024) on the EAV dataset. As development progressed "
        "and several practical constraints came into focus, the thesis was "
        "deliberately reframed on Day 5 to a more defensible and novel form: "
        "the study of cross-modal affective coherence. Two specific issues "
        "drove the reformulation. First, the ultimate practical scenario "
        "originally envisioned — continuous, per-second emotion tracking on "
        "in-the-wild video such as movie scenes — is incompatible with the "
        "EEG modality, because EEG cannot be collected from arbitrary "
        "audiovisual content. Second, EAV is a controlled-condition, "
        "cue-based laboratory dataset, and its emotions are necessarily "
        "elicited rather than spontaneous. Together these mean that any "
        "honest application of an EAV-trained model to in-the-wild content "
        "would be a stretch of validity that the available time did not "
        "permit closing."
    )
    doc.add_paragraph(
        "The reformulated thesis exploits, rather than apologises for, the "
        "fact that EEG is an internal, involuntary signal while audio and "
        "video are external, voluntary signals. Where they agree, the "
        "subject is producing a coherent affective response. Where they "
        "diverge, the divergence itself is the signal: a subject who "
        "appears outwardly calm while their physiology suggests distress "
        "is exhibiting affective incoherence, and that incoherence is "
        "clinically and behaviourally meaningful (associated, in the "
        "literature, with depression, anxiety, alexithymia, post-traumatic "
        "stress disorder, and other conditions involving emotion "
        "suppression). The thesis develops the methodology to detect and "
        "characterise such incoherence within the EAV dataset and frames "
        "clinical-population validation as Phase 2 future work."
    )

    doc.add_heading("1.2 Working Title", level=2)
    doc.add_paragraph(
        "Two candidate titles, either of which is suitable for the "
        "submitted cover page. The formal variant emphasises the scientific "
        "construct; the more direct variant is preferred for the defence "
        "slide and any short oral abstract."
    )
    _bullet(doc, "Formal: *Cross-Modal Affective Coherence: A Multimodal "
                 "Framework for Detecting Discrepancies Between Externally "
                 "Expressed and Internally Experienced Emotion Using EEG, "
                 "Audio and Video.*")
    _bullet(doc, "Direct: *When the Face Lies but the Brain Doesn’t: "
                 "Multimodal Detection of Suppressed and Mismatched "
                 "Affective States.*")

    doc.add_heading("1.3 Research Questions", level=2)
    doc.add_paragraph(
        "The thesis addresses five questions, each of which corresponds to "
        "a specific empirical contribution in the results section."
    )
    _numbered(doc,
        "Per-modality baseline. Can each of the audio, video and EEG "
        "modalities be classified into the five EAV emotion categories at "
        "a rate that meaningfully exceeds the EAV paper’s published "
        "per-modality baselines?"
    )
    _numbered(doc,
        "Fusion gain. Does multimodal fusion improve classification over "
        "the best single modality, and by how much?"
    )
    _numbered(doc,
        "Architectural contribution. Does learned cross-modal attention "
        "fusion outperform a naïve softmax-averaging late-fusion baseline?"
    )
    _numbered(doc,
        "Coherence as signal. Do the three modalities diverge more on "
        "trials where the subject is actively producing an emotional "
        "expression (Speak phase) than on trials where the subject is "
        "passively receiving one (Listen phase), and does that divergence "
        "vary systematically by emotion category and by subject?"
    )
    _numbered(doc,
        "Robustness. Does the system continue to produce meaningful "
        "predictions when one modality (specifically EEG) is unavailable "
        "at inference time?"
    )

    doc.add_heading("1.4 Architectural Overview", level=2)
    doc.add_paragraph(
        "The complete system comprises three layers: per-modality "
        "encoders, a trimodal fusion module, and two output heads."
    )
    _bullet(doc,
        "Per-modality encoders. The Audio Spectrogram Transformer (AST, "
        "pre-trained on AudioSet) encodes audio. A facial-emotion "
        "fine-tuned Vision Transformer (ViT) encodes video. EEGNet, a "
        "compact convolutional network designed for EEG, is trained from "
        "scratch on the EAV data. Each encoder produces a single pooled "
        "feature vector per clip."
    )
    _bullet(doc,
        "Trimodal Attention Fusion. The three pooled vectors are "
        "projected to a common embedding dimension, augmented with "
        "learnable modality embeddings, and stacked as a three-token "
        "sequence. A two-layer transformer encoder applies self-attention "
        "over the modality tokens, after which the tokens are "
        "mean-pooled to a single fused vector. The module is implemented "
        "in `fusion/trimodal_attention.py` and tested with synthetic "
        "inputs in the same file."
    )
    _bullet(doc,
        "Dual output heads. The fused vector feeds two heads: a "
        "five-way emotion classifier, and a coherence-metric computation "
        "(currently pairwise symmetric Kullback–Leibler divergence "
        "between the three per-modality softmaxes, computed at inference "
        "rather than learned)."
    )
    doc.add_paragraph(
        "Two architectural commitments, decided on Day 5 and recorded in "
        "CLAUDE.md, are load-bearing for the project’s long-term "
        "extensibility and for the demo. First, the fusion module’s "
        "`forward()` method returns a dictionary that includes the three "
        "per-modality post-attention embeddings as named keys; this "
        "permits a future temporal head (LSTM, Residual-TCN or temporal "
        "transformer) to be attached without rewriting the fusion module. "
        "Second, `forward(x_audio, x_visual, x_eeg)` is required to "
        "produce finite outputs when `x_eeg` is a tensor of zeros; the "
        "Day-7 / Day-11 modality-dropout training step is what makes those "
        "outputs *useful*, but the architecture must permit them from "
        "Day 4 onwards."
    )

    doc.add_heading("1.5 Dataset Reference", level=2)
    doc.add_paragraph(
        "The EAV dataset (Lee, Shomanov, Kabidenova and Yazici, "
        "Scientific Data, 2024) comprises synchronised 30-channel EEG, "
        "audio and video recordings from 42 participants. Each participant "
        "contributed 200 interactions across five emotion categories — "
        "Neutral, Sadness, Anger, Happiness, Calmness — within a "
        "cue-based listening-and-speaking protocol. After preprocessing "
        "into 5-second windows, each subject has 280 training and 120 test "
        "trials per modality (per the EAV repository’s `h_idx=56` split). "
        "The Listen and Speak phases of each trial pair carry directly "
        "onto the coherence research question and require that the "
        "preprocessing preserve the per-trial task tag — a step "
        "scheduled for Day 1 of the new plan."
    )


# ---------------------------------------------------------------------------
# Section 2 — Current status
# ---------------------------------------------------------------------------

def build_section_2(doc: Document) -> None:
    doc.add_heading("2. Current Project Status", level=1)

    doc.add_heading("2.1 Summary of Work Completed (Days 1–5)", level=2)
    doc.add_paragraph(
        "Five development days have been spent before the deadline "
        "extension. Each day produced concrete artefacts that are now "
        "part of the project’s working state."
    )
    rows = [
        ("Day 1", "2026-05-19",
         "Workflow established (Mac for code, Colab for compute, Drive "
         "for persistence, GitHub for sync). Three-subject smoke test "
         "across all three modalities completed."),
        ("Day 2", "2026-05-20",
         "Full-budget per-modality training on three subjects. Two "
         "non-trivial bugs in the EAV repository fixed (softmax-CE "
         "composition; train/eval mode in `Trainer_uni`). EEG accuracy "
         "rose from a chance-level mean of 24.4% on Day 1 to a "
         "meaningfully-above-baseline mean of 50.6% on Day 2."),
        ("Day 3", "2026-05-21",
         "Naïve late-fusion baseline computed on three subjects using "
         "Day-2 logits. Mean-softmax fusion achieved 81.9%, a 6.7 "
         "percentage-point gain over the best single modality. "
         "Per-trial pairwise Kullback–Leibler divergence added to the "
         "Day-3 driver for downstream coherence analysis."),
        ("Day 4", "2026-05-22",
         "`fusion/trimodal_attention.py` created: 2.28-million-parameter "
         "self-attention fusion over three modality tokens. Three "
         "contract tests passed (full-modality forward; zero-EEG "
         "forward; gradient flow to all parameter groups). Project "
         "framing pivoted to cross-modal affective coherence; CLAUDE.md "
         "updated."),
        ("Day 5", "2026-05-23",
         "Day-5 pipeline executed on three subjects on a Tesla T4 (and "
         "subsequently moved to a local RTX 4080). Cross-attention "
         "fusion produced a three-subject mean of 0.800, approximately "
         "1.9 percentage points below the Day-3 naïve baseline and with "
         "substantial between-subject variance. Iteration on the "
         "fusion-module hyperparameters has begun."),
    ]
    table = doc.add_table(rows=len(rows) + 1, cols=3)
    table.rows[0].cells[0].text = "Day"
    table.rows[0].cells[1].text = "Date"
    table.rows[0].cells[2].text = "Outcome"
    for i, (d, dt, outcome) in enumerate(rows, start=1):
        table.rows[i].cells[0].text = d
        table.rows[i].cells[1].text = dt
        table.rows[i].cells[2].text = outcome
    _format_table(table)
    _para(doc, "Table 1. Summary of Days 1 to 5 of the original schedule.",
          italic=True, size=10, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()

    doc.add_heading("2.2 Empirical Results Available Today", level=2)
    doc.add_paragraph(
        "The three-subject pilot has produced the following measured "
        "accuracies, summarised here so that the remaining work can be "
        "calibrated against them. All accuracies are five-class "
        "classification, where chance is 20 per cent and the EAV paper "
        "baselines are 36.7% (SCNN audio), 52.8% (DeepFace vision) and "
        "36.7% (EEGNet)."
    )
    results = [
        ("Audio (AST)",          "61.1%", "above SCNN baseline by 24 pp"),
        ("Vision (ViT)",         "75.6%", "above DeepFace baseline by 23 pp"),
        ("EEG (EEGNet)",         "50.6%", "above EEGNet baseline by 14 pp"),
        ("Naïve late fusion",    "81.9%", "+6.3 pp over best single modality (vision)"),
        ("Cross-attention fusion", "80.0%", "−1.9 pp vs naïve baseline; high variance (std 0.10)"),
    ]
    table = doc.add_table(rows=len(results) + 1, cols=3)
    table.rows[0].cells[0].text = "Method"
    table.rows[0].cells[1].text = "Mean (3 subj.)"
    table.rows[0].cells[2].text = "Note"
    for i, (m, acc, note) in enumerate(results, start=1):
        table.rows[i].cells[0].text = m
        table.rows[i].cells[1].text = acc
        table.rows[i].cells[2].text = note
    _format_table(table)
    _para(doc, "Table 2. Measured per-method accuracy on the three-subject "
                "pilot. All numbers are means across subjects 1, 2 and 3.",
          italic=True, size=10, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()

    doc.add_heading("2.3 Outstanding Issues and Methodological Risks", level=2)
    doc.add_paragraph(
        "Several open issues are blockers for the headline contribution "
        "and must be resolved before the remaining work has scientific "
        "validity."
    )
    _numbered(doc,
        "Trial alignment between modalities is unverified. The EAV "
        "repository’s audio and vision preprocessors keep only Speaking "
        "clips, while the EEG preprocessor selects classes [1, 3, 5, 7, 9] "
        "with a source comment that ambiguously says ‘listening’. "
        "Whether the three modalities’ test trials correspond to the "
        "same underlying events is currently unproven. The Day-3 "
        "alignment check passed only because `EAVDataSplit` sorts trials "
        "by class, so the class label sequences match even if the "
        "underlying trials do not. This must be resolved on Day 1 of the "
        "new schedule, before any further computation is committed."
    )
    _numbered(doc,
        "Cross-attention fusion under-performs naïve late fusion. On "
        "three subjects the cross-attention model lost by 1.9 percentage "
        "points to mean-softmax fusion. The three plausible causes "
        "(insufficient regularisation, model too large for available "
        "training samples, suboptimal best-test checkpointing) are all "
        "tractable; a small hyperparameter sweep is scheduled for Day 2."
    )
    _numbered(doc,
        "The full forty-two-subject rollout has not yet been run. Only "
        "three subjects have per-modality state dictionaries and "
        "extracted features. The remaining thirty-nine subjects require "
        "scheduling against the available compute budget (the 4080 with "
        "twelve-hour session limits)."
    )
    _numbered(doc,
        "Per-trial task tags (Listen versus Speak) are not preserved "
        "through preprocessing. The headline coherence analysis "
        "requires these tags, which means the preprocessing scripts "
        "must be modified to emit an additional metadata column, and "
        "the pickles must be regenerated."
    )
    _numbered(doc,
        "Modality-dropout training is not yet implemented. Without it, "
        "the inference-time zero-EEG forward path (which the demo "
        "depends on) produces architecturally-valid but "
        "semantically-undefined predictions. This is the difference "
        "between ‘the model permits missing EEG’ and ‘the model handles "
        "missing EEG sensibly’."
    )
    _numbered(doc,
        "No subject-independent (leave-one-subject-out) evaluation has "
        "been run. The current numbers are all within-subject and "
        "therefore biased upward."
    )
    _numbered(doc,
        "No statistical testing has been applied to method comparisons "
        "yet. Any claim of one method beating another requires a paired "
        "test across subjects."
    )


# ---------------------------------------------------------------------------
# Section 3 — Necessary remaining steps
# ---------------------------------------------------------------------------

def build_section_3(doc: Document) -> None:
    doc.add_heading("3. Necessary Remaining Steps", level=1)
    doc.add_paragraph(
        "The remaining work falls into nine workstreams. Each is described "
        "below at the level of detail required to plan and estimate it. "
        "Section 4 sequences these workstreams into the ten-day window."
    )

    doc.add_heading("3.1 Methodology Cleanup", level=2)
    doc.add_paragraph(
        "Three foundational corrections are needed before further "
        "experiments are run. First, the Listen/Speak task tag must be "
        "preserved through preprocessing for all three modalities; this "
        "requires editing `Dataload_audio.py`, `Dataload_vision.py` and "
        "`Dataload_eeg.py` to emit an additional per-trial metadata "
        "column, then regenerating the pickle files. Second, a held-out "
        "validation split must be introduced so that best-epoch "
        "checkpointing does not contaminate the test set; the cleanest "
        "implementation is to take a small fraction of the training "
        "trials (per subject, per class) as a validation fold. Third, "
        "trial alignment across modalities must be verified empirically "
        "by checking that the same subject-trial-emotion combination "
        "appears in all three modalities’ pickles."
    )

    doc.add_heading("3.2 Fusion Module Iteration", level=2)
    doc.add_paragraph(
        "The Day-5 cross-attention result of 0.800 on three subjects is "
        "below the Day-3 naïve baseline of 0.819 and must be improved "
        "before any large-scale rollout. Three hyperparameter "
        "configurations will be tested on the cached three-subject "
        "features (so iteration costs minutes rather than hours): a more "
        "regularised version of the existing model (dropout 0.3, weight "
        "decay 1e-3, learning rate 5e-4, 150 epochs); a smaller version "
        "(d_model=128, num_layers=1, num_heads=4); and a concat-then-MLP "
        "baseline as a sanity check that the attention mechanism is "
        "doing meaningful work. The best-performing configuration "
        "becomes the production fusion architecture."
    )

    doc.add_heading("3.3 Scale to Forty-Two Subjects", level=2)
    doc.add_paragraph(
        "All remaining experiments depend on having per-modality state "
        "dictionaries and features for every subject. With a reduced "
        "epoch budget (AST 5+5 instead of 10+15; ViT 3+2 instead of "
        "10+5; EEGNet retained at 350), per-subject training time on the "
        "RTX 4080 is approximately fifteen minutes including feature "
        "extraction. The full forty-two-subject rollout therefore "
        "requires approximately ten and a half hours of GPU time, "
        "comfortably fitting one twelve-hour session. The pipeline is "
        "already resume-aware at the (subject, modality) granularity."
    )

    doc.add_heading("3.4 Coherence Analysis", level=2)
    doc.add_paragraph(
        "The central novelty of the reformulated thesis lives in this "
        "workstream. With Listen/Speak tags preserved and per-trial "
        "coherence metrics computed (Day-3 work, already in place for "
        "three subjects), the headline experiment is a paired statistical "
        "test of pairwise modality disagreement between Speak and Listen "
        "trials, across all forty-two subjects. Secondary analyses "
        "include a per-emotion breakdown of mean coherence (under the "
        "hypothesis that high-arousal emotions such as Anger produce "
        "more cross-modally coherent expressions than low-arousal ones "
        "such as Calmness) and a per-subject breakdown identifying "
        "subjects whose expressions are systematically incoherent."
    )

    doc.add_heading("3.5 Modality-Dropout Training", level=2)
    doc.add_paragraph(
        "The fusion module must be retrained with the softhard "
        "modality-dropout scheme adapted from Chumachenko, Iosifidis "
        "and Gabbouj (ICPR, 2022) and extended to three modalities. Each "
        "training batch contains a mix of full-modality, audio-only, "
        "vision-only, and audio-plus-vision-without-EEG samples; the "
        "model thereby learns to make sensible predictions when one or "
        "more modalities are absent. This is the precondition for the "
        "demo (which feeds zero-EEG at inference) to produce defensible "
        "outputs. Robustness is then quantified by comparing the "
        "full-modality and zero-EEG accuracies on the test set."
    )

    doc.add_heading("3.6 MERCL Contrastive Pre-Training (Stretch)", level=2)
    doc.add_paragraph(
        "If schedule permits, the three contrastive losses of Lee et al. "
        "(2024) — Intra-Modal Contrastive Learning (AMCL), Inter-Modal "
        "Contrastive Learning (EMCL) and Sample-wise Multimodal "
        "Alignment Contrastive Learning (SMCL) — will be implemented "
        "and used to pre-train the encoders for a small number of "
        "epochs before fine-tuning the cross-attention fusion module. "
        "The expected improvement from contrastive pre-training, "
        "extrapolating from the paper’s ablation study, is approximately "
        "four percentage points. This workstream is the first item to "
        "be cut if the schedule slips."
    )

    doc.add_heading("3.7 Subject-Independent Evaluation (LOSO)", level=2)
    doc.add_paragraph(
        "All accuracies reported so far are within-subject, where the "
        "training and test sets come from the same individual. "
        "Leave-one-subject-out cross-validation (LOSO) is the standard "
        "credibility test in affective computing and is essential for "
        "claiming any form of generalisation. The implementation cost is "
        "compute-heavy: it multiplies the training cost by the number of "
        "held-out subjects. A pragmatic approximation is five-fold "
        "across-subject cross-validation, which gives statistically "
        "interpretable results at a fifth of the cost. The choice "
        "between full LOSO and 5-fold will be made on Day 7 based on "
        "remaining GPU budget."
    )

    doc.add_heading("3.8 Demonstration", level=2)
    doc.add_paragraph(
        "A file-based demo will be produced for the defence. Input: a "
        "personal twenty-to-thirty-second video clip with synchronous "
        "audio. Output: a copy of the same video with a coloured "
        "per-segment emotion label and a small five-bar confidence "
        "indicator rendered as an overlay, plus a separate coherence "
        "indicator visualising when the modalities disagree. EEG is fed "
        "as zeros at inference time; this is the use case the "
        "modality-dropout training enables. The implementation uses "
        "OpenCV for frame-level overlay and FFmpeg for re-attaching the "
        "original audio track."
    )

    doc.add_heading("3.9 Writing and Defence Preparation", level=2)
    doc.add_paragraph(
        "The thesis manuscript and the defence slide deck are produced "
        "in parallel over the final three days. The manuscript follows "
        "the standard format: Introduction, Related Work, Methods, "
        "Experiments, Results, Discussion, Limitations, Future Work, "
        "Conclusion. The Limitations section explicitly acknowledges "
        "that the coherence methodology is validated behaviourally (via "
        "the Listen-vs-Speak contrast) but not clinically, and that "
        "in-the-wild evaluation is Phase 2 future work. The slide deck "
        "structures the defence around the five research questions from "
        "Section 1.3 of this plan."
    )


# ---------------------------------------------------------------------------
# Section 4 — 10-day plan
# ---------------------------------------------------------------------------

PLAN = [
    (1, "Methodology cleanup, fusion iteration on cached features",
     "Verify trial alignment across modalities; modify preprocessing to "
     "preserve Listen/Speak tag; run three fusion-hyperparameter "
     "configurations on the cached three-subject features and pick the best.",
     "Updated preprocessing scripts; alignment verification report; chosen "
     "fusion configuration; updated CLAUDE.md."),
    (2, "Re-preprocess and start forty-two-subject rollout",
     "Regenerate per-subject pickles with task tag preserved. Launch "
     "Stage A (per-modality training) for all forty-two subjects on the "
     "4080 with reduced epoch budgets.",
     "New pickles in Drive; state dictionaries for first ~24 subjects."),
    (3, "Complete per-modality rollout; extract features",
     "Finish Stage A for remaining subjects; run Stage B feature "
     "extraction for all subjects; begin Stage C fusion training.",
     "All 42 state dictionaries and feature files cached in Drive; "
     "fusion training in progress."),
    (4, "Complete fusion training; naive-fusion baseline on 42 subjects",
     "Finish cross-attention fusion training for all subjects. Compute "
     "the Day-3-style naïve late-fusion baseline on the full set so the "
     "two methods are directly comparable across forty-two subjects.",
     "results/day8_fusion_42subjects.csv with per-method accuracy; "
     "first comparison table for the thesis."),
    (5, "Coherence analysis — Listen vs Speak",
     "Using the preserved task tags, compute paired Speak-vs-Listen "
     "tests of pairwise modality disagreement across all 42 subjects. "
     "Per-emotion and per-subject breakdowns.",
     "results/day9_coherence_analysis.csv; statistical-test report; "
     "first draft of the coherence figure for the thesis."),
    (6, "Modality-dropout training",
     "Implement the softhard modality-dropout scheme. Retrain the "
     "fusion module with mixed full / single-modality / "
     "audio-plus-vision batches. Evaluate full-modality and zero-EEG "
     "accuracies side by side.",
     "Modality-dropout-trained model checkpoint; "
     "results/day10_robustness.csv; robustness table for thesis."),
    (7, "Subject-independent evaluation (LOSO or 5-fold)",
     "Re-evaluate the best fusion model under a subject-independent "
     "protocol. Run full LOSO if compute permits; otherwise 5-fold "
     "across subjects.",
     "results/day11_loso.csv; subject-independent mean ± std numbers."),
    (8, "Demo build and contrastive pre-training (if time)",
     "Build the file-based emotion-overlay demo using the "
     "modality-dropout-trained model. Record a personal demo clip. "
     "If schedule permits, implement and run MERCL contrastive "
     "pre-training as a final empirical addition.",
     "demo_inference.py and demo_overlay.py scripts; rendered demo "
     "MP4; optional MERCL results."),
    (9, "Thesis writing — methods and results",
     "Draft Methods, Experiments and Results chapters in full. "
     "Produce architecture diagram; finalise all result tables.",
     "Thesis draft sections 1–4 complete (introduction, related work, "
     "methods, experiments)."),
    (10, "Thesis writing — discussion and polish; defence prep",
     "Draft Discussion, Limitations and Conclusion. Build defence "
     "slide deck. Final read-through of the manuscript and rendered "
     "report. Push final results to GitHub.",
     "Final thesis PDF; defence slide deck; final commit on main "
     "branch."),
]


def build_section_4(doc: Document) -> None:
    doc.add_heading("4. Ten-Day Plan (2026-05-24 to 2026-06-02)", level=1)

    doc.add_heading("4.1 Plan at a Glance", level=2)
    doc.add_paragraph(
        "The remaining work is sequenced across ten consecutive days. "
        "Each day has a single principal goal, a set of concrete tasks, "
        "and a defined expected output. Days 1 to 4 build the empirical "
        "foundation by scaling the work to all forty-two subjects; "
        "Days 5 to 7 produce the headline contributions (coherence "
        "analysis, modality-dropout robustness, subject-independent "
        "evaluation); Days 8 to 10 deliver the demo and the thesis "
        "manuscript. Table 3 summarises; subsequent subsections "
        "elaborate."
    )
    doc.add_paragraph()

    table = doc.add_table(rows=len(PLAN) + 1, cols=5)
    headers = ("Day", "Date", "Goal", "Key Tasks", "Expected Output")
    for j, h in enumerate(headers):
        table.rows[0].cells[j].text = h
    for i, (day, goal, tasks, output) in enumerate(PLAN, start=1):
        dt = (START_DATE + timedelta(days=day - 1)).strftime("%a %b %d")
        table.rows[i].cells[0].text = str(day)
        table.rows[i].cells[1].text = dt
        table.rows[i].cells[2].text = goal
        table.rows[i].cells[3].text = tasks
        table.rows[i].cells[4].text = output
    _format_table(table)
    _para(doc, f"Table 3. Ten-day plan, {START_DATE.strftime('%b %d')} to "
                f"{END_DATE.strftime('%b %d, %Y')}.",
          italic=True, size=10, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()

    doc.add_heading("4.2 Day-by-Day Elaboration", level=2)

    elaborations = [
        ("Day 1 (Sun May 24) — Methodology cleanup and fusion iteration",
         "The day’s output gates everything that follows. First, write a "
         "small diagnostic script that, for each of the three already-"
         "processed subjects, reads back the original EAV trial file "
         "names and verifies that the train/test indices in the pickle "
         "correspond to the same subject-emotion-task combinations across "
         "audio, vision and EEG. This produces an explicit "
         "alignment-or-misalignment finding. Second, modify the three "
         "`Dataload_*.py` scripts to preserve a per-trial `task` column "
         "(Listen/Speak), so that future regeneration of the pickles "
         "carries the information forward. Third, iterate on the fusion "
         "module hyperparameters using the cached three-subject features: "
         "Config 1 with stronger regularisation, Config 2 with a smaller "
         "model, Config 3 as a concat-MLP sanity check. Pick the "
         "highest-mean configuration and commit it as the production "
         "fusion architecture. The output of Day 1 is a clean, validated, "
         "single-configuration system ready to scale."),
        ("Day 2 (Mon May 25) — Re-preprocess and launch 42-subject Stage A",
         "Run the modified preprocessing across all 42 subjects to "
         "produce new pickles with task tags preserved. This is a one-off "
         "cost of one to two hours of CPU work; output goes to a new "
         "subdirectory in Drive (`Input_images_v2/`) so the original "
         "pickles remain untouched as a fall-back. Once pickles are "
         "ready, kick off Stage A of the day5 pipeline on the 4080 with "
         "reduced epoch budgets (`AUD_EPOCHS_FT=5`, `VIS_EPOCHS_FT=2`). "
         "Per-subject Stage A time is approximately ten minutes; in a "
         "single twelve-hour session approximately seventy subjects fit, "
         "so all forty-two will complete within Day 2."),
        ("Day 3 (Tue May 26) — Feature extraction and start fusion training",
         "All Stage A state dictionaries should be available from Day 2. "
         "Run Stage B feature extraction for all forty-two subjects "
         "(approximately one minute per subject; less than an hour "
         "total). Begin Stage C fusion training: at approximately two "
         "minutes per subject this completes within ninety minutes for "
         "the whole cohort. Optionally also re-run the Day-3 naïve-"
         "fusion baseline script on the new forty-two-subject logit "
         "files so that the two methods are directly comparable on the "
         "same data."),
        ("Day 4 (Wed May 27) — Finalise classification numbers",
         "By the end of Day 4 the thesis should have its complete per-"
         "modality, naïve-fusion and cross-attention-fusion accuracies "
         "for all forty-two subjects. Run paired statistical tests "
         "(Wilcoxon signed-rank, paired t-test) between method pairs. "
         "Produce the first complete results table for the thesis "
         "(Table 1 of the eventual manuscript). Decide which fusion "
         "method (cross-attention vs naïve) is the production model for "
         "downstream coherence and demo work."),
        ("Day 5 (Thu May 28) — Coherence analysis",
         "The headline experiment of the reformulated thesis. For every "
         "test trial of every subject, compute the three pairwise "
         "symmetric KL divergences between the per-modality softmax "
         "outputs (Day-3 work generalised to all subjects). Tag each "
         "trial as Listen or Speak using the Day-1 metadata. Run a "
         "paired Wilcoxon test across subjects of mean Speak-minus-Listen "
         "disagreement. Hypothesise (and test) that Speak coherence is "
         "lower than Listen coherence — that voluntarily-produced "
         "expressions diverge from internal physiology more than "
         "involuntarily-elicited ones do. Produce a per-emotion "
         "breakdown (which emotions are produced most and least "
         "coherently) and a per-subject breakdown (which subjects are "
         "systematically incoherent). This day’s output is the empirical "
         "core of the thesis."),
        ("Day 6 (Fri May 29) — Modality-dropout training",
         "Implement the softhard scheme: each training batch is the "
         "concatenation of four sub-batches — full-modality, "
         "audio-with-zeroed-EEG, vision-with-zeroed-EEG-and-zeroed-audio, "
         "and EEG-with-zeroed-audio-and-zeroed-vision — with the "
         "appropriate target labels duplicated. Retrain the fusion "
         "module. Evaluate the resulting model under three inference "
         "conditions: full modalities, EEG zeroed, both audio and EEG "
         "zeroed. Report each accuracy across all subjects. The "
         "zero-EEG number is the headline robustness claim and is what "
         "the demo will run on."),
        ("Day 7 (Sat May 30) — Subject-independent evaluation",
         "Re-run the best fusion model under a leave-one-subject-out "
         "protocol (or, if compute is short, 5-fold cross-subject). For "
         "each fold, train on the held-in subjects only and evaluate on "
         "the held-out subject. Aggregate by mean and standard deviation "
         "across folds. This is the credibility check that the rest of "
         "the thesis depends on; without it the within-subject numbers "
         "in earlier tables cannot be defended as evidence of "
         "generalisation. If full LOSO would exceed the remaining "
         "compute budget, fall back to 5-fold and note the choice "
         "explicitly in the manuscript."),
        ("Day 8 (Sun May 31) — Demo and stretch-goal MERCL",
         "Write `demo_inference.py` (consumes a video file, returns a "
         "sequence of per-second emotion-and-coherence predictions) and "
         "`demo_overlay.py` (renders the predictions onto the original "
         "video using OpenCV, then re-attaches the original audio with "
         "FFmpeg). Record a personal twenty-to-thirty-second demo clip. "
         "If MERCL contrastive pre-training is still on the schedule, "
         "spend the second half of the day implementing AMCL, EMCL and "
         "SMCL losses and running a brief pre-training run; otherwise "
         "defer MERCL to future work and use the time for demo polish."),
        ("Day 9 (Mon Jun 1) — Thesis writing: methods, experiments, results",
         "Sit at a desk and write. The Methods chapter describes the "
         "architecture (with the diagram from Day 4’s notes), the "
         "training procedure, the coherence metric and the modality-"
         "dropout scheme. The Experiments chapter describes the EAV "
         "dataset and the evaluation protocol, including the Listen-vs-"
         "Speak distinction and the subject-independent fold. The "
         "Results chapter presents the tables and figures from Days 4, "
         "5, 6 and 7. The day’s output is a draft of sections 1 to 4 of "
         "the manuscript, complete enough that a careful reader could "
         "reproduce the work from it."),
        ("Day 10 (Tue Jun 2) — Discussion, polish, defence prep",
         "Write the Discussion section, framing the coherence analysis "
         "as the thesis’s headline contribution and the architectural "
         "and robustness results as supporting evidence. Write the "
         "Limitations section explicitly: clinical validation is Phase "
         "2 future work; cross-modal coherence as measured here is "
         "behaviourally validated but not clinically validated; the "
         "model has been evaluated only on the EAV dataset and "
         "generalisation to other multimodal corpora is unproven. "
         "Build the defence slide deck following the structure of the "
         "five research questions. Final read-through of the "
         "manuscript. Push everything to GitHub. The plan window "
         "closes."),
    ]
    for title, body in elaborations:
        doc.add_heading(title, level=3)
        doc.add_paragraph(body)


# ---------------------------------------------------------------------------
# Section 5 — Risk and contingency
# ---------------------------------------------------------------------------

def build_section_5(doc: Document) -> None:
    doc.add_heading("5. Risk and Contingency", level=1)
    doc.add_paragraph(
        "Three risks are large enough to warrant explicit contingency "
        "plans. Each is described together with the trigger that would "
        "cause the contingency to be invoked and the action that "
        "follows."
    )

    doc.add_heading("5.1 Risk — Trial misalignment between modalities is real", level=2)
    doc.add_paragraph(
        "If the Day 1 alignment check finds that audio/vision and EEG "
        "trials do not correspond to the same underlying events (because "
        "of the listening-versus-speaking selection mismatch in the EAV "
        "repository’s preprocessing), then the Day-3 naïve-fusion number "
        "and the Day-5 cross-attention number are both methodologically "
        "questionable, and the central coherence analysis cannot proceed "
        "in its current form. The contingency in this case is to limit "
        "the trimodal portion of the thesis to the subset of trial pairs "
        "that *do* align (which requires re-preprocessing to construct), "
        "and report on the bimodal audio-plus-vision case across all "
        "trials. The thesis would then frame the coherence analysis as "
        "audio-vision-versus-EEG agreement across aligned trial pairs "
        "rather than fully-trimodal. This is a strictly smaller claim, "
        "but a defensible one."
    )

    doc.add_heading("5.2 Risk — Cross-attention fusion cannot beat naïve fusion at scale", level=2)
    doc.add_paragraph(
        "If the forty-two-subject cross-attention fusion result on Day 4 "
        "is still below the naïve-fusion baseline after the Day-1 "
        "hyperparameter iteration, the thesis pivots its emphasis from "
        "‘we propose a fusion architecture that beats baselines’ to ‘we "
        "characterise cross-modal coherence and find that EAV is a "
        "low-headroom regime for learned fusion methods’. The coherence "
        "analysis on Day 5 then becomes the sole headline contribution; "
        "the fusion architecture work is reported as a negative result "
        "with a defensible interpretation. No work is wasted, but the "
        "discussion framing shifts."
    )

    doc.add_heading("5.3 Risk — Compute budget exceeded", level=2)
    doc.add_paragraph(
        "If the forty-two-subject rollout (Days 2–3) takes longer than "
        "expected — for example because the 4080 session disconnects "
        "repeatedly or because per-subject training is slower than the "
        "twelve-minute estimate — the cut order is: first drop MERCL "
        "contrastive pre-training (Day 8 stretch); then reduce LOSO to "
        "5-fold cross-subject (Day 7); then, as a last resort, "
        "complete the per-modality and fusion training on a subset of "
        "twenty to thirty subjects and report results on that subset "
        "explicitly. The headline coherence claim is robust to subject "
        "count above approximately twenty, since the paired statistical "
        "test gains power quickly with n."
    )

    doc.add_heading("5.4 Non-negotiable Items", level=2)
    doc.add_paragraph(
        "Three items are not subject to cuts under any contingency. "
        "First, the Listen-vs-Speak coherence analysis (Day 5) — this is "
        "the thesis novelty and must be reported. Second, subject-"
        "independent evaluation in some form (Day 7) — a thesis without "
        "subject-independent numbers does not pass affective-computing "
        "review. Third, the manuscript itself (Days 9 and 10) — a "
        "thesis defence requires a thesis document."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    doc = Document()

    styles = doc.styles
    styles["Normal"].font.name = "Calibri"
    styles["Normal"].font.size = Pt(11)

    build_title_page(doc)
    build_toc(doc)
    build_section_1(doc)
    doc.add_page_break()
    build_section_2(doc)
    doc.add_page_break()
    build_section_3(doc)
    doc.add_page_break()
    build_section_4(doc)
    doc.add_page_break()
    build_section_5(doc)

    doc.save(OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
