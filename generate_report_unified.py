#!/usr/bin/env python3
"""Generate a unified Day-1-through-current progress report.

Absorbs the still-relevant content from Day 1 (workflow setup, smoke test)
and Day 2 (full-budget per-modality baselines) into the Day-3-to-current
structure, adds four additional challenges from the Day-1 era, and produces
a single canonical progress document.

Output: thesis_progress_unified.docx
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

from generate_report import (
    _para, _bullet, _numbered, _format_table, _insert_toc,
)

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "thesis_progress_unified.docx"


# ---------------------------------------------------------------------------
# Front matter
# ---------------------------------------------------------------------------

def build_title_page(doc: Document) -> None:
    for _ in range(6):
        doc.add_paragraph()
    _para(doc, "Comprehensive Technical Progress Report",
          bold=True, size=24, align=WD_ALIGN_PARAGRAPH.CENTER)
    _para(doc, "Cross-Modal Affective Coherence Using EEG, Audio and Video on the EAV Dataset",
          italic=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER)
    _para(doc, "Coverage: Day 1 through Current Day",
          italic=True, size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    for _ in range(6):
        doc.add_paragraph()
    _para(doc, "Author", bold=True, size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    _para(doc, "Shafin", size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()
    _para(doc, "Institution", bold=True, size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    _para(doc, "[Institution name — to be filled in]",
          size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()
    _para(doc, "Date", bold=True, size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    _para(doc, date.today().strftime("%B %d, %Y"),
          size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_page_break()


def build_toc(doc: Document) -> None:
    _para(doc, "Table of Contents", bold=True, size=18)
    doc.add_paragraph()
    _insert_toc(doc)
    doc.add_page_break()


# ---------------------------------------------------------------------------
# Section 1 — Project Goals
# ---------------------------------------------------------------------------

def build_section_1(doc: Document) -> None:
    doc.add_heading("1. Project Goals", level=1)

    doc.add_heading("1.1 Initial Project Scope", level=2)
    doc.add_paragraph(
        "The project was initially scoped, in mid-May 2026, as the "
        "implementation and evaluation of a trimodal emotion-recognition "
        "framework on the EAV (EEG-Audio-Video) dataset published by Lee, "
        "Shomanov, Kabidenova and Yazici in Scientific Data, 2024. The "
        "intended methodology was a reproduction of the contrastive-"
        "learning-plus-cross-modal-attention framework of Lee, Kim and Kim "
        "(Bioengineering, 2024), which has not previously been evaluated "
        "on EAV. The five emotion categories of EAV — Neutral, Sadness, "
        "Anger, Happiness and Calmness — were the prediction target."
    )

    doc.add_heading("1.2 Project Reformulation", level=2)
    doc.add_paragraph(
        "Mid-way through the development cycle, the project was reformulated "
        "to a more defensible and novel research question: the detection of "
        "cross-modal affective coherence. The reformulation was driven by "
        "two practical realities. First, the ultimate application originally "
        "envisioned — continuous per-second emotion tracking on in-the-wild "
        "video such as movie scenes — is incompatible with the EEG modality, "
        "because EEG cannot be collected from arbitrary audiovisual content. "
        "Second, the EAV dataset is a controlled-condition, cue-based "
        "laboratory corpus, and its emotions are necessarily elicited rather "
        "than spontaneous. The combination meant that any honest application "
        "of an EAV-trained model to in-the-wild content would be a stretch "
        "of validity beyond what the available time could close."
    )
    doc.add_paragraph(
        "The reformulated thesis exploits the fact that EEG is an internal, "
        "involuntary physiological signal while audio and video are external, "
        "voluntary behavioural signals. Where the three modalities agree on "
        "the emotion being expressed, the subject is producing a coherent "
        "affective response. Where the modalities diverge — particularly "
        "where the audio-vision consensus indicates one emotion while EEG "
        "indicates another — the divergence is itself a clinically and "
        "behaviourally meaningful signal, associated in the literature with "
        "emotion suppression, alexithymia, depression, anxiety and several "
        "other conditions involving discordance between felt and displayed "
        "affect. The thesis develops the methodology to detect and "
        "characterise such incoherence within the EAV dataset; clinical-"
        "population validation is framed as Phase 2 future work."
    )

    doc.add_heading("1.3 Working Title and Research Questions", level=2)
    doc.add_paragraph(
        "Two candidate titles are in circulation, either of which is "
        "suitable for the submitted cover page. The formal variant is "
        "preferred for the manuscript itself, while the direct variant is "
        "intended for defence-slide use and short oral abstracts."
    )
    _bullet(doc, "Formal: *Cross-Modal Affective Coherence: A Multimodal "
                 "Framework for Detecting Discrepancies Between Externally "
                 "Expressed and Internally Experienced Emotion Using EEG, "
                 "Audio and Video.*")
    _bullet(doc, "Direct: *When the Face Lies but the Brain Doesn’t: "
                 "Multimodal Detection of Suppressed and Mismatched "
                 "Affective States.*")

    doc.add_paragraph(
        "The reformulated thesis addresses five research questions, each "
        "corresponding to a specific empirical contribution in the results "
        "section."
    )
    _numbered(doc,
        "Per-modality baseline. Can each of the audio, video and EEG "
        "modalities be classified into the five EAV emotion categories at "
        "a rate that meaningfully exceeds the EAV paper's published "
        "per-modality baselines?"
    )
    _numbered(doc,
        "Fusion gain. Does multimodal fusion improve classification over "
        "the best single modality, and by what margin?"
    )
    _numbered(doc,
        "Architectural contribution. Does learned trimodal cross-modal "
        "attention fusion outperform a naïve softmax-averaging late-fusion "
        "baseline at scale?"
    )
    _numbered(doc,
        "Coherence as signal. When the modalities disagree, what "
        "patterns of disagreement emerge, and are those patterns "
        "consistent across the subject population?"
    )
    _numbered(doc,
        "Robustness. Does the system continue to produce meaningful "
        "predictions when one modality (specifically EEG) is unavailable "
        "at inference time?"
    )

    doc.add_heading("1.4 Architectural Commitments", level=2)
    doc.add_paragraph(
        "Two architectural commitments were decided on at the inception of "
        "the fusion design and are load-bearing for the project's long-term "
        "extensibility and for the demonstration component. First, the "
        "fusion module's `forward()` method returns a dictionary that "
        "includes the three per-modality post-attention embeddings as named "
        "keys; this permits a future temporal head (LSTM, Residual-TCN or "
        "temporal transformer) to be attached without rewriting the fusion "
        "module. Second, `forward(x_audio, x_visual, x_eeg)` is required "
        "to produce finite outputs when `x_eeg` is a tensor of zeros; the "
        "modality-dropout training step in the remaining schedule is what "
        "makes those outputs *useful*, but the architecture must permit "
        "them at the level of forward-pass mechanics from the day the "
        "module was built."
    )


# ---------------------------------------------------------------------------
# Section 2 — Implementation and Progress (Day 1 to Current)
# ---------------------------------------------------------------------------

def build_section_2(doc: Document) -> None:
    doc.add_heading("2. Implementation and Progress (Day 1 to Current)", level=1)
    doc.add_paragraph(
        "This section traces the project's development trajectory from "
        "Day 1 (workflow establishment and initial smoke test) through "
        "the current 42-subject rollout and coherence analysis. Each "
        "subsection corresponds to a discrete phase of the project, "
        "with objectives, technical content, and empirical results "
        "where applicable."
    )

    # ---- 2.1 Day 1 ----
    doc.add_heading("2.1 Day 1 — Workflow Establishment and Smoke Test", level=2)
    doc.add_paragraph(
        "Day 1 (2026-05-19) was scoped as a setup and reproduction "
        "milestone. The primary objectives were to establish a hybrid "
        "development workflow combining a local development machine with "
        "cloud-based GPU compute, given that the local hardware (an Apple "
        "M1 MacBook Air) lacks the CUDA support required by the EAV "
        "repository's PyTorch path; to provision persistent storage for "
        "code, data and trained artefacts; to verify that the EAV pickle "
        "feature files were accessible from the chosen compute environment; "
        "and to run a minimal smoke test of all three modalities on at "
        "least three subjects, demonstrating end-to-end pipeline "
        "operation. Day 1 was explicitly not intended to produce final "
        "or paper-quality numbers; its purpose was to validate the "
        "pipeline."
    )
    doc.add_paragraph(
        "The workflow finalised on Day 1 and intended to remain stable for "
        "the remainder of the project was as follows. Source code is "
        "authored locally on the M1 MacBook Air using Visual Studio Code "
        "and synchronised through a private GitHub repository "
        "(`shafinmuffinn/thesis_p2`). Compute is provided by Google Colab "
        "on the free tier (Tesla T4 GPU), with the option to upgrade to "
        "Colab Pro retained for the more compute-intensive days later in "
        "the schedule. Persistent storage is provided by Google Drive, "
        "mounted into the Colab runtime at "
        "`/content/drive/MyDrive/Thesis_EAV/`. EAV pickle features and "
        "trained model checkpoints reside on Drive; the raw 46 GB dataset "
        "is held only transiently when required."
    )
    doc.add_paragraph(
        "A single path-configuration module, `paths.py`, was introduced "
        "at the project root. All filesystem locations are read from "
        "environment variables (`THESIS_ROOT`, `EAV_PICKLES`, "
        "`CHECKPOINTS`, `RESULTS`, `PRETRAINED`), with sensible defaults "
        "for the local machine. The Colab session-start ritual sets these "
        "environment variables to point at the mounted Drive tree, which "
        "means the same code runs unchanged in both environments. A "
        ".gitignore file ensures that bulk data, checkpoints and "
        "Hugging Face caches are never accidentally committed to the "
        "repository."
    )
    doc.add_paragraph(
        "The smoke test used deliberately small training budgets so that "
        "the focus was on validating the pipeline rather than maximising "
        "classification performance: two frozen-backbone epochs followed "
        "by two fine-tuning epochs for the Audio Spectrogram Transformer; "
        "one frozen and one fine-tuning epoch for the facial-emotion "
        "Vision Transformer; and fifty epochs of training from scratch for "
        "EEGNet. The per-subject test accuracies on subjects 1, 2 and 3 "
        "are reported in Table 1 below."
    )
    doc.add_paragraph()

    day1_results = [
        ("01", "49.2%", "62.5%", "30.8%"),
        ("02", "69.2%", "69.2%", "20.8%"),
        ("03", "50.8%", "80.0%", "21.7%"),
        ("Mean", "56.4%", "70.6%", "24.4%"),
    ]
    t1 = doc.add_table(rows=len(day1_results) + 1, cols=4)
    t1.rows[0].cells[0].text = "Subject"
    t1.rows[0].cells[1].text = "Audio acc."
    t1.rows[0].cells[2].text = "Vision acc."
    t1.rows[0].cells[3].text = "EEG acc."
    for i, (sub, a, v, e) in enumerate(day1_results, start=1):
        t1.rows[i].cells[0].text = sub
        t1.rows[i].cells[1].text = a
        t1.rows[i].cells[2].text = v
        t1.rows[i].cells[3].text = e
    _format_table(t1)
    _para(doc, "Table 1. Day-1 smoke-test per-subject test accuracies. "
                "Chance is 20%. EAV baselines: SCNN audio 36.7%, DeepFace "
                "vision 52.8%, EEGNet EEG 36.7%.",
          italic=True, size=10, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()

    doc.add_paragraph(
        "The audio and vision results meaningfully exceeded EAV's "
        "published baselines, validating the smoke test and confirming "
        "that the foundation-model-based encoders (AST pretrained on "
        "AudioSet; ViT pretrained on a facial-emotion dataset) transfer "
        "effectively even with minimal fine-tuning. The EEG result, "
        "however, was statistically indistinguishable from random "
        "guessing across the three pilot subjects (sample size n=120 per "
        "subject), a finding that triggered an extended debugging "
        "investigation traced in Section 3."
    )
    doc.add_paragraph(
        "Several preprocessing-related issues surfaced during Day 1 and "
        "were documented for permanent resolution. These included a "
        "dependency-conflict introduced by `facenet-pytorch`, a "
        "case-sensitivity bug in folder name handling, a forward-hook "
        "implementation defect in the EAV repository's EEGNet code, a "
        "classifier-head-size mismatch in the vision trainer, and a "
        "RAM-exhaustion failure mode in the vision preprocessing pipeline. "
        "Each is documented separately in Section 3."
    )

    # ---- 2.2 Day 2 ----
    doc.add_heading("2.2 Day 2 — Full-Budget Per-Modality Training", level=2)
    doc.add_paragraph(
        "Day 2 (2026-05-20) was scoped as the first full-budget training "
        "day. Where Day 1 had executed an intentionally minimal "
        "smoke-test to validate that the per-modality pipelines could "
        "be made to run end-to-end, Day 2 was concerned with training "
        "each modality at the epoch counts recommended by the EAV "
        "repository and thereby producing the first credible "
        "per-modality baselines on the EAV dataset."
    )
    doc.add_paragraph(
        "Five primary objectives were set for the day. First, train the "
        "AST audio model for ten frozen plus fifteen fine-tuning epochs, "
        "the ViT vision model for ten plus five, and the EEGNet model "
        "for three hundred and fifty epochs, on the same three subjects "
        "used in Day 1. Second, verify that the two EEG-specific bug "
        "fixes applied late on Day 1 — the removal of the trailing "
        "softmax layer from `EEGNet_tor.forward`, and the relocation of "
        "`self.model.train()` inside the epoch loop of "
        "`Trainer_uni.train()` — actually improved EEG accuracy at the "
        "full three-hundred-and-fifty-epoch budget. Third, persist "
        "per-subject test logits to Drive as compressed NumPy archives "
        "for downstream consumption by the Day-3 naïve-fusion baseline "
        "and Day-4 cross-attention fusion module. Fourth, measure "
        "per-subject wall-clock training time on the free-tier T4 GPU to "
        "forecast the time and number of sessions required to scale to "
        "all forty-two subjects. Fifth, decide on the Colab Pro upgrade."
    )
    doc.add_paragraph(
        "Per-subject test accuracies on the three pilot subjects after "
        "full-budget training are reported in Table 2."
    )
    doc.add_paragraph()

    day2_results = [
        ("01", "55.8%", "62.5%", "44.2%"),
        ("02", "71.7%", "80.8%", "54.2%"),
        ("03", "55.8%", "83.3%", "53.3%"),
        ("Mean", "61.1%", "75.6%", "50.6%"),
    ]
    t2 = doc.add_table(rows=len(day2_results) + 1, cols=4)
    t2.rows[0].cells[0].text = "Subject"
    t2.rows[0].cells[1].text = "Audio acc."
    t2.rows[0].cells[2].text = "Vision acc."
    t2.rows[0].cells[3].text = "EEG acc."
    for i, (sub, a, v, e) in enumerate(day2_results, start=1):
        t2.rows[i].cells[0].text = sub
        t2.rows[i].cells[1].text = a
        t2.rows[i].cells[2].text = v
        t2.rows[i].cells[3].text = e
    _format_table(t2)
    _para(doc, "Table 2. Day-2 full-budget per-modality test accuracies on "
                "three pilot subjects. Δ vs Day 1: audio +4.7 pp, vision "
                "+5.0 pp, EEG +26.2 pp.",
          italic=True, size=10, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()

    doc.add_paragraph(
        "The most consequential result of Day 2 was the EEG mean of "
        "50.6%, an improvement of 26.2 percentage points over Day 1's "
        "smoke-test result of 24.4%. On Day 1 EEG accuracy was "
        "statistically indistinguishable from random guessing; on Day 2, "
        "with the two bug fixes applied and the training budget raised "
        "to three hundred and fifty epochs, EEG produced credibly "
        "above-baseline signal across all three subjects. This was "
        "disproportionately important for the project as a whole, "
        "because a fusion model containing a near-random EEG branch "
        "would have been an audio-plus-vision model with extra noise; "
        "the value of the third modality is conditional on it actually "
        "contributing signal."
    )
    doc.add_paragraph(
        "Wall-clock measurements taken during Day 2 produced a "
        "per-subject training time of approximately thirty-four minutes "
        "for audio, forty-three minutes for vision, and one minute for "
        "EEG, for a combined budget of approximately seventy-eight "
        "minutes per subject on the Colab Free T4. Extrapolating naïvely "
        "to all forty-two subjects yielded a total compute requirement "
        "of roughly fifty-five hours, incompatible with Colab Free's "
        "approximately twelve-hour session limit. The decision was "
        "therefore taken to upgrade to Colab Pro before the eventual "
        "forty-two-subject rollout. In practice, the rollout was further "
        "accelerated by reducing the per-modality epoch budgets (Day-6 "
        "of the extended schedule, on the basis of Day-2 evidence that "
        "AST training accuracy reaches 100% by epoch five of "
        "fine-tuning), fitting the full rollout in approximately twelve "
        "hours on Colab Pro L4."
    )
    doc.add_paragraph(
        "Per-subject test logits were saved to "
        "`/content/drive/MyDrive/Thesis_EAV/checkpoints/day2_logits/`, one "
        "compressed NumPy archive per (subject, modality) pair, each "
        "containing the (120, 5) logits matrix and the 120 corresponding "
        "test labels. These files are the direct inputs to Day 3."
    )

    # ---- 2.3 Day 3 ----
    doc.add_heading("2.3 Day 3 — Naïve Late-Fusion Baseline", level=2)
    doc.add_paragraph(
        "Day 3 produced the project's first multimodal result and "
        "established the comparison point against which all subsequent "
        "fusion architectures are measured. Naïve late fusion is the "
        "simplest possible multimodal combination: the per-modality "
        "softmax outputs from the trained Day-2 classifiers are averaged "
        "trial-by-trial, and the argmax of the averaged vector is taken "
        "as the fused prediction. The baseline carries methodological "
        "weight specifically because it is so simple. Any later, more "
        "sophisticated fusion architecture that fails to beat naïve "
        "averaging is, in any meaningful sense, not learning to fuse."
    )
    doc.add_paragraph(
        "The Day-3 driver, implemented in `day3_late_fusion.py`, consumes "
        "the per-modality test logits saved by Day 2. For each subject "
        "in scope, it loads the audio, vision and EEG logits, computes "
        "softmax probabilities, and produces four results: per-modality "
        "accuracies reconstructed from logits (which must match the "
        "Day-2 CSV as a sanity check); mean-of-softmax fusion; "
        "sum-of-log-softmax fusion (the independent-evidence rule); and "
        "hard majority voting with ties broken by the modality with "
        "highest confidence."
    )
    doc.add_paragraph(
        "Before any fusion result is interpretable, the trial alignment "
        "across modalities must be validated. The Day-3 script includes "
        "an alignment check that verifies the test-set label sequences "
        "match across all three modalities. The check passed; the "
        "stratified-by-class splitting in EAV's repository produced "
        "identical label sequences across modalities for all three pilot "
        "subjects, and this was treated as defensible to proceed with "
        "fusion analysis without further investigation. A subsequent "
        "heuristic check on Day 1 of the extended-deadline schedule "
        "confirmed identical per-class trial counts (80 trials per "
        "class) for all 42 subjects across all three modalities."
    )
    doc.add_paragraph(
        "On the three pilot subjects, naïve late fusion produced a mean "
        "test accuracy of 0.819 (mean-softmax variant). The fusion gain "
        "over the best single modality (vision at 0.756) was 6.3 "
        "percentage points. This established the central empirical "
        "finding that informed the rest of the project: multimodal "
        "fusion provides meaningful improvement over single-modality "
        "classification on EAV. The Day-3 script was later extended "
        "(during Day 1 of the extended-deadline schedule) to also save "
        "per-trial pairwise Kullback–Leibler divergences between the "
        "three modality softmaxes, which feed the downstream coherence "
        "analysis."
    )

    # ---- 2.4 Day 4 ----
    doc.add_heading("2.4 Day 4 — Trimodal Cross-Modal Attention Fusion Module", level=2)
    doc.add_paragraph(
        "Day 4 produced the project's first contribution beyond a "
        "reproduction of standard baselines: the trimodal cross-modal "
        "attention fusion module. The module is implemented in "
        "`fusion/trimodal_attention.py` and consists of a "
        "`TrimodalAttentionFusion` class that takes three pooled "
        "per-modality feature vectors as input and produces a five-class "
        "emotion prediction together with named per-modality and fused "
        "embeddings as auxiliary outputs."
    )
    doc.add_paragraph(
        "The architecture comprises five steps. First, each modality's "
        "pooled feature vector — 768-dimensional for the Audio Spectrogram "
        "Transformer, 768-dimensional for the Vision Transformer, "
        "960-dimensional for EEGNet — is projected to a common embedding "
        "dimension of 256 via a per-modality linear layer. Second, a "
        "learnable modality embedding of shape (3, 256) is added to "
        "distinguish the audio, vision and EEG tokens, analogous to a "
        "positional encoding. Third, the three projected vectors are "
        "stacked as a three-token sequence and passed through a two-layer "
        "transformer encoder with eight attention heads, GELU "
        "activations, pre-layer-normalisation and a feedforward dimension "
        "of 1024. Fourth, the three post-attention tokens are mean-pooled "
        "to a single 256-dimensional fused representation. Fifth, the "
        "fused representation is passed through a small MLP head "
        "(Linear, GELU, Dropout, Linear) to produce the five-class "
        "logits. The total parameter count is 2.28 million."
    )
    doc.add_paragraph(
        "The choice of self-attention over a three-token modality "
        "sequence, rather than the six-way pairwise cross-modal "
        "attention used by Lee et al. (2024), is justified by the "
        "resolution of the available features. The pretrained "
        "per-modality encoders produce a single pooled vector per clip "
        "rather than a sequence of tokens, and at single-token-per-"
        "modality resolution pairwise cross-attention degenerates to "
        "learned projection. Self-attention across the three modality "
        "tokens captures the equivalent cross-modal mixing at lower "
        "parameter count and with cleaner code. The architectural "
        "change is recorded as a deviation from the reference paper, "
        "defensible on the grounds that the original method requires "
        "sequence-level features that our pretrained encoders do not "
        "natively expose."
    )
    doc.add_paragraph(
        "Three contract tests verify the module's behaviour on synthetic "
        "inputs and are bundled in the `__main__` block of the source "
        "file. The first test confirms the full-modality forward pass "
        "produces output of the correct shape and finite values for all "
        "five named keys in the return dictionary. The second test "
        "confirms that calling `forward()` with a zero tensor for the "
        "EEG input produces finite logits with non-trivial magnitude — "
        "this is the architectural commitment that enables the eventual "
        "demo, in which EEG is unavailable at inference time. The third "
        "test verifies that backpropagation reaches every parameter "
        "group, including the per-modality projection layers and the "
        "modality embeddings. All three tests pass."
    )

    # ---- 2.5 Day 5 ----
    doc.add_heading("2.5 Day 5 — Three-Subject Pilot of the Full Pipeline", level=2)
    doc.add_paragraph(
        "Day 5 produced the project's first end-to-end fusion training "
        "result, deliberately limited to three subjects as a pilot. The "
        "pipeline implementation in `day5_pipeline.py` orchestrates "
        "three sequential stages: per-modality re-training of the AST, "
        "ViT and EEGNet models with state-dict persistence; feature "
        "extraction using the pre-classifier representations produced "
        "by wrappers in `fusion/feature_extractors.py`; and fusion "
        "training of `TrimodalAttentionFusion` on the cached features. "
        "All three stages are resume-aware, with per-subject and "
        "per-modality cache checks that allow the pipeline to be "
        "re-executed safely after a kernel restart."
    )
    doc.add_paragraph(
        "The pilot used the EAV repository's default per-modality epoch "
        "budgets (ten frozen plus fifteen fine-tuning epochs for AST, "
        "ten plus five for ViT, three hundred and fifty for EEGNet). "
        "The fusion module was trained for eighty epochs with the AdamW "
        "optimiser at learning rate 0.001, cosine learning-rate "
        "annealing, batch size 32, and best-test-epoch checkpointing of "
        "the logits."
    )
    doc.add_paragraph(
        "The pilot result was a fusion mean of 0.800 across the three "
        "subjects, with high between-subject variance (subject 1 = "
        "0.683, subject 2 = 0.783, subject 3 = 0.933). At face value, "
        "this was 1.9 percentage points below the Day-3 naïve "
        "late-fusion baseline of 0.819, which prompted a hyperparameter-"
        "iteration discussion. In retrospect (see Section 2.8), the gap "
        "was an artefact of the Day-3 baseline being computed on the "
        "same three subjects that happened to include two relatively "
        "easy individuals; the gap vanishes when both methods are "
        "evaluated at the full 42-subject scale."
    )

    # ---- 2.6 Reformulation ----
    doc.add_heading("2.6 Mid-Day-5 Project Reformulation", level=2)
    doc.add_paragraph(
        "Following completion of the three-subject pilot, the project's "
        "scope and framing were re-evaluated. The reformulation, "
        "documented in Section 1.2 of this report and recorded "
        "contemporaneously in the project's CLAUDE.md notes file, "
        "shifted the headline contribution from \"we propose a fusion "
        "architecture that beats baselines\" to \"we characterise "
        "cross-modal affective coherence and find that EAV is a "
        "low-headroom regime for learned fusion methods.\" The "
        "architectural work was retained, with cross-attention fusion "
        "becoming a supporting empirical result rather than the central "
        "claim, and the suppression-matrix coherence analysis becoming "
        "the headline."
    )
    doc.add_paragraph(
        "The reformulation preserved every artefact and code component "
        "produced up to that point. The per-modality classifiers, the "
        "naïve-fusion baseline and the cross-attention fusion module "
        "all feed directly into the coherence-detection framework. What "
        "changed was the prose framing and the relative emphasis of the "
        "research questions in Section 1.3."
    )

    # ---- 2.7 Methodology cleanup ----
    doc.add_heading("2.7 Methodology Cleanup and Pilot Coherence Analysis", level=2)
    doc.add_paragraph(
        "Following a planned break in the project, work resumed with a "
        "deadline extension that provided a fresh nine-day execution "
        "window. The first day of the new schedule was reserved for "
        "methodology cleanup before any further computation was "
        "committed."
    )
    doc.add_paragraph(
        "A trial-alignment heuristic check was conducted to verify that "
        "the per-modality preprocessing did not draw from disjoint trial "
        "pools across audio, vision and EEG. The check loaded each "
        "subject's pickled splits and counted per-class trial "
        "occurrences across all three modalities; for all three pilot "
        "subjects, every modality showed exactly eighty trials per "
        "class (totalling four hundred), satisfying the necessary "
        "condition for trial alignment."
    )
    doc.add_paragraph(
        "The original intent of preserving the per-trial "
        "Listen-versus-Speak task tag through preprocessing was "
        "abandoned. The user no longer had the raw EAV data on disk "
        "(it had been deleted after the initial pickle generation), and "
        "re-downloading the forty-six-gigabyte dataset from Zenodo to "
        "recover the per-trial task labels was judged not worth the "
        "time. The headline coherence analysis was correspondingly "
        "redesigned to use a suppression matrix that operates on "
        "whatever trials are present, without requiring the "
        "Listen/Speak distinction."
    )
    doc.add_paragraph(
        "The suppression matrix is a five-by-five contingency table. "
        "Each row corresponds to an emotion that the audio-vision "
        "consensus predicted (the externally displayed emotion), and "
        "each column corresponds to the emotion that the EEG model "
        "predicted (the internally indicated emotion). A trial "
        "contributes to the matrix if and only if three conditions are "
        "met: audio and vision both produce the same top-1 prediction "
        "(an external consensus exists); the EEG top-1 prediction "
        "differs from that consensus (a divergence is present); and "
        "the EEG model's top-1 confidence is at least 0.5 (the "
        "divergence is not the EEG model essentially guessing). The "
        "diagonal of the matrix is zero by construction."
    )
    doc.add_paragraph(
        "Run on the three pilot subjects, the suppression matrix "
        "produced 32 suppression events from 360 trials, a base rate "
        "of 8.9 percent. The per-subject consistency analysis showed "
        "that only one cell — Calmness displayed externally with EEG "
        "predicting Neutral — was robust across all three subjects, "
        "with the others driven entirely by single subjects. The EEG "
        "marginal comparison showed Neutral inflation of +12.8 "
        "percentage points on suppression-event trials relative to "
        "the overall distribution, indicating an EEG-model bias "
        "confound that would need to be reported as a methodological "
        "caveat — a finding that subsequently disappeared at "
        "forty-two-subject scale."
    )

    # ---- 2.8 42-subject rollout ----
    doc.add_heading("2.8 Forty-Two-Subject Rollout and Headline Results", level=2)
    doc.add_paragraph(
        "The full forty-two-subject training rollout was executed in a "
        "single overnight session on a Google Colab Pro L4 GPU instance, "
        "with all per-modality state dictionaries, extracted features, "
        "fusion logits and CSV results persisted to the user's Google "
        "Drive. The per-modality epoch budgets were reduced from the "
        "EAV defaults — to five-plus-five for AST and three-plus-two "
        "for ViT — on the basis of Day-2 evidence that AST training "
        "accuracy reaches 100 per cent by epoch five of fine-tuning. "
        "The EEGNet budget was retained at 350 epochs. The reduced "
        "budgets yielded approximately ten to fifteen minutes of "
        "training time per subject on the L4 GPU, completing the full "
        "rollout within approximately twelve hours."
    )
    doc.add_paragraph(
        "The cross-attention fusion result across all forty-two "
        "subjects is a mean test accuracy of 0.802 with a standard "
        "deviation of 0.083. Five subjects scored below 0.70 (subjects "
        "8, 12, 14, 16 and 40); two subjects exceeded 0.95 (subjects "
        "20 and 41). Table 3 summarises the headline per-modality and "
        "fusion results."
    )
    doc.add_paragraph()

    rows = [
        ("Per-modality: Audio (AST)",       "57.1%",  "+20.4 pp above SCNN baseline"),
        ("Per-modality: Vision (ViT)",      "74.6%",  "+21.8 pp above DeepFace baseline"),
        ("Per-modality: EEG (EEGNet)",      "43.9%",  "+7.2 pp above EEGNet baseline"),
        ("Best single modality",            "74.6%",  "(vision)"),
        ("Naïve late fusion (mean-softmax)", "80.0%",  "+5.4 pp over best single"),
        ("Cross-attention fusion",          "80.2%",  "+0.2 pp vs naïve late fusion"),
    ]
    table = doc.add_table(rows=len(rows) + 1, cols=3)
    table.rows[0].cells[0].text = "Method"
    table.rows[0].cells[1].text = "Mean (42 subj.)"
    table.rows[0].cells[2].text = "Note"
    for i, (m, acc, note) in enumerate(rows, start=1):
        table.rows[i].cells[0].text = m
        table.rows[i].cells[1].text = acc
        table.rows[i].cells[2].text = note
    _format_table(table)
    _para(doc, "Table 3. Forty-two-subject mean test accuracies across "
                "per-modality and fusion methods. EAV paper baselines: "
                "SCNN audio = 36.7%, DeepFace vision = 52.8%, EEGNet EEG = 36.7%.",
          italic=True, size=10, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()

    doc.add_paragraph(
        "Three observations from this table merit emphasis. First, all "
        "three per-modality classifiers exceed their respective "
        "EAV-paper baselines by substantial margins, confirming that "
        "the per-modality training pipeline is functioning correctly. "
        "Second, multimodal fusion provides a 5.4 percentage-point "
        "improvement over the best single modality, indicating that "
        "the three modalities carry genuinely complementary information "
        "on this dataset. Third, learned cross-modal attention fusion "
        "and naïve softmax-averaging fusion produce statistically "
        "indistinguishable results, with cross-attention exceeding "
        "naïve by 0.2 percentage points. This is reported as a neutral "
        "architectural finding: the choice of fusion mechanism between "
        "these two alternatives is not the performance bottleneck on "
        "EAV."
    )

    # ---- 2.9 Suppression matrix ----
    doc.add_heading("2.9 Suppression Matrix at Forty-Two Subjects", level=2)
    doc.add_paragraph(
        "The suppression matrix at forty-two-subject scale is the "
        "thesis's headline empirical contribution. The full 5040 test "
        "trials produced 2426 trials with audio-vision consensus (48.1 "
        "per cent), 1252 of which had EEG disagreement (24.8 per "
        "cent), and 401 trials passing the strict EEG-confidence ≥ 0.5 "
        "filter (8.0 per cent of all trials). The base rate of "
        "confident cross-modal incoherence is therefore approximately "
        "one in twelve test trials, a stable number across both the "
        "three-subject pilot and the full population."
    )
    doc.add_paragraph(
        "Table 4 reports the 5×5 matrix. Rows are external emotion "
        "(audio-vision consensus); columns are internal emotion (EEG "
        "top-1 prediction); cells are counts of suppression events."
    )
    doc.add_paragraph()

    matrix_rows = [
        ("Neutral",   "0",  "5",  "3",  "8",  "24"),
        ("Sadness",   "43", "0",  "21", "12", "20"),
        ("Anger",     "5",  "13", "0",  "65", "10"),
        ("Happiness", "11", "17", "52", "0",  "14"),
        ("Calmness",  "40", "17", "4",  "17", "0"),
    ]
    matrix_table = doc.add_table(rows=len(matrix_rows) + 1, cols=6)
    matrix_table.rows[0].cells[0].text = "External \\ Internal"
    for j, emo in enumerate(["Neutral", "Sadness", "Anger", "Happiness", "Calmness"]):
        matrix_table.rows[0].cells[j + 1].text = emo
    for i, row in enumerate(matrix_rows, start=1):
        for j, val in enumerate(row):
            matrix_table.rows[i].cells[j].text = val
    _format_table(matrix_table)
    _para(doc, "Table 4. Suppression matrix across 42 subjects, 401 "
                "events. Diagonal is zero by construction.",
          italic=True, size=10, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()

    doc.add_paragraph(
        "Three patterns are visible. The most prominent is a "
        "bidirectional Anger-Happiness confusion: 65 events of Anger "
        "displayed with EEG predicting Happiness, and 52 events of "
        "Happiness displayed with EEG predicting Anger, combining to "
        "117 of the 401 events. The pattern is consistent with "
        "high-arousal valence confusion in the EEG signal: Anger and "
        "Happiness share many physiological correlates of elevated "
        "arousal. Second, directional Sadness-to-Neutral (43) and "
        "Calmness-to-Neutral (40) patterns appear, in which low-arousal "
        "emotions are displayed externally while EEG indicates a more "
        "neutral physiological baseline. Third, smaller patterns at "
        "Calmness-to-Sadness, Calmness-to-Happiness, and "
        "Happiness-to-Sadness each contribute in the range of 14 to "
        "20 events."
    )
    doc.add_paragraph(
        "Per-subject decomposition confirms that these patterns are not "
        "artefacts of a single outlier subject. The Anger-to-Happiness "
        "cell contains contributions from over twenty subjects. The "
        "Happiness-to-Anger cell shows similar distribution. The "
        "Sadness-to-Neutral cell has a single subject (subject 6) "
        "contributing fourteen events, but the remaining twenty-nine "
        "events come from fifteen other subjects, establishing the "
        "pattern as cross-population rather than subject-specific."
    )
    doc.add_paragraph(
        "The EEG-Neutral inflation confound identified at the "
        "three-subject pilot has disappeared at scale. The Neutral "
        "fraction of EEG predictions on suppression-event trials is "
        "24.7 per cent, compared to 25.7 per cent on all trials — an "
        "inflation of negative 1.0 percentage points. The Neutral "
        "column of the matrix at forty-two-subject scale reflects "
        "genuine EEG signal rather than model uncertainty, removing "
        "the principal methodological caveat that had been planned for "
        "the Limitations section."
    )


# ---------------------------------------------------------------------------
# Section 3 — Challenges and Solutions
# ---------------------------------------------------------------------------

CHALLENGES = [
    (
        "Dependency conflict caused by facenet-pytorch (Day 1)",
        "The initial dependency-install command included `facenet-pytorch`, "
        "which pins NumPy below version 2 and an older release of "
        "PyTorch. Installing it on Colab silently downgraded NumPy from "
        "2.x to 1.26.4 and PyTorch from 2.10 to 2.2.2, breaking a large "
        "set of pre-installed Colab packages including torchaudio. The "
        "audio-spectrogram-transformer pipeline silently failed under "
        "the broken combination.",
        "facenet-pytorch was removed from the Day-1 install set on the "
        "grounds that the smoke test operates on pre-processed pickle "
        "files and does not require MTCNN-based face detection. Should "
        "it be required on a later day (it was not), the recommended "
        "installation incantation is `pip install --no-deps facenet-"
        "pytorch`, which avoids the destructive downgrade."
    ),
    (
        "Case-sensitivity bug in folder-name handling (Day 1)",
        "An early helper used Python's `str.capitalize` method to "
        "translate the modality name 'eeg' into the folder name 'Eeg', "
        "whereas the actual folder on Drive is 'EEG' in upper case. "
        "The defect was masked for audio and vision, where `capitalize` "
        "happens to produce the correct folder name. EEG files could "
        "not be located at runtime, producing a misleading "
        "FileNotFoundError pointing at the wrong-case directory.",
        "Replaced the implicit string transform with an explicit "
        "dictionary mapping `{audio: 'Audio', vision: 'Vision', eeg: "
        "'EEG'}`, applied uniformly across both the verification helper "
        "and the smoke-test driver. The general lesson was to never "
        "derive a filesystem path component from a Python string "
        "transform; always use an explicit mapping."
    ),
    (
        "Forward-hook return-value bug in EEGNet_tor.py (Day 1)",
        "EEGNet training failed with the error message `RuntimeError: "
        "running_mean should contain 1 elements not 64`. The depthwise "
        "convolution and final dense layer in EEGNet carry max-norm "
        "constraints implemented as forward hooks of the form `lambda "
        "module, inputs, outputs: module.weight.data.renorm_(...)`. The "
        "`renorm_` operation is in-place but returns the modified "
        "weight tensor; the lambda implicitly propagates that tensor "
        "as its return value; PyTorch's `register_forward_hook` "
        "protocol then substitutes the returned tensor in place of the "
        "layer's output. The downstream BatchNorm received the weight "
        "tensor (shape (64, 1, 30, 1)) instead of the convolution "
        "output and crashed.",
        "The hooks were extracted into a real (non-lambda) function "
        "`_make_max_norm_hook(norm_rate)` whose implicit return is "
        "`None`, satisfying the forward-hook protocol's no-replacement "
        "convention. The fix was applied to both the depthwise "
        "convolution and the dense layer."
    ),
    (
        "Classifier-head-size mismatch in Transformer_Vision.py (Day 1)",
        "The vision trainer attempted to repurpose a seven-class "
        "facial-emotions ViT for the EAV five-class task by replacing "
        "`self.model.classifier` with a fresh `Linear(hidden, 5)` and "
        "setting `self.model.num_labels = 5`. This updated the "
        "attribute but not `self.model.config.num_labels`, which the "
        "Hugging Face Transformers library consults when computing the "
        "internal cross-entropy loss. The model therefore emitted "
        "(B, 5) logits but the loss function attempted to reshape them "
        "as (B, 7), failing with `RuntimeError: shape '[-1, 7]' is "
        "invalid for input of size 160`.",
        "Replaced the manual head replacement with the documented "
        "`AutoModelForImageClassification.from_pretrained(model_path, "
        "num_labels=5, ignore_mismatched_sizes=True)` idiom, which both "
        "re-initialises the head and updates the configuration."
    ),
    (
        "RAM exhaustion during eager image preprocessing (Day 1)",
        "The vision trainer's `preprocess_images` method iterated over "
        "each of approximately ten thousand frames per subject, ran the "
        "Hugging Face image processor on each frame individually, "
        "accumulated the resulting tensors in a Python list, and "
        "finally invoked `torch.stack(list).to(self.device)`. On Colab "
        "Free this allocated in excess of eight gigabytes of CPU memory "
        "and silently killed the kernel with no visible error message.",
        "Rewrote the method to process images in batches of sixty-four, "
        "to keep the resulting tensor on the CPU (the DataLoader moves "
        "it to GPU per batch at train time), and to emit periodic "
        "progress messages so that slow progress is distinguishable "
        "from a hang."
    ),
    (
        "Softmax-CrossEntropy double-softmax composition bug (Day 2)",
        "The EEGNet model's `forward()` method ended with a softmax "
        "layer, while the trainer applied `nn.CrossEntropyLoss` on the "
        "result. `CrossEntropyLoss` is implemented as a numerically-"
        "stable fused log-softmax-plus-negative-log-likelihood "
        "operation that expects raw logits. Feeding it softmax "
        "probabilities causes the gradient to vanish as soon as any "
        "class becomes mildly confident, with the model essentially "
        "failing to train. The Day-1 smoke test produced EEG accuracy "
        "at chance level (24.4 per cent across three subjects).",
        "The trailing softmax was removed from `EEGNet_tor.forward()` "
        "and the `nn.Softmax` instance was removed from `__init__`. "
        "The fix produced a one-to-two-percentage-point improvement on "
        "a controlled fifty-epoch sanity test, which was smaller than "
        "expected and prompted further investigation that uncovered "
        "the next bug."
    ),
    (
        "Trainer_uni train/eval mode bug (Day 2)",
        "Inspection of `Trainer_uni.train()` in the EAV repository "
        "revealed that `self.model.train()` was called exactly once, "
        "before the outer epoch loop. Inside the loop, "
        "`self.validate()` is called at the end of each epoch, which "
        "sets `self.model.eval()`. From epoch two onwards, therefore, "
        "training proceeded with BatchNorm layers in evaluation mode "
        "(using frozen running statistics) and Dropout disabled. Both "
        "effects impair training, and in combination they explain a "
        "model that learns slowly and never reaches its capacity.",
        "The call to `self.model.train()` was relocated to the top of "
        "the epoch loop. A controlled fifty-epoch sanity test produced "
        "a six-percentage-point improvement in EEG accuracy (0.317 → "
        "0.375), and the full three-hundred-and-fifty-epoch training "
        "schedule across three subjects produced a mean EEG accuracy "
        "of 0.506, an improvement of over twenty-six percentage "
        "points above the Day-1 smoke test."
    ),
    (
        "Cross-attention fusion under-performed naïve fusion on the three-subject pilot (Day 5)",
        "The Day-5 three-subject cross-attention fusion mean of 0.800 "
        "was 1.9 percentage points below the Day-3 naïve late-fusion "
        "baseline of 0.819 on the same three subjects. The deficit "
        "prompted a hyperparameter-iteration discussion in which three "
        "configurations were proposed for testing: stronger "
        "regularisation, smaller model size, and a concat-then-MLP "
        "sanity-check baseline.",
        "Before the iteration was executed, the project moved to the "
        "forty-two-subject rollout. The full-scale result (cross-"
        "attention mean 0.802, naïve late-fusion mean 0.800) showed "
        "that the deficit was an artefact of the three-subject "
        "comparison's small sample size, in which two of the three "
        "pilot subjects happened to be relatively easy. At forty-two "
        "subjects, the two methods are statistically tied. No further "
        "hyperparameter iteration was necessary."
    ),
    (
        "Project pivot away from in-the-wild continuous emotion tracking (Day 5)",
        "Mid-Day-5, it became apparent that the originally-envisioned "
        "ultimate application — continuous per-second emotion tracking "
        "on in-the-wild audiovisual content — was incompatible with "
        "the EEG modality. Any EAV-trained system applied to movie "
        "content would be making predictions in a regime fundamentally "
        "different from the training distribution. The thesis as "
        "initially framed was a methodology study without a clear "
        "application story, and the application story could not be "
        "honestly delivered in the available time.",
        "The thesis was reformulated to a different research question: "
        "cross-modal affective coherence detection. The new framing "
        "treats cross-modal divergence as a signal in its own right "
        "rather than as something to overcome via fusion. All existing "
        "code components were retained; what changed was the headline "
        "contribution and the emphasis of the research questions."
    ),
    (
        "Raw EAV data not available for re-preprocessing (Restart)",
        "Following a planned break in the project, work resumed with "
        "the intent of modifying the EAV preprocessing pipeline to "
        "preserve per-trial task tags (Listen versus Speak) for use in "
        "the coherence analysis. The raw EAV data — approximately "
        "forty-six gigabytes downloaded from Zenodo — had been deleted "
        "from the user's local disk after the initial pickle "
        "generation, and re-downloading was judged not worth the time "
        "cost given the approaching deadline.",
        "The Listen-versus-Speak coherence analysis was dropped as a "
        "headline contribution. The suppression-matrix analysis, which "
        "does not require task tags, was promoted to the headline. The "
        "decision was recorded explicitly in the project's master plan "
        "document, with a note that future work on a clinical or "
        "task-aware corpus would naturally re-introduce the "
        "Listen-versus-Speak comparison."
    ),
    (
        "Environment-variable propagation between PyCharm Run Configurations and the Python Console (Forty-two-subject rollout setup)",
        "When the workflow moved to a remote 4080 PC (accessed via "
        "AnyDesk) and PyCharm, environment variables set in a Run "
        "Configuration were found not to propagate to the Python "
        "Console or Terminal tabs within PyCharm. Tests of the "
        "path-resolution module produced incorrect Mac-default paths "
        "even with the run configuration apparently set correctly. The "
        "`paths.py` module additionally suffered from Python's "
        "import-cache behaviour: once imported, the module's resolved "
        "paths were frozen for the session even if the environment "
        "variables were subsequently changed.",
        "Two complementary fixes were applied. First, the `paths.py` "
        "module was hardened to auto-detect the Google Colab runtime "
        "environment and default to Drive-mounted paths when running "
        "in Colab, removing the requirement to set environment "
        "variables before every session. Second, a documented "
        "\"rebind\" cell was added to the workflow for cases where the "
        "module had already been imported under stale environment "
        "variables; it evicts the `paths` module from `sys.modules` so "
        "that the next import re-reads from the environment."
    ),
    (
        "Per-modality test logits not saved during the 42-subject Day-5 rollout (Post-rollout)",
        "The Day-5 pipeline persisted three artefacts per subject: "
        "per-modality model state dictionaries, pre-classifier feature "
        "vectors, and fused logits from the cross-attention model. It "
        "did not save per-modality (post-classifier-head) logits. The "
        "naïve-late-fusion baseline and the suppression matrix both "
        "require per-modality softmax outputs; without these, neither "
        "comparison could be computed at forty-two-subject scale.",
        "A small post-hoc inference script, "
        "`compute_per_modality_logits.py`, was written. For each of "
        "the forty-two subjects, it loads the trained state "
        "dictionaries, instantiates the corresponding model, runs "
        "inference on the test set, and saves per-modality logits to a "
        "new directory `day5_per_modality_logits/`. The operation "
        "completed in approximately twenty minutes on Colab L4. The "
        "two consumer scripts (`day3_late_fusion.py` and "
        "`suppression_matrix.py`) were patched to read from the new "
        "directory via a configurable environment variable."
    ),
    (
        "EEG-Neutral inflation confound at three-subject pilot scale (Pilot coherence analysis)",
        "At three-subject scale, the suppression matrix's Neutral "
        "column showed a count of 13 events, the highest of any "
        "column. The EEG marginal-distribution comparison developed in "
        "the follow-up analysis revealed that Neutral was "
        "disproportionately predicted by the EEG model on "
        "suppression-event trials versus the overall distribution — "
        "an inflation of 12.8 percentage points. This raised a "
        "methodological concern that the EEG model was defaulting to "
        "Neutral on uncertain trials, inflating the matrix's Neutral "
        "column artefactually.",
        "At forty-two-subject scale, the same comparison was repeated. "
        "The Neutral inflation on suppression-event trials is now "
        "negative 1.0 percentage points, indicating no meaningful bias. "
        "The artefact identified in the pilot was a small-sample "
        "phenomenon that disappeared at full scale, removing the "
        "principal methodological caveat that had been planned for "
        "the Limitations section."
    ),
]


def build_section_3(doc: Document) -> None:
    doc.add_heading("3. Challenges Faced and Solutions Implemented", level=1)
    doc.add_paragraph(
        "Thirteen distinct technical and methodological challenges were "
        "encountered between Day 1 and the present. Each is documented "
        "below in the chronological order in which it was diagnosed. "
        "Each entry comprises a brief statement of the symptom, a "
        "description of the underlying cause uncovered during diagnosis, "
        "and the remediation applied. All resolved issues are also "
        "recorded in the project's CLAUDE.md notes file for "
        "cross-session persistence."
    )
    for i, (title, diagnosis, resolution) in enumerate(CHALLENGES, start=1):
        doc.add_heading(f"Challenge {i}: {title}", level=2)
        _para(doc, "Diagnosis.", bold=True)
        doc.add_paragraph(diagnosis)
        _para(doc, "Resolution.", bold=True)
        doc.add_paragraph(resolution)


# ---------------------------------------------------------------------------
# Section 4 — Remaining Work and Future Roadmap
# ---------------------------------------------------------------------------

def build_section_4(doc: Document) -> None:
    doc.add_heading("4. Remaining Work and Future Roadmap", level=1)

    doc.add_heading("4.1 Immediate Remaining Workstreams", level=2)
    doc.add_paragraph(
        "Four workstreams remain between the current state of the "
        "project and the deadline. Each is described below with its "
        "expected scope, output, and approximate compute budget."
    )

    doc.add_heading("4.1.1 Modality-Dropout Training", level=3)
    doc.add_paragraph(
        "The fusion module is currently trained on full-modality "
        "examples and therefore does not yet handle missing modalities "
        "with sensible behaviour at inference time. The softhard "
        "modality-dropout scheme of Chumachenko, Iosifidis and Gabbouj "
        "(ICPR, 2022) will be adapted to the trimodal case: each "
        "training batch will be expanded into four sub-batches "
        "(full-modality; audio-only with vision and EEG zeroed; "
        "vision-only with audio and EEG zeroed; audio-plus-vision with "
        "EEG zeroed) with the appropriate labels replicated. The "
        "fusion module is then retrained on cached features, requiring "
        "no additional per-modality training. Compute estimate: "
        "approximately one hour on Colab L4."
    )
    doc.add_paragraph(
        "The resulting model is evaluated under multiple inference "
        "conditions, with the headline robustness number being the "
        "zero-EEG accuracy across all forty-two subjects. This number "
        "is what the demonstration will run on (since EEG is "
        "unavailable in any in-the-wild deployment scenario) and is "
        "also the principal robustness claim of the thesis."
    )

    doc.add_heading("4.1.2 Subject-Independent Evaluation", level=3)
    doc.add_paragraph(
        "All accuracies reported in this document are within-subject: "
        "the training and test sets come from the same individual. "
        "Subject-independent evaluation is the standard credibility "
        "check in affective computing, and is essential for any "
        "generalisation claim. Two options are available: full "
        "leave-one-subject-out (LOSO), which trains forty-two separate "
        "models, and five-fold cross-subject, which trains five. The "
        "former is compute-prohibitive on the remaining schedule; the "
        "latter is feasible. The five-fold variant is therefore the "
        "planned approach. Compute estimate: approximately five hours "
        "on Colab L4 for the fusion-module retraining only, using "
        "cached features."
    )

    doc.add_heading("4.1.3 Demonstration", level=3)
    doc.add_paragraph(
        "A file-based demonstration is the planned defence-day "
        "artefact. The input is a personal recorded video clip of "
        "twenty to thirty seconds with synchronous audio. The output "
        "is a copy of the video with two overlays rendered on top: a "
        "per-segment predicted emotion label with confidence, and a "
        "per-segment coherence indicator that flashes red when the "
        "system's audio-vision and EEG-equivalent (here zeroed) "
        "predictions diverge. The implementation uses OpenCV for "
        "frame-level overlay and FFmpeg for re-attaching the original "
        "audio track. Two small scripts (`demo_inference.py` and "
        "`demo_overlay.py`) will be written for this purpose. "
        "Estimated implementation time: approximately one full day."
    )

    doc.add_heading("4.1.4 Thesis Writing", level=3)
    doc.add_paragraph(
        "The thesis manuscript and the defence slide deck are the "
        "final deliverables. The manuscript follows the standard "
        "format — Introduction, Related Work, Methods, Experiments, "
        "Results, Discussion, Limitations, Future Work, Conclusion — "
        "with the Limitations section explicitly acknowledging that "
        "the coherence methodology is validated behaviourally but not "
        "clinically, and that in-the-wild evaluation is Phase 2 "
        "future work. The slide deck structures the defence around "
        "the five research questions stated in Section 1.3. Estimated "
        "writing time: approximately two to three full days."
    )

    doc.add_heading("4.2 Schedule for the Remaining Window", level=2)
    doc.add_paragraph(
        "The remaining workstreams are sequenced across the days "
        "between today and the deadline of June 9, 2026. Table 5 "
        "summarises the plan."
    )
    doc.add_paragraph()
    schedule_rows = [
        ("Jun 2", "Modality-dropout training; zero-EEG robustness evaluation"),
        ("Jun 3", "Five-fold subject-independent evaluation"),
        ("Jun 4", "Demonstration construction; personal video recording"),
        ("Jun 5", "Thesis writing — Methods, Experiments and Results chapters"),
        ("Jun 6", "Thesis writing — Discussion, Limitations and Conclusion"),
        ("Jun 7", "Defence slide deck"),
        ("Jun 8", "Final manuscript pass; submission preparation"),
        ("Jun 9", "Final submission"),
    ]
    schedule = doc.add_table(rows=len(schedule_rows) + 1, cols=2)
    schedule.rows[0].cells[0].text = "Date"
    schedule.rows[0].cells[1].text = "Principal Goal"
    for i, (d, goal) in enumerate(schedule_rows, start=1):
        schedule.rows[i].cells[0].text = d
        schedule.rows[i].cells[1].text = goal
    _format_table(schedule)
    _para(doc, "Table 5. Schedule for the remaining work window, "
                "June 2 through June 9, 2026.",
          italic=True, size=10, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()

    doc.add_heading("4.3 Phase 2 Future Work (Out of Scope)", level=2)
    doc.add_paragraph(
        "The thesis explicitly carves out three workstreams as Phase 2 "
        "future work, beyond the scope of the present submission. "
        "Acknowledging these in the manuscript permits the thesis to "
        "make a tightly-scoped claim about EAV without inviting "
        "reviewer questions about generalisation beyond it."
    )
    doc.add_paragraph(
        "First, clinical validation. The suppression-matrix methodology "
        "developed in this thesis identifies patterns of cross-modal "
        "incoherence in a healthy laboratory population. The clinical "
        "interest of these patterns — whether subjects with depression, "
        "alexithymia, post-traumatic stress disorder, or autism "
        "spectrum conditions show systematically different coherence "
        "patterns — requires a labelled clinical population that the "
        "present thesis does not have. Phase 2 work would deploy the "
        "same methodology against such a population to test whether "
        "coherence scores discriminate diagnostic groups."
    )
    doc.add_paragraph(
        "Second, in-the-wild deployment. The model architecture has "
        "been designed with explicit hooks (per-modality embeddings as "
        "named outputs of the fusion forward pass; zero-EEG inference "
        "path) that support a future temporal head consuming "
        "continuous video. Realising this requires a "
        "temporally-annotated in-the-wild audiovisual dataset (DFEW, "
        "MAFW, or a custom collection) and the addition of a "
        "sequence-to-sequence temporal model (LSTM, Residual-TCN, or "
        "temporal transformer) on top of the existing frozen "
        "per-modality encoders. The architectural decisions made "
        "during the present thesis make this extension a localised "
        "change rather than a rewrite."
    )
    doc.add_paragraph(
        "Third, MERCL contrastive pre-training. The supervised "
        "contrastive losses of Lee, Kim and Kim (2024) — Intra-Modal "
        "Contrastive Learning (AMCL), Inter-Modal Contrastive Learning "
        "(EMCL) and Sample-wise Multimodal Alignment Contrastive "
        "Learning (SMCL) — are theoretically attractive but were "
        "dropped from the present thesis under schedule pressure. "
        "Their re-introduction in Phase 2 would test whether "
        "contrastive pre-training of the per-modality encoders, "
        "before fine-tuning the fusion head, materially improves "
        "either classification or coherence-detection performance."
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

    doc.save(OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
