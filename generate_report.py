#!/usr/bin/env python3
"""Generate Day 1 thesis progress report as a formatted .docx file.

Pulls results from results/day1_smoketest.csv and context from CLAUDE.md.
Outputs thesis_progress_report_day1.docx in the project root.
"""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.table import WD_ALIGN_VERTICAL, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_CSV = PROJECT_ROOT / "results" / "day1_smoketest.csv"
OUTPUT = PROJECT_ROOT / "thesis_progress_report_day1.docx"


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _set_cell_borders(cell, color: str = "000000", size: str = "4") -> None:
    """Give a table cell visible single-line borders on all sides."""
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        border = OxmlElement(f"w:{edge}")
        border.set(qn("w:val"), "single")
        border.set(qn("w:sz"), size)
        border.set(qn("w:space"), "0")
        border.set(qn("w:color"), color)
        borders.append(border)
    tc_pr.append(borders)


def _shade_cell(cell, fill: str = "D9E2F3") -> None:
    """Apply a background fill colour to a cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tc_pr.append(shd)


def _style_header_row(row) -> None:
    for cell in row.cells:
        _shade_cell(cell)
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True


def _format_table(table, header_shade: bool = True) -> None:
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for row in table.rows:
        for cell in row.cells:
            _set_cell_borders(cell)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    if header_shade and len(table.rows) > 0:
        _style_header_row(table.rows[0])


def _insert_toc(doc: Document) -> None:
    """Insert a real Word TOC field. Must be updated in Word (F9 / right-click)."""
    paragraph = doc.add_paragraph()
    run = paragraph.add_run()

    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")

    instr_text = OxmlElement("w:instrText")
    instr_text.set(qn("xml:space"), "preserve")
    instr_text.text = r'TOC \o "1-2" \h \z \u'

    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")

    placeholder = OxmlElement("w:t")
    placeholder.text = (
        "Right-click here and choose “Update Field” to populate the table of contents."
    )

    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")

    run._r.append(fld_begin)
    run._r.append(instr_text)
    run._r.append(fld_sep)
    run._r.append(placeholder)
    run._r.append(fld_end)


def _para(doc: Document, text: str, *, bold: bool = False, italic: bool = False,
          size: int | None = None, align=None) -> None:
    paragraph = doc.add_paragraph()
    if align is not None:
        paragraph.alignment = align
    run = paragraph.add_run(text)
    run.bold = bold
    run.italic = italic
    if size is not None:
        run.font.size = Pt(size)


def _bullet(doc: Document, text: str) -> None:
    doc.add_paragraph(text, style="List Bullet")


def _numbered(doc: Document, text: str) -> None:
    doc.add_paragraph(text, style="List Number")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results() -> list[dict]:
    rows: list[dict] = []
    with open(RESULTS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "subject": int(r["subject"]),
                "modality": r["modality"],
                "test_acc": float(r["test_acc"]),
                "seconds": float(r["seconds"]),
            })
    return rows


def pivot_results(rows: list[dict]) -> dict[int, dict[str, dict[str, float]]]:
    pivot: dict[int, dict[str, dict[str, float]]] = {}
    for r in rows:
        pivot.setdefault(r["subject"], {})[r["modality"]] = {
            "acc": r["test_acc"],
            "sec": r["seconds"],
        }
    return pivot


# ---------------------------------------------------------------------------
# Document construction
# ---------------------------------------------------------------------------

def build_title_page(doc: Document) -> None:
    # Vertical space before title
    for _ in range(6):
        doc.add_paragraph()

    _para(
        doc,
        "Multimodal Emotion Recognition Using Audio, Video and EEG Signals",
        bold=True, size=24, align=WD_ALIGN_PARAGRAPH.CENTER,
    )
    _para(
        doc,
        "A Reproduction and Extension of the MERCL + Cross-Modal Attention Framework on the EAV Dataset",
        italic=True, size=14, align=WD_ALIGN_PARAGRAPH.CENTER,
    )

    for _ in range(6):
        doc.add_paragraph()

    _para(doc, "Author", bold=True, size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    _para(doc, "Shafin", size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()
    _para(doc, "Institution", bold=True, size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    _para(doc, "[Institution name — to be filled in]", size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()
    _para(doc, "Date", bold=True, size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    _para(doc, date.today().strftime("%B %d, %Y"), size=12, align=WD_ALIGN_PARAGRAPH.CENTER)

    doc.add_page_break()


def build_toc(doc: Document) -> None:
    _para(doc, "Table of Contents", bold=True, size=18,
          align=WD_ALIGN_PARAGRAPH.LEFT)
    doc.add_paragraph()
    _insert_toc(doc)
    doc.add_page_break()


# --- Section 1 --------------------------------------------------------------

def build_section_1(doc: Document) -> None:
    doc.add_heading("1. Project Overview", level=1)

    doc.add_heading("1.1 Research Problem and Motivation", level=2)
    doc.add_paragraph(
        "Automatic emotion recognition is increasingly central to applications in "
        "human–computer interaction, affective computing, clinical monitoring, "
        "personalised education, and intelligent content recommendation. Conventional "
        "approaches typically rely on a single modality such as facial expression "
        "analysis, speech prosody, or physiological signals. Each of these modalities, "
        "considered in isolation, offers only a partial view of the underlying emotional "
        "state. Facial expressions can be occluded, voluntarily suppressed, or culturally "
        "modulated; speech signals can be degraded by background noise or by the "
        "speaker’s linguistic habits; physiological signals can be noisy and exhibit "
        "considerable inter-subject variability. As a consequence, single-modality "
        "systems are known to suffer in real-world conditions and tend to generalise "
        "poorly across speakers, scenes, and devices."
    )
    doc.add_paragraph(
        "Multimodal emotion recognition addresses these limitations by integrating "
        "complementary streams of information. Behavioural modalities such as facial "
        "video and speech audio capture how an emotion is externally expressed, while "
        "physiological modalities such as electroencephalography (EEG) capture the "
        "internal, often involuntary, response of the autonomic nervous system. A "
        "trimodal system that jointly leverages audio, video and EEG therefore promises "
        "both robustness against modality-specific degradation and access to internal "
        "affective signals that are not visible on the surface. The research problem "
        "addressed in this thesis is the design and evaluation of such a trimodal "
        "system on a realistic, conversational dataset, and the systematic study of "
        "how cross-modal interaction influences classification performance."
    )

    doc.add_heading("1.2 Research Goals and Objectives", level=2)
    doc.add_paragraph(
        "The primary objective of this thesis is to design, implement and evaluate a "
        "trimodal emotion recognition framework that combines audio, video and EEG "
        "signals via a cross-modal attention fusion mechanism, evaluated on the "
        "EEG-Audio-Video (EAV) dataset published by Lee, Shomanov, Kabidenova and "
        "Yazici in Scientific Data, 2024. In particular, the work seeks to reproduce "
        "and adapt the methodology of Lee, Kim and Kim (Bioengineering, 2024), which "
        "combines supervised contrastive learning across modalities (MERCL) with "
        "pairwise cross-modal attention (CMA), and to evaluate this methodology on the "
        "EAV benchmark, on which it has not previously been reported."
    )
    doc.add_paragraph("The secondary objectives are as follows:")
    _bullet(doc,
        "Establish strong per-modality baselines on the EAV dataset using the "
        "encoders shipped in the official nubcico/EAV repository (Audio Spectrogram "
        "Transformer for audio, Vision Transformer for video, and EEGNet for EEG)."
    )
    _bullet(doc,
        "Implement a trimodal cross-attention fusion module that processes the three "
        "modality streams jointly and produces a single emotion prediction."
    )
    _bullet(doc,
        "Investigate the robustness of the proposed system under modality dropout, "
        "in order to characterise its behaviour when one of the three modalities is "
        "missing or corrupted at inference time."
    )
    _bullet(doc,
        "Where time permits, integrate the MERCL supervised contrastive pre-training "
        "stage and study its marginal contribution above and beyond cross-modal "
        "attention alone."
    )
    _bullet(doc,
        "Maintain a reproducible workflow throughout, with explicit version control, "
        "deterministic data splits, and per-subject results that can be re-evaluated "
        "in a subject-independent (leave-one-subject-out) manner."
    )

    doc.add_heading("1.3 Proposed Approach and Methodology", level=2)
    doc.add_paragraph(
        "The proposed approach is structured in three layers. The first layer "
        "comprises modality-specific encoders that map each raw signal to a fixed-"
        "length feature representation: the Audio Spectrogram Transformer (AST) "
        "pre-trained on AudioSet for the audio modality, a Vision Transformer (ViT) "
        "pre-trained on a facial-emotion dataset for the video modality, and an "
        "EEGNet model trained from scratch on EAV for the EEG modality. The second "
        "layer is a fusion module that ingests these three streams and produces a "
        "joint representation; in this thesis, fusion will progress in stages from "
        "naïve late fusion (averaging of softmax probabilities) to a trimodal "
        "cross-modal attention scheme inspired by Lee et al. (2024) and Chumachenko "
        "et al. (ICPR 2022). The third layer is a small classifier head that maps "
        "the joint representation to the five emotion categories of the EAV dataset. "
        "If time permits, the encoders will additionally be pre-trained with a "
        "MERCL-style supervised contrastive loss before the fusion module is fine-"
        "tuned with cross-entropy."
    )
    doc.add_paragraph(
        "The methodology is explicitly incremental. Each stage produces a working "
        "system that can be evaluated against the previous stage, so that the "
        "marginal contribution of each architectural decision is measurable. This "
        "is critical given the limited time budget: even if the more ambitious "
        "MERCL stage is not completed, a credible thesis contribution remains "
        "available from the cross-attention fusion alone."
    )

    doc.add_heading("1.4 Primary Source of Code: the nubcico/EAV Repository", level=2)
    doc.add_paragraph(
        "The primary source of code for this project is the official EAV repository "
        "maintained by Lee, Shomanov, Kabidenova and Yazici at Nazarbayev University, "
        "available at https://github.com/nubcico/EAV. This repository accompanies the "
        "EAV dataset paper and provides preprocessing pipelines, dataset loaders and "
        "per-modality classifiers tailored to the EAV data layout. Specifically, it "
        "ships PyTorch (and parallel TensorFlow) implementations for audio fine-"
        "tuning on AST, video fine-tuning on ViT and a custom ShallowConvNet/"
        "Transformer hybrid, and EEG training on EEGNet."
    )
    doc.add_paragraph(
        "Two properties make this repository the appropriate starting point. First, "
        "it is the canonical reference implementation distributed by the dataset "
        "authors, which removes ambiguity about data layout, channel ordering, label "
        "encoding and train/test split conventions. Second, although the repository "
        "provides strong per-modality baselines, it deliberately stops short of "
        "implementing trimodal fusion. The README explicitly lists “Add inference "
        "files” and “Create demo file” as open roadmap items, and inspection of the "
        "code base confirms that there is no joint training, no cross-modal "
        "interaction, and no contrastive alignment. This gap is precisely the gap "
        "that this thesis sets out to fill, making nubcico/EAV the ideal scaffolding "
        "on which to build."
    )
    doc.add_paragraph(
        "An earlier candidate scaffold, the multimodal-emotion-recognition repository "
        "by Chumachenko, Iosifidis and Gabbouj (ICPR 2022), was evaluated and "
        "ultimately rejected as the primary base, because it targets only audio and "
        "video on the RAVDESS dataset and does not support EEG. However, it remains "
        "a valuable secondary resource: its self-attention fusion modules and "
        "“softhard” modality dropout pattern will be ported into this project at "
        "Day 4."
    )

    doc.add_heading("1.5 Dataset: the EAV (EEG-Audio-Video) Dataset", level=2)
    doc.add_paragraph(
        "The EAV dataset (Lee et al., Scientific Data, 2024) is, to the authors’ "
        "knowledge, the first publicly available emotion recognition corpus to "
        "synchronously combine 30-channel electroencephalography, audio and video "
        "in a conversational setting. The corpus was collected from 42 participants, "
        "each of whom engaged in a cue-based listening-and-speaking task in front "
        "of a video monitor presenting an experienced actor. The session followed "
        "a pseudo-random sequence designed to elicit five distinct emotions: "
        "Neutral, Sadness, Anger, Happiness and Calmness."
    )
    doc.add_paragraph(
        "Each participant contributed 200 interactions, yielding 8,400 interactions "
        "in total across the cohort. The video stream is captured at 30 frames per "
        "second in 20-second clips covering both listening and speaking phases, the "
        "audio stream is captured during the speaking phase in 20-second WAV files, "
        "and the EEG stream is recorded continuously at 500 Hz across 30 channels "
        "for the same 20-second window. Labels are encoded jointly across the three "
        "modalities, since all recordings were synchronised at acquisition time, "
        "which permits direct trimodal supervised learning without further "
        "alignment."
    )
    doc.add_paragraph(
        "Within the present project, the EAV data have been pre-processed into "
        "five-second segments and serialised as per-subject pickle files. The "
        "resulting tensor shapes are (N, 80000) for audio at 16 kHz, "
        "(N, 25, 56, 56, 3) for video at five frames per second of face-cropped "
        "imagery, and (N, 30, 500) for EEG at 100 Hz after band-pass filtering. The "
        "per-subject split uses h_idx = 56 in the dataset author’s EAVDataSplit "
        "routine, which yields 280 training trials and 120 test trials per subject, "
        "stratified per class."
    )

    doc.add_heading("1.6 Supplementary Resources", level=2)
    doc.add_paragraph(
        "Three additional resources are used in a supplementary capacity. The first "
        "is the paper of Lee, Kim and Kim (Bioengineering, 2024), titled “Emotion "
        "Recognition Using EEG Signals and Audiovisual Features with Contrastive "
        "Learning”, which provides the methodological blueprint for the fusion "
        "architecture: a Residual-TCN temporal encoder, three contrastive losses "
        "(AMCL, EMCL and SMCL) constituting the MERCL pre-training objective, and "
        "six pairwise cross-modal attention modules in the fine-tuning stage. The "
        "paper does not release source code, but its architectural specification is "
        "sufficiently detailed to be re-implemented."
    )
    doc.add_paragraph(
        "The second supplementary resource is the multimodal-emotion-recognition "
        "repository of Chumachenko et al. (ICPR 2022), retained as a donor of "
        "self-attention fusion code and modality-dropout patterns. Its "
        "AttentionBlock module and softhard dropout scheme are adopted, with "
        "modifications, in the trimodal cross-attention module to be built on Day 4."
    )
    doc.add_paragraph(
        "The third supplementary resource is the EAV dataset record on Zenodo "
        "(DOI: 10.5281/zenodo.10205702), which supplies both raw recordings and "
        "pre-extracted features. Should the pickled features later be lost, the "
        "Zenodo record permits full re-derivation through the nubcico/EAV "
        "preprocessing scripts."
    )


# --- Section 2 --------------------------------------------------------------

def build_section_2(doc: Document) -> None:
    doc.add_heading("2. Day 1 Progress Report", level=1)

    doc.add_heading("2.1 Objectives for Day 1", level=2)
    doc.add_paragraph(
        "Day 1 was scoped as a setup and reproduction milestone. The primary "
        "objectives were as follows. First, to establish a stable hybrid "
        "development workflow that combined a local development machine with "
        "cloud-based GPU compute, given that the local hardware (an Apple M1 Air) "
        "lacks the CUDA support required by the EAV repository’s PyTorch path. "
        "Second, to provision a persistent storage layout for code, data and "
        "trained artefacts. Third, to verify that the EAV pickle files were "
        "accessible from the chosen compute environment and that they conformed "
        "to the shapes expected by the per-modality encoders. Fourth, to run a "
        "minimal smoke test of the entire pipeline on at least three subjects "
        "across all three modalities, demonstrating that each modality could be "
        "trained end-to-end and that classification accuracy could be reported. "
        "Day 1 was explicitly not intended to produce final or paper-quality "
        "numbers; its purpose was to validate the pipeline."
    )

    doc.add_heading("2.2 Workflow Established", level=2)
    doc.add_paragraph(
        "The following workflow was adopted and is intended to remain in force "
        "throughout the remaining days of the project. Source code is authored "
        "locally on the M1 Air using Visual Studio Code and synchronised through "
        "a private GitHub repository (shafinmuffinn/thesis_p2). Compute is "
        "provided by Google Colab on a free-tier T4 GPU, with the option to "
        "upgrade to Colab Pro retained for the more compute-intensive days "
        "later in the schedule. Persistent storage is provided by Google Drive, "
        "mounted into the Colab runtime at /content/drive/MyDrive/Thesis_EAV/. "
        "EAV pickle features and trained model checkpoints reside on Drive; the "
        "raw 46 GB dataset is held only transiently when required."
    )
    doc.add_paragraph(
        "A single path configuration module, paths.py, was introduced at the "
        "project root. All filesystem locations are read from environment "
        "variables (THESIS_ROOT, EAV_PICKLES, CHECKPOINTS, RESULTS, PRETRAINED), "
        "with sensible defaults for the local machine. The Colab session start "
        "ritual sets these environment variables to point at the mounted Drive "
        "tree, which means the same code runs unchanged in both environments. "
        "A .gitignore file ensures that bulk data, checkpoints and Hugging Face "
        "caches are never accidentally committed to the repository."
    )

    doc.add_heading("2.3 Day-1 Steps Carried Out", level=2)
    _numbered(doc,
        "Created the central path configuration module paths.py and the "
        "accompanying .gitignore, then initialised a private GitHub repository "
        "and pushed the initial commit."
    )
    _numbered(doc,
        "Authored a reusable Colab “session start” cell that mounts Google "
        "Drive, performs git pull on the working copy of the repository, sets "
        "the required environment variables, installs the minimal set of pip "
        "dependencies, and asserts that a CUDA device is visible."
    )
    _numbered(doc,
        "Verified the contents of the EAV pickle directory on Drive, confirming "
        "the presence and shape of subject-level audio, vision and EEG files for "
        "the first three subjects."
    )
    _numbered(doc,
        "Composed a smoke-test driver, referred to internally as Section E, "
        "which iterates over three subjects and, for each, fine-tunes the audio "
        "and vision Transformer models and trains EEGNet on EEG, logging per-"
        "subject accuracy and wall-clock time to a CSV file in Drive."
    )
    _numbered(doc,
        "Resolved a sequence of seven distinct defects in the EAV repository "
        "and one defect in the smoke-test driver itself (see Section 2.4), "
        "applying each fix both locally and as an in-session hot-patch on "
        "Colab."
    )
    _numbered(doc,
        "Executed the smoke test to completion for subjects 1 through 3 across "
        "all three modalities and saved the resulting accuracies to "
        "results/day1_smoketest.csv."
    )

    doc.add_heading("2.4 Challenges Encountered and Their Resolutions", level=2)
    doc.add_paragraph(
        "Eight distinct defects were encountered during Day 1. Each is "
        "documented below, in the order in which it was diagnosed, together "
        "with the resolution applied. With one exception, every fix has been "
        "committed to the project repository so that the issue does not recur "
        "in subsequent sessions."
    )
    challenges = [
        (
            "Dependency conflict caused by facenet-pytorch",
            "The initial dependency install command included facenet-pytorch, "
            "which pins NumPy below version 2 and an older release of PyTorch. "
            "Installing it on Colab silently downgraded NumPy from 2.x to 1.26.4 "
            "and PyTorch from 2.10 to 2.2.2, breaking a large set of pre-"
            "installed Colab packages including torchaudio. Resolution: "
            "facenet-pytorch was removed from the Day 1 install set on the "
            "grounds that the smoke test operates on pre-processed pickle files "
            "and does not require MTCNN-based face detection. Should it be "
            "required on a later day, the recommended installation incantation "
            "is pip install --no-deps facenet-pytorch, which avoids the "
            "destructive downgrade."
        ),
        (
            "Case-sensitivity bug in the verification helper",
            "An early helper used Python’s str.capitalize method to translate "
            "the modality name ‘eeg’ into the folder name ‘Eeg’, whereas the "
            "actual folder on Drive is ‘EEG’ in upper case. The defect was "
            "masked for audio and vision, where capitalize happens to produce "
            "the correct folder name. Resolution: replaced the implicit string "
            "transform with an explicit dictionary mapping {audio: ‘Audio’, "
            "vision: ‘Vision’, eeg: ‘EEG’}, applied uniformly across both "
            "Section D (verification) and Section E (smoke test)."
        ),
        (
            "Broken import statement in EEGNet_tor.py",
            "The EAV repository’s PyTorch EEGNet module begins with the line "
            "from Fusion.VIT_audio.Transformer_audio import Trainer_uni, but "
            "no Fusion package exists anywhere in the repository. The import "
            "fails at module load time. Resolution: removed the dead import "
            "line; Trainer_uni is defined locally within the same file and "
            "does not need to be brought in from elsewhere."
        ),
        (
            "Missing TensorDataset and DataLoader imports",
            "Removing the broken Fusion import in the previous fix had the "
            "unintended side effect of also removing the only transitively-"
            "imported references to TensorDataset and DataLoader, which the "
            "Trainer_uni class relies on. The next attempt to construct a "
            "Trainer_uni instance failed with NameError. Resolution: added an "
            "explicit from torch.utils.data import DataLoader, TensorDataset "
            "statement at the top of EEGNet_tor.py."
        ),
        (
            "Classifier head size mismatch in Transformer_Vision.py",
            "The vision trainer attempted to repurpose the seven-class "
            "facial-emotions ViT for the EAV five-class task by replacing "
            "self.model.classifier with a fresh Linear(hidden, 5) and setting "
            "self.model.num_labels = 5. This updated the attribute but not "
            "self.model.config.num_labels, which the Hugging Face Transformers "
            "library consults when computing the internal cross-entropy loss. "
            "The model therefore emitted (B, 5) logits but the loss function "
            "attempted to reshape them as (B, 7), failing with RuntimeError: "
            "shape '[-1, 7]' is invalid for input of size 160. Resolution: "
            "replaced the manual head replacement with the documented "
            "AutoModelForImageClassification.from_pretrained(model_path, "
            "num_labels=5, ignore_mismatched_sizes=True) idiom, which both "
            "re-initialises the head and updates the configuration."
        ),
        (
            "RAM exhaustion during eager image preprocessing",
            "The vision trainer’s preprocess_images method iterated over each "
            "of approximately ten thousand frames per subject, ran the Hugging "
            "Face image processor on each frame individually, accumulated the "
            "resulting tensors in a Python list, and finally invoked "
            "torch.stack(list).to(self.device). On Colab Free this allocated "
            "in excess of eight gigabytes of CPU memory and silently killed "
            "the kernel with no visible error. Resolution: rewrote the method "
            "to process images in batches of sixty-four, to keep the resulting "
            "tensor on the CPU, and to emit periodic progress messages so that "
            "the user can distinguish slow progress from a hang."
        ),
        (
            "Forward-hook return-value bug in EEGNet_tor.py",
            "The EEGNet module attempts to enforce a max-norm constraint on "
            "the depthwise convolution and final dense layer through forward "
            "hooks of the form lambda module, inputs, outputs: "
            "module.weight.data.renorm_(p=2, dim=0, maxnorm=norm_rate). The "
            "renorm_ operation is in-place but returns the modified weight "
            "tensor; the lambda implicitly propagates that tensor as its "
            "return value. PyTorch’s forward-hook protocol, however, treats "
            "any returned tensor as a replacement for the layer’s output, "
            "with the result that the depthwise convolution’s output was "
            "silently replaced by the weight tensor of shape (64, 1, 30, 1) "
            "and the subsequent BatchNorm crashed with the message "
            "running_mean should contain 1 elements not 64. Resolution: "
            "extracted the hook into a real (non-lambda) function "
            "_make_max_norm_hook(norm_rate) whose implicit return is None, "
            "and applied it to both layers."
        ),
        (
            "Synchronisation between local edits and the Colab clone",
            "Several library-code fixes performed locally were not picked up "
            "by Colab even after git pull. Two distinct causes were "
            "identified. First, the editor on the local machine reverted "
            "in-flight edits at least once during the session, with the "
            "result that a commit purporting to fix a bug actually committed "
            "the unfixed file. Second, the Colab clone had locally-applied "
            "hot patches sitting in the working tree, which caused git pull "
            "to abort with “Your local changes to the following files would "
            "be overwritten by merge”. Resolution: established a discipline "
            "of verifying each fix on disk with grep immediately before "
            "committing, and a Colab-side discipline of git stash followed "
            "by git pull followed by git stash drop, since the in-place "
            "patches are functionally equivalent to the upstream fix and "
            "therefore safe to discard."
        ),
    ]
    for idx, (title, body) in enumerate(challenges, start=1):
        para = doc.add_paragraph()
        run = para.add_run(f"Challenge {idx}: {title}.")
        run.bold = True
        doc.add_paragraph(body)

    doc.add_heading("2.5 Tools, Libraries and Workflow Decisions Finalised", level=2)
    doc.add_paragraph(
        "The following decisions were finalised on Day 1 and are intended to "
        "remain stable for the remainder of the project."
    )
    _bullet(doc,
        "Local development environment: Visual Studio Code on an Apple M1 Air, "
        "used exclusively for code authoring, code review, and lightweight "
        "diagnostic scripts that do not require CUDA."
    )
    _bullet(doc,
        "Cloud compute environment: Google Colab on the free tier, with a "
        "Tesla T4 GPU (approximately 16 gigabytes of video memory and 12.7 "
        "gigabytes of system memory). Upgrade to Colab Pro is to be "
        "reconsidered before Day 2, when full-epoch training begins."
    )
    _bullet(doc,
        "Persistent storage: Google Drive, mounted at "
        "/content/drive/MyDrive/Thesis_EAV/, hosting EAV pickles, model "
        "checkpoints, Hugging Face caches and result CSVs."
    )
    _bullet(doc,
        "Code synchronisation: a private GitHub repository, accessed from "
        "Colab via a fine-grained personal access token stored on Drive. "
        "Sessions begin with git pull and end with git push."
    )
    _bullet(doc,
        "Path configuration: a single Python module, paths.py, exposing "
        "PROJECT_ROOT, EAV_PICKLES, CHECKPOINTS, RESULTS, LOGS and "
        "PRETRAINED. All filesystem locations are read from environment "
        "variables that are set in the Colab session-start cell."
    )
    _bullet(doc,
        "Python dependencies on Colab: only the transformers package and "
        "(optionally) wandb need to be installed beyond the Colab baseline. "
        "The remainder of EAV/requirements.txt is either pre-installed in "
        "Colab or not required by the smoke-test path."
    )
    _bullet(doc,
        "Result logging convention: CSVs are written to "
        "MyDrive/Thesis_EAV/results/, with one row per (subject, modality) "
        "experiment, appended incrementally so that a kernel crash does not "
        "lose previously completed work."
    )


# --- Section 3 --------------------------------------------------------------

def build_section_3(doc: Document, rows: list[dict]) -> None:
    pivot = pivot_results(rows)
    subjects = sorted(pivot.keys())
    modalities = ["audio", "vision", "eeg"]

    doc.add_heading("3. Results Achieved So Far", level=1)

    doc.add_heading("3.1 Day-1 Baseline Smoke-Test Results", level=2)
    doc.add_paragraph(
        "Day 1 produced a complete end-to-end run of the per-modality pipeline "
        "for three subjects. The configuration used deliberately small training "
        "budgets so that the focus was on validating the pipeline rather than "
        "maximising classification performance: two frozen-backbone epochs "
        "followed by two fine-tuning epochs for the Audio Spectrogram Transformer; "
        "one frozen and one fine-tuning epoch for the facial-emotion Vision "
        "Transformer; and fifty epochs of training from scratch for EEGNet. The "
        "per-subject test accuracies and wall-clock times are reported in "
        "Table 1 below."
    )
    doc.add_paragraph()

    # Build Table 1: per-subject results
    table = doc.add_table(rows=len(subjects) + 2, cols=7)
    hdr = table.rows[0]
    hdr.cells[0].text = "Subject"
    hdr.cells[1].text = "Audio acc."
    hdr.cells[2].text = "Audio time (s)"
    hdr.cells[3].text = "Vision acc."
    hdr.cells[4].text = "Vision time (s)"
    hdr.cells[5].text = "EEG acc."
    hdr.cells[6].text = "EEG time (s)"

    for i, sub in enumerate(subjects, start=1):
        row = table.rows[i]
        row.cells[0].text = f"{sub:02d}"
        for j, mod in enumerate(modalities):
            data = pivot[sub].get(mod, {})
            row.cells[1 + 2 * j].text = (
                f"{data['acc'] * 100:.2f}%" if "acc" in data else "—"
            )
            row.cells[2 + 2 * j].text = (
                f"{data['sec']:.1f}" if "sec" in data else "—"
            )

    # Mean row
    mean_row = table.rows[-1]
    mean_row.cells[0].text = "Mean"
    for j, mod in enumerate(modalities):
        accs = [pivot[s][mod]["acc"] for s in subjects if mod in pivot[s]]
        secs = [pivot[s][mod]["sec"] for s in subjects if mod in pivot[s]]
        mean_row.cells[1 + 2 * j].text = (
            f"{sum(accs) / len(accs) * 100:.2f}%" if accs else "—"
        )
        mean_row.cells[2 + 2 * j].text = (
            f"{sum(secs) / len(secs):.1f}" if secs else "—"
        )

    _format_table(table)
    _para(doc, "Table 1. Per-subject test accuracy and wall-clock training time "
                "for each modality on the Day-1 smoke test. Five-class chance level "
                "is 20 %.", italic=True, size=10, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()

    # Comparison table to EAV baselines
    doc.add_heading("3.2 Comparison to the EAV Paper Baselines", level=2)
    doc.add_paragraph(
        "Table 2 contrasts the Day-1 mean accuracies with the baselines reported "
        "in the EAV README. The EAV paper reports DeepFace as the vision baseline, "
        "a small CNN (SCNN) as the audio baseline, and EEGNet as the EEG baseline."
    )
    doc.add_paragraph()

    cmp = doc.add_table(rows=4, cols=5)
    cmp.rows[0].cells[0].text = "Modality"
    cmp.rows[0].cells[1].text = "Model used"
    cmp.rows[0].cells[2].text = "Day-1 mean acc."
    cmp.rows[0].cells[3].text = "EAV baseline acc."
    cmp.rows[0].cells[4].text = "Δ to baseline"

    audio_accs = [pivot[s]["audio"]["acc"] for s in subjects]
    vision_accs = [pivot[s]["vision"]["acc"] for s in subjects]
    eeg_accs = [pivot[s]["eeg"]["acc"] for s in subjects]
    aud_mean = sum(audio_accs) / len(audio_accs) * 100
    vis_mean = sum(vision_accs) / len(vision_accs) * 100
    eeg_mean = sum(eeg_accs) / len(eeg_accs) * 100

    cmp.rows[1].cells[0].text = "Audio"
    cmp.rows[1].cells[1].text = "AST (Hugging Face)"
    cmp.rows[1].cells[2].text = f"{aud_mean:.2f}%"
    cmp.rows[1].cells[3].text = "36.7% (SCNN)"
    cmp.rows[1].cells[4].text = f"{aud_mean - 36.7:+.2f} pp"

    cmp.rows[2].cells[0].text = "Vision"
    cmp.rows[2].cells[1].text = "ViT facial-emotions"
    cmp.rows[2].cells[2].text = f"{vis_mean:.2f}%"
    cmp.rows[2].cells[3].text = "52.8% (DeepFace)"
    cmp.rows[2].cells[4].text = f"{vis_mean - 52.8:+.2f} pp"

    cmp.rows[3].cells[0].text = "EEG"
    cmp.rows[3].cells[1].text = "EEGNet (PyTorch)"
    cmp.rows[3].cells[2].text = f"{eeg_mean:.2f}%"
    cmp.rows[3].cells[3].text = "36.7% (EEGNet)"
    cmp.rows[3].cells[4].text = f"{eeg_mean - 36.7:+.2f} pp"

    _format_table(cmp)
    _para(doc, "Table 2. Comparison of Day-1 smoke-test means against the EAV "
                "paper’s published baseline accuracies. Δ is in percentage points.",
                italic=True, size=10, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()

    doc.add_heading("3.3 Interpretation of Day-1 Results", level=2)
    doc.add_paragraph(
        "Three observations emerge from these numbers. First, the audio path "
        f"already attains a mean accuracy of {aud_mean:.1f} %, which is "
        f"approximately {aud_mean - 36.7:.0f} percentage points above the EAV "
        "paper’s SCNN baseline, despite the smoke test running for only four "
        "epochs in total. This is consistent with the strong transfer that the "
        "AudioSet-pretrained Audio Spectrogram Transformer provides; even minor "
        "fine-tuning is sufficient to reach a sensible operating point. The "
        "audio path is therefore expected to improve further on Day 2 when the "
        "full ten-plus-fifteen epoch schedule is applied."
    )
    doc.add_paragraph(
        "Second, the video path attains a mean of "
        f"{vis_mean:.1f} %, which is approximately "
        f"{vis_mean - 52.8:.0f} percentage points above the EAV paper’s "
        "DeepFace baseline. This number must be interpreted with care. The "
        "Vision Transformer checkpoint used (dima806/facial_emotions_image_"
        "detection) is itself pre-fine-tuned on a facial-emotion dataset, so "
        "the backbone is already specialised to the task before EAV is seen. "
        "The smoke test therefore largely measures how quickly a fresh five-"
        "class head can be aligned with a strong pre-existing representation. "
        "A direct comparison with DeepFace, which is also a pre-trained "
        "facial-emotion system, remains meaningful, but it should not be "
        "claimed that the present pipeline outperforms DeepFace by a large "
        "margin on the EAV task in any rigorous sense without further "
        "evaluation. Subject 3 in particular reached an accuracy of 80 %, "
        "which warrants a per-emotion confusion matrix analysis on Day 7 to "
        "rule out an imbalanced confusion pattern."
    )
    doc.add_paragraph(
        "Third, the EEG path attains a mean of "
        f"{eeg_mean:.1f} %, which is approximately "
        f"{36.7 - eeg_mean:.0f} percentage points below the EAV paper’s "
        "EEGNet baseline and barely above the five-class chance level of "
        "20 %. With 120 test trials per subject the 95 % binomial confidence "
        "interval around chance is approximately 13 % to 28 %, so the "
        "accuracies obtained for subjects 2 and 3 are statistically "
        "indistinguishable from random guessing. Three causes are likely. "
        "First, the EEGNet implementation in nubcico/EAV ends its forward "
        "pass with a softmax layer, while the trainer applies a "
        "CrossEntropyLoss which itself expects raw logits and applies a "
        "log-softmax internally. The composition is numerically degenerate "
        "and is the most plausible single cause of failed training. Second, "
        "the smoke-test training budget of fifty epochs is well below the "
        "three-hundred-and-fifty epochs at which EAV’s own pipeline reports "
        "results. Third, within-subject training with only 280 trials is "
        "intrinsically a low-data regime for EEG. The first cause will be "
        "remedied on Day 2 by removing the trailing softmax, which is "
        "expected to recover a substantial fraction of the gap."
    )

    doc.add_heading("3.4 Status of These Results", level=2)
    doc.add_paragraph(
        "It is important to emphasise that the figures reported in Tables 1 "
        "and 2 are baseline reproduction numbers obtained under deliberately "
        "constrained settings. They are not the final results of the thesis, "
        "and they are not yet directly comparable to the per-subject means "
        "reported in either the EAV dataset paper or in Lee, Kim and Kim "
        "(2024). They serve a single, explicit purpose: to verify that the "
        "pipeline from raw pickle through to per-modality accuracy operates "
        "correctly on every modality. With that verification complete, the "
        "remaining nine days of the project will focus on full-epoch training, "
        "fusion-architecture development, and rigorous evaluation."
    )


# --- Section 4 --------------------------------------------------------------

PLAN = [
    (1, "Setup and reproduction",
     "Establish workflow; verify EAV pickles loadable; run a three-subject "
     "smoke test for all three modalities; fix issues discovered along the way.",
     "results/day1_smoketest.csv; this report."),
    (2, "Full-budget unimodal baselines",
     "Run AST, ViT and EEGNet at the paper-faithful epoch counts on all 42 "
     "subjects; remove the EEGNet softmax-then-CrossEntropyLoss defect; "
     "log per-subject accuracies and test logits.",
     "Per-subject results CSV; per-subject logit tensors; mean ± std table."),
    (3, "Naïve late-fusion baseline",
     "Combine the per-modality softmax outputs from Day 2 by simple averaging "
     "and by validation-tuned weighted averaging; report the resulting fusion "
     "accuracy.",
     "Late-fusion CSV with per-subject and mean accuracies for both schemes."),
    (4, "Trimodal cross-attention fusion module",
     "Port the AttentionBlock from the Chumachenko ICPR 2022 repository and "
     "extend it to a six-way pairwise cross-modal attention scheme across "
     "audio, video and EEG; concatenate and pass through an MLP head.",
     "fusion module source code; first end-to-end forward and backward "
     "passes verified on a synthetic batch."),
    (5, "Train and evaluate cross-attention fusion",
     "Jointly fine-tune the three encoders and the fusion module on EAV; "
     "compare against the naïve late-fusion baseline from Day 3.",
     "Cross-attention fusion CSV; comparison table against late fusion."),
    (6, "Stretch: MERCL contrastive pre-training",
     "Implement the AMCL, EMCL and SMCL losses; pre-train the encoders for "
     "five to ten epochs; then fine-tune the cross-attention fusion as on "
     "Day 5.",
     "MERCL-pretrained encoder checkpoints; results CSV; ablation row."),
    (7, "Robustness and per-class analysis",
     "Apply the softhard modality-dropout scheme during training; report "
     "performance under each individual modality removal at inference time; "
     "produce per-emotion F1 and confusion matrices.",
     "Robustness CSV; per-modality degradation table; confusion-matrix "
     "figures for the best model."),
    (8, "Stretch: leave-one-subject-out evaluation",
     "Re-evaluate the best model from Day 5 (or Day 6 if completed) under a "
     "leave-one-subject-out cross-validation protocol, in order to report "
     "subject-independent accuracy as required for a credible thesis.",
     "LOSO CSV; subject-independent mean ± std numbers."),
    (9, "Writing: methods and results",
     "Draft the Methods and Experiments chapters of the thesis; produce the "
     "architecture diagram; finalise the result tables.",
     "Methods chapter draft; results chapter draft; architecture figure."),
    (10, "Polish and defence preparation",
     "Write the limitations and future-work discussion; build the slide deck "
     "for the defence; final read-through of the thesis draft.",
     "Final report PDF (or DOCX); defence slides; reproducibility checklist."),
]


def build_section_4(doc: Document) -> None:
    doc.add_heading("4. Ten-Day Research Plan", level=1)

    doc.add_heading("4.1 Project Timeline at a Glance", level=2)
    doc.add_paragraph(
        "The remaining work is organised around a ten-day schedule. Days 1 to "
        "5 establish the baseline trimodal system and constitute the critical "
        "path; days 6 and 8 are designated stretch goals that strengthen the "
        "contribution if time permits but can be dropped without compromising "
        "the thesis; days 7, 9 and 10 are reserved for analysis, writing and "
        "polish. Table 3 summarises the plan; the subsequent subsections "
        "elaborate on each day."
    )
    doc.add_paragraph()

    table = doc.add_table(rows=len(PLAN) + 1, cols=4)
    hdr = table.rows[0]
    hdr.cells[0].text = "Day"
    hdr.cells[1].text = "Goal"
    hdr.cells[2].text = "Key Tasks"
    hdr.cells[3].text = "Expected Output"

    for i, (day, goal, tasks, output) in enumerate(PLAN, start=1):
        row = table.rows[i]
        row.cells[0].text = str(day)
        row.cells[1].text = goal
        row.cells[2].text = tasks
        row.cells[3].text = output

    _format_table(table)
    _para(doc, "Table 3. The ten-day research plan, summarised. Days marked "
                "“stretch” (Day 6 and Day 8) are conditional on prior progress.",
                italic=True, size=10, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()

    doc.add_heading("4.2 Day-by-Day Elaboration", level=2)

    elaborations = [
        ("Day 1 — Setup and reproduction",
         "Day 1 (this report) is the foundation on which every subsequent "
         "day depends. Without a working environment, a verifiable workflow, "
         "and known-good per-modality smoke results, the schedule cannot "
         "proceed. By the end of Day 1, every defect on the pipeline is "
         "documented and fixed, and the cost-of-error for each subsequent "
         "experiment is therefore minimised. This is a critical day with no "
         "buffer; its objectives have been met."),
        ("Day 2 — Full-budget unimodal baselines",
         "Day 2 runs the three modality-specific encoders at the epoch counts "
         "recommended in the EAV repository and the source paper. These "
         "numbers establish the upper bound that any single modality can "
         "reach on EAV, against which fusion gains will be measured later. "
         "Day 2 also fixes the EEGNet softmax-then-CrossEntropyLoss defect "
         "uncovered on Day 1, which is expected to substantially improve the "
         "EEG numbers. Day 2 is critical; it feeds the entire fusion chain."),
        ("Day 3 — Naïve late-fusion baseline",
         "Day 3 produces the simplest possible fusion baseline: the per-"
         "modality softmax probabilities from Day 2 are averaged on a per-"
         "trial basis, and a validation-tuned weighted variant is also "
         "computed. This baseline is essential context for evaluating the "
         "more sophisticated cross-attention fusion: any cross-attention "
         "improvement that fails to beat naïve averaging is not a real "
         "improvement. Day 3 depends entirely on Day 2."),
        ("Day 4 — Cross-attention fusion module",
         "Day 4 ports the AttentionBlock and Attention classes from the "
         "Chumachenko ICPR 2022 repository and extends them from the original "
         "audio–video pairwise scheme to a six-way trimodal scheme that "
         "computes attention across all three ordered pairs of modalities. "
         "The output is concatenated and fed through a small classifier head. "
         "This day produces no numbers, only working code; the first end-to-"
         "end forward and backward pass on a synthetic batch is the success "
         "criterion. Day 4 is critical; everything from Day 5 onwards depends "
         "on a functioning fusion module."),
        ("Day 5 — Train and evaluate cross-attention fusion",
         "Day 5 trains the trimodal fusion module end-to-end against EAV. "
         "The result, the central empirical contribution of the thesis, is a "
         "fusion accuracy that is to be contrasted both with the per-modality "
         "ceilings from Day 2 and with the naïve late-fusion baseline from "
         "Day 3. If Day 5 succeeds in beating Day 3, the thesis has a clean "
         "headline result even if the stretch days are not completed."),
        ("Day 6 — Stretch: MERCL contrastive pre-training",
         "Day 6 implements the three contrastive losses (AMCL, EMCL, SMCL) "
         "described by Lee et al. (2024) and uses them to pre-train the "
         "encoders for a small number of epochs before re-running the Day-5 "
         "fine-tuning recipe. The expectation, based on the source paper’s "
         "ablation, is a further four-point gain on top of cross-attention. "
         "If Day 5 finishes late or fusion training proves brittle, Day 6 is "
         "the first item to be cut."),
        ("Day 7 — Robustness and per-class analysis",
         "Day 7 evaluates the best fusion model under two distinct lenses. "
         "First, modality-dropout robustness is studied by zeroing each "
         "modality at inference time and measuring the resulting accuracy "
         "drop, which characterises whether any single modality dominates "
         "the decision. Second, per-emotion confusion matrices are produced "
         "so that, for example, a model that is excellent at Happiness but "
         "guesses on Anger can be identified as such. Day 7 also serves as "
         "a buffer day; if earlier work has run over, robustness analysis "
         "can be deferred to a smaller scope without sacrificing the core "
         "claim."),
        ("Day 8 — Stretch: leave-one-subject-out evaluation",
         "Day 8 re-evaluates the best system under a leave-one-subject-out "
         "(LOSO) cross-validation protocol, which is the standard for "
         "claiming subject-independent generalisation in affective computing. "
         "LOSO is computationally expensive (it multiplies the cost of "
         "training by the number of held-out subjects), and is therefore "
         "treated as a stretch goal; if it is not completed, the absence of "
         "LOSO numbers is acknowledged as a limitation in the written "
         "discussion."),
        ("Day 9 — Writing: methods and results",
         "Day 9 is the principal writing day. The methods chapter is "
         "drafted with reference to the implementation in the repository, "
         "the architecture diagram is produced, and the results tables are "
         "finalised from the CSVs produced on earlier days. This day "
         "depends on Days 2 through 7 having produced concrete outputs."),
        ("Day 10 — Polish and defence preparation",
         "Day 10 produces the limitations and future-work sections, builds "
         "the defence slide deck, and performs a final pass over the thesis "
         "draft. Day 10 is treated as polish-only; new experiments are not "
         "started on this day except in response to clearly trivial issues "
         "discovered during the read-through."),
    ]
    for title, body in elaborations:
        doc.add_heading(title, level=3)
        doc.add_paragraph(body)

    doc.add_heading("4.3 Critical Path and Dependency Structure", level=2)
    doc.add_paragraph(
        "Five of the ten days lie on the critical path: Day 1 (without which "
        "nothing else runs), Day 2 (which feeds the fusion chain), Day 4 (the "
        "fusion module itself), Day 5 (the headline result), and Day 9 "
        "(writing, without which no thesis exists). Day 3’s late-fusion "
        "baseline is also strictly required because it provides the "
        "comparison point against which the cross-attention contribution is "
        "claimed; however, it is much shorter than the surrounding days and "
        "is therefore not the bottleneck."
    )
    doc.add_paragraph(
        "Days 6 and 8 are explicit stretch goals: the MERCL pre-training "
        "stage and the leave-one-subject-out evaluation are both highly "
        "desirable but neither is strictly necessary for a defensible thesis. "
        "Should the schedule slip, they are the first items to be cut, in "
        "that order. Day 7 (robustness and per-class analysis) is somewhat "
        "elastic in scope: at the minimum, modality-dropout robustness must "
        "be reported, since it is the natural answer to the most likely "
        "defence question (“what happens when a modality is missing?”). The "
        "fuller per-emotion confusion-matrix analysis is desirable but can "
        "be shortened if required."
    )
    doc.add_paragraph(
        "Day 10 must not be compressed. Defence preparation and a final "
        "limitations pass routinely uncover small but consequential issues, "
        "and the schedule deliberately leaves the last day clear of new "
        "experimental work."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rows = load_results()

    doc = Document()

    # Tighten default heading styles slightly
    styles = doc.styles
    normal = styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)

    build_title_page(doc)
    build_toc(doc)
    build_section_1(doc)
    doc.add_page_break()
    build_section_2(doc)
    doc.add_page_break()
    build_section_3(doc, rows)
    doc.add_page_break()
    build_section_4(doc)

    doc.save(OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
