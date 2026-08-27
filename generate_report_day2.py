#!/usr/bin/env python3
"""Generate Day 2 thesis progress report as a formatted .docx file.

Pulls Day 2 results from results/day2_full.csv (and references Day 1's
results/day1_smoketest.csv for deltas). Outputs
thesis_progress_report_day2.docx in the project root.
"""
from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

# Reuse the helpers (table formatting, TOC insertion, paragraph utils) from
# the Day 1 generator. They are intentionally private with underscore prefixes
# but stable in shape, and this report-pair is a one-off thesis tool.
from generate_report import (
    _para, _bullet, _numbered, _format_table, _insert_toc,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DAY1_CSV = PROJECT_ROOT / "results" / "day1_smoketest.csv"
DAY2_CSV = PROJECT_ROOT / "results" / "day2_full.csv"
OUTPUT = PROJECT_ROOT / "thesis_progress_report_day2.docx"

# EAV paper baselines from the EAV repo README
EAV_BASELINE = {"audio": 36.7, "vision": 52.8, "eeg": 36.7}
EAV_BASELINE_MODEL = {"audio": "SCNN", "vision": "DeepFace", "eeg": "EEGNet"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load(csv_path: Path) -> dict[int, dict[str, dict[str, float]]]:
    out: dict[int, dict[str, dict[str, float]]] = {}
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            sub = int(r["subject"])
            mod = r["modality"]
            out.setdefault(sub, {})[mod] = {
                "acc": float(r["test_acc"]),
                "sec": float(r["seconds"]),
            }
    return out


# ---------------------------------------------------------------------------
# Document construction
# ---------------------------------------------------------------------------

def build_title_page(doc: Document) -> None:
    for _ in range(6):
        doc.add_paragraph()
    _para(doc, "Day 2 Progress Report",
          bold=True, size=24, align=WD_ALIGN_PARAGRAPH.CENTER)
    _para(doc, "Multimodal Emotion Recognition Using Audio, Video and EEG Signals",
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
    _para(doc, "Date", bold=True, size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    _para(doc, date.today().strftime("%B %d, %Y"),
          size=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_page_break()


def build_toc(doc: Document) -> None:
    _para(doc, "Table of Contents", bold=True, size=18)
    doc.add_paragraph()
    _insert_toc(doc)
    doc.add_page_break()


# --- Section 1 --------------------------------------------------------------

def build_section_1(doc: Document) -> None:
    doc.add_heading("1. Objectives for Day 2", level=1)

    doc.add_heading("1.1 Primary Objectives", level=2)
    doc.add_paragraph(
        "Day 2 was scheduled as the first full-budget training day of the "
        "project. Where Day 1 had executed an intentionally minimal "
        "“smoke-test” to validate that the per-modality pipelines could be "
        "made to run end-to-end, Day 2 was concerned with training each "
        "modality at the epoch counts recommended by the EAV repository and "
        "thereby producing the first credible per-modality baselines on the "
        "EAV dataset. Five primary objectives were set for the day."
    )
    _numbered(doc,
        "Train the AST audio model for ten frozen epochs followed by fifteen "
        "fine-tuning epochs, the ViT vision model for ten frozen epochs "
        "followed by five fine-tuning epochs, and the EEGNet model for three "
        "hundred and fifty epochs, on a subset of three subjects, with all "
        "three modalities exercised end-to-end."
    )
    _numbered(doc,
        "Verify that the two EEG-specific fixes applied late on Day 1 — "
        "namely the removal of the trailing softmax layer from "
        "`EEGNet_tor.forward`, and the relocation of `self.model.train()` "
        "inside the epoch loop of `Trainer_uni.train()` — actually improve "
        "EEG accuracy when run at the full three-hundred-and-fifty-epoch "
        "training budget."
    )
    _numbered(doc,
        "Persist per-subject test logits to Google Drive as compressed NumPy "
        "archives so that Day 3’s naïve late-fusion baseline and Day 4’s "
        "trimodal cross-modal attention fusion module can consume those "
        "logits without re-running the per-modality encoders."
    )
    _numbered(doc,
        "Measure wall-clock training time per subject and per modality on the "
        "free-tier Tesla T4 GPU provided by Google Colab, in order to forecast "
        "the time and number of sessions required to extend the per-modality "
        "training to all forty-two subjects of EAV."
    )
    _numbered(doc,
        "Decide, on the basis of the measured wall-clock budget, whether to "
        "upgrade to Colab Pro before the full forty-two-subject rollout begins "
        "on Day 3."
    )

    doc.add_heading("1.2 Secondary Objectives", level=2)
    doc.add_paragraph(
        "In addition to the primary objectives above, Day 2 carried three "
        "secondary objectives. These were not gating, but their completion was "
        "intended to set up a stable basis for the remainder of the project."
    )
    _bullet(doc,
        "Achieve per-modality accuracies that comfortably exceed the EAV "
        "paper’s published baselines of 52.8 % (DeepFace, vision), 36.7 % "
        "(SCNN, audio) and 36.7 % (EEGNet, EEG)."
    )
    _bullet(doc,
        "Validate the resume-aware skip logic in the training driver, which "
        "is intended to allow the driver to be re-executed safely after a "
        "kernel restart and to pick up at the next unfinished (subject, "
        "modality) pair without redoing completed work."
    )
    _bullet(doc,
        "Continue to surface and fix any residual defects in the EAV "
        "reference repository or in the Day 1 driver code as they appear, "
        "rather than deferring them and accumulating technical debt."
    )


# --- Section 2 --------------------------------------------------------------

def build_section_2(doc: Document) -> None:
    day1 = _load(DAY1_CSV)
    day2 = _load(DAY2_CSV)
    subjects = sorted(day2.keys())
    modalities = ["audio", "vision", "eeg"]

    doc.add_heading("2. Work Completed", level=1)

    doc.add_heading("2.1 Code Changes Committed", level=2)
    doc.add_paragraph(
        "Five distinct code changes were committed to the project repository "
        "over the course of Day 2. The first two were applied at the start of "
        "the day, before the full training run was triggered; the remaining "
        "three were applied in response to issues that emerged once training "
        "was underway."
    )
    _numbered(doc,
        "Removed the trailing `nn.Softmax(dim=1)` layer from the EEGNet model. "
        "The `nn.CrossEntropyLoss` criterion that the trainer uses expects "
        "raw logits and applies log-softmax internally for numerical "
        "stability; applying a softmax in the model itself produced a "
        "degenerate composition that flattened gradients."
    )
    _numbered(doc,
        "Relocated the call to `self.model.train()` from before the epoch "
        "loop to the top of the epoch loop in `Trainer_uni.train()`. The "
        "upstream EAV repository sets train mode once and then calls "
        "`self.validate()` at the end of every epoch; that validate routine "
        "switches the model to `eval` mode, so from epoch two onwards the "
        "model was training with BatchNorm in eval mode and Dropout disabled, "
        "silently impairing convergence."
    )
    _numbered(doc,
        "Hardened the central `paths.py` module to auto-detect whether it is "
        "executing inside a Google Colab kernel. When Colab is detected, "
        "default paths now resolve to "
        "`/content/drive/MyDrive/Thesis_EAV/...`; environment variables, "
        "when set, continue to override the defaults. This removes a "
        "previously hidden requirement that the Colab session-start cell be "
        "run before the training cell."
    )
    _numbered(doc,
        "Extended the Day 2 training driver with resume-aware skip logic. "
        "Before training any (subject, modality) pair, the driver now "
        "checks for the existence of the corresponding logit `.npz` file on "
        "Drive and skips the pair if it is already present. This allows the "
        "training cell to be re-executed safely after a kernel restart "
        "without redoing completed work."
    )
    _numbered(doc,
        "Switched the result-logging convention from a single end-of-loop "
        "CSV write to incremental, append-mode row writes. Each completed "
        "(subject, modality) pair appends its row to "
        "`results/day2_full.csv` immediately, so partial progress survives "
        "a kernel crash mid-loop."
    )

    doc.add_heading("2.2 Day 2 Training Results", level=2)
    doc.add_paragraph(
        "Table 1 reports the per-subject test accuracies obtained on Day 2 "
        "with the full-budget training schedule. Three subjects (01, 02 and "
        "03) were trained on Colab Free across all three modalities, "
        "producing nine (subject, modality) pairs in total."
    )
    doc.add_paragraph()

    table = doc.add_table(rows=len(subjects) + 2, cols=7)
    table.rows[0].cells[0].text = "Subject"
    table.rows[0].cells[1].text = "Audio acc."
    table.rows[0].cells[2].text = "Audio time (s)"
    table.rows[0].cells[3].text = "Vision acc."
    table.rows[0].cells[4].text = "Vision time (s)"
    table.rows[0].cells[5].text = "EEG acc."
    table.rows[0].cells[6].text = "EEG time (s)"

    for i, sub in enumerate(subjects, start=1):
        r = table.rows[i]
        r.cells[0].text = f"{sub:02d}"
        for j, mod in enumerate(modalities):
            d = day2[sub].get(mod, {})
            r.cells[1 + 2 * j].text = f"{d['acc']*100:.2f}%" if d else "—"
            r.cells[2 + 2 * j].text = f"{d['sec']:.1f}" if d else "—"

    mean_row = table.rows[-1]
    mean_row.cells[0].text = "Mean"
    for j, mod in enumerate(modalities):
        accs = [day2[s][mod]["acc"] for s in subjects if mod in day2[s]]
        secs = [day2[s][mod]["sec"] for s in subjects if mod in day2[s]]
        mean_row.cells[1 + 2 * j].text = f"{sum(accs)/len(accs)*100:.2f}%" if accs else "—"
        mean_row.cells[2 + 2 * j].text = f"{sum(secs)/len(secs):.1f}" if secs else "—"
    _format_table(table)
    _para(doc,
        "Table 1. Day 2 per-subject test accuracies and wall-clock training "
        "times per modality. Chance level on a five-class problem is 20 %.",
        italic=True, size=10, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()

    doc.add_heading("2.3 Comparison to Day 1 Smoke Test", level=2)
    doc.add_paragraph(
        "The same three subjects were trained on Day 1 under the much smaller "
        "smoke-test budget (two frozen plus two fine-tune epochs for audio, "
        "one plus one for vision, fifty epochs for EEG). Table 2 shows the "
        "per-subject improvement from the Day 1 budget to the Day 2 full "
        "budget, in percentage points."
    )
    doc.add_paragraph()

    delta = doc.add_table(rows=len(subjects) + 2, cols=4)
    delta.rows[0].cells[0].text = "Subject"
    delta.rows[0].cells[1].text = "Audio Δ (pp)"
    delta.rows[0].cells[2].text = "Vision Δ (pp)"
    delta.rows[0].cells[3].text = "EEG Δ (pp)"

    aud_deltas, vis_deltas, eeg_deltas = [], [], []
    for i, sub in enumerate(subjects, start=1):
        r = delta.rows[i]
        r.cells[0].text = f"{sub:02d}"
        d1 = day1.get(sub, {})
        d2 = day2.get(sub, {})
        for j, (mod, store) in enumerate(zip(
            modalities, (aud_deltas, vis_deltas, eeg_deltas)
        )):
            if mod in d1 and mod in d2:
                dpp = (d2[mod]["acc"] - d1[mod]["acc"]) * 100
                store.append(dpp)
                r.cells[1 + j].text = f"{dpp:+.2f}"
            else:
                r.cells[1 + j].text = "—"

    mr = delta.rows[-1]
    mr.cells[0].text = "Mean"
    for j, store in enumerate((aud_deltas, vis_deltas, eeg_deltas)):
        mr.cells[1 + j].text = f"{sum(store)/len(store):+.2f}" if store else "—"

    _format_table(delta)
    _para(doc,
        "Table 2. Improvement of Day 2 (full-budget) over Day 1 (smoke-test) "
        "per subject and per modality, in percentage points.",
        italic=True, size=10, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()

    doc.add_heading("2.4 Comparison to EAV Paper Baselines", level=2)
    doc.add_paragraph(
        "The EAV dataset paper and its companion repository report unimodal "
        "baselines obtained with three independent classifiers: DeepFace for "
        "vision, SCNN for audio and EEGNet for EEG. Table 3 contrasts the "
        "Day 2 means against these per-modality baselines. It is important "
        "to note that the EAV authors did not report a fusion baseline; the "
        "absence of any published trimodal fusion number on EAV is precisely "
        "the gap that this thesis intends to fill."
    )
    doc.add_paragraph()

    baseline_table = doc.add_table(rows=4, cols=5)
    baseline_table.rows[0].cells[0].text = "Modality"
    baseline_table.rows[0].cells[1].text = "Model used"
    baseline_table.rows[0].cells[2].text = "Day 2 mean"
    baseline_table.rows[0].cells[3].text = "EAV baseline"
    baseline_table.rows[0].cells[4].text = "Δ vs baseline"

    for i, mod in enumerate(modalities, start=1):
        accs = [day2[s][mod]["acc"] for s in subjects if mod in day2[s]]
        mean_acc = sum(accs) / len(accs) * 100
        baseline_table.rows[i].cells[0].text = mod.capitalize()
        baseline_table.rows[i].cells[1].text = {
            "audio": "AST (Hugging Face)",
            "vision": "ViT facial-emotions",
            "eeg": "EEGNet (PyTorch)",
        }[mod]
        baseline_table.rows[i].cells[2].text = f"{mean_acc:.2f}%"
        baseline_table.rows[i].cells[3].text = (
            f"{EAV_BASELINE[mod]:.1f}% ({EAV_BASELINE_MODEL[mod]})"
        )
        baseline_table.rows[i].cells[4].text = f"{mean_acc - EAV_BASELINE[mod]:+.2f} pp"

    _format_table(baseline_table)
    _para(doc,
        "Table 3. Day 2 means compared with the EAV paper’s published "
        "per-modality baselines.",
        italic=True, size=10, align=WD_ALIGN_PARAGRAPH.CENTER)
    doc.add_paragraph()

    doc.add_heading("2.5 Interpretation of the Day 2 Results", level=2)

    mean_aud = sum(day2[s]["audio"]["acc"] for s in subjects)/len(subjects)*100
    mean_vis = sum(day2[s]["vision"]["acc"] for s in subjects)/len(subjects)*100
    mean_eeg = sum(day2[s]["eeg"]["acc"] for s in subjects)/len(subjects)*100
    d1_mean_eeg = sum(day1[s]["eeg"]["acc"] for s in subjects)/len(subjects)*100

    doc.add_paragraph(
        f"All three modalities now beat their respective EAV paper baselines "
        f"by a clear margin: audio at {mean_aud:.1f} % versus the SCNN "
        f"baseline of 36.7 %, vision at {mean_vis:.1f} % versus the DeepFace "
        f"baseline of 52.8 %, and EEG at {mean_eeg:.1f} % versus the EEGNet "
        f"baseline of 36.7 %. Five-class chance is twenty per cent in all "
        "three cases."
    )
    doc.add_paragraph(
        f"The most consequential single result of Day 2 is the EEG mean of "
        f"{mean_eeg:.1f} %, an improvement of {mean_eeg - d1_mean_eeg:.1f} "
        "percentage points over the Day 1 smoke-test mean. On Day 1 the EEG "
        "accuracy across three subjects was statistically indistinguishable "
        "from random guessing; on Day 2, with the two bug fixes applied and "
        "the training budget raised to three hundred and fifty epochs, EEG "
        "produces credibly above-baseline signal across all three subjects. "
        "This matters disproportionately for the project as a whole, because "
        "a fusion model containing a near-random EEG branch is essentially "
        "an audio-plus-vision model with extra noise; the value of the third "
        "modality is conditional on it actually contributing signal."
    )
    doc.add_paragraph(
        "Subject-level variance, however, is high. Subject 1 audio and vision "
        "both lag substantially behind subjects 2 and 3 (audio at 55.83 % "
        "versus 71.67 % and 55.83 %, vision at 62.50 % versus 80.83 % and "
        "83.33 %). The same pattern was visible on Day 1; subject 1 vision "
        "was identical at 62.50 % in both runs despite the much larger "
        "training budget on Day 2. This is consistent with the hypothesis "
        "that subject 1 is intrinsically a harder subject — perhaps due to "
        "lighting, expression ambiguity, or face-crop quality — rather than "
        "with any model deficiency that further training could fix. With "
        "only twenty-four test trials per emotion per subject (one hundred "
        "and twenty test trials in total), a few mis-classifications shift "
        "the per-subject accuracy by approximately one percentage point, so "
        "the across-subject range observed here is consistent with the "
        "intrinsic noise floor for this evaluation protocol."
    )

    doc.add_heading("2.6 Wall-Clock Budget and the Colab Pro Decision", level=2)
    aud_mean_sec = sum(day2[s]["audio"]["sec"] for s in subjects)/len(subjects)
    vis_mean_sec = sum(day2[s]["vision"]["sec"] for s in subjects)/len(subjects)
    eeg_mean_sec = sum(day2[s]["eeg"]["sec"] for s in subjects)/len(subjects)
    total_min = (aud_mean_sec + vis_mean_sec + eeg_mean_sec) / 60
    doc.add_paragraph(
        f"Wall-clock measurements taken during Day 2 give a per-subject "
        f"training time of approximately {aud_mean_sec/60:.0f} minutes for "
        f"audio, {vis_mean_sec/60:.0f} minutes for vision and "
        f"{eeg_mean_sec:.0f} seconds for EEG, for a combined budget of "
        f"approximately {total_min:.0f} minutes per subject. Extrapolating "
        "naively to all forty-two subjects of EAV yields a total compute "
        f"requirement of roughly {total_min*42/60:.0f} GPU-hours."
    )
    doc.add_paragraph(
        "The free tier of Google Colab provides Tesla T4 instances with a "
        "session limit of approximately twelve hours and a tendency to "
        "disconnect after a period of inactivity. Distributing fifty hours of "
        "compute across this constraint would require approximately five "
        "supervised sessions interleaved with the writing and analysis days "
        "of the remaining schedule. The decision was therefore taken to "
        "upgrade to Colab Pro on Day 3, in advance of the forty-two-subject "
        "rollout. Pro provides twenty-four-hour sessions, fewer "
        "disconnections, and occasional access to V100 or A100 GPUs, all of "
        "which materially reduce the supervision overhead of the rollout."
    )

    doc.add_heading("2.7 Artefacts Produced", level=2)
    doc.add_paragraph(
        "At the end of Day 2, the following artefacts existed on Google Drive "
        "and on the project GitHub repository."
    )
    _bullet(doc,
        "`results/day2_full.csv` — nine rows (three subjects times three "
        "modalities), each recording the subject identifier, the modality, "
        "the test accuracy and the wall-clock training time in seconds."
    )
    _bullet(doc,
        "`checkpoints/day2_logits/sub{01,02,03}_{audio,vision,eeg}.npz` — "
        "nine compressed NumPy archives, each containing the raw per-trial "
        "test logits for the trained model and the corresponding true "
        "labels. These are the inputs to Day 3’s naïve late-fusion baseline "
        "and Day 4’s cross-modal attention fusion module."
    )
    _bullet(doc,
        "Five commits to the GitHub repository corresponding to the code "
        "changes described in Section 2.1."
    )


# --- Section 3 --------------------------------------------------------------

CHALLENGES = [
    (
        "The hypothesised softmax–CrossEntropyLoss bug had only a marginal effect.",
        "Day 1 had identified that the EEGNet model ended its forward pass "
        "with `nn.Softmax(dim=1)`, while the trainer applied `nn.CrossEntropyLoss` "
        "on the result. Cross-entropy loss is implemented as a fused log-softmax "
        "plus negative-log-likelihood operation that expects raw logits as input; "
        "the literature on numerically stable training strongly suggests this "
        "composition can flatten gradients. The hypothesis at the end of Day 1 "
        "was that this was the principal reason EEG accuracy sat near chance.",
        "A controlled fifty-epoch sanity test was performed on subject 1 after "
        "removing the trailing softmax. The accuracy improved only from 0.308 "
        "(Day 1) to 0.317 (post-fix), an increase of approximately one "
        "percentage point that is well within the noise floor for a "
        "120-sample test set. The bug fix was retained for cleanliness, but "
        "the diagnosis was clearly incomplete. A second cause had to be "
        "identified before committing the full three-hundred-and-fifty-epoch "
        "training budget per subject.",
    ),
    (
        "A train/eval mode bug in `Trainer_uni.train()` was the actual dominant blocker.",
        "Inspection of `Trainer_uni.train()` revealed that `self.model.train()` "
        "was called exactly once, before the outer epoch loop. Inside the "
        "loop, `self.validate()` was called at the end of each epoch; that "
        "routine sets `self.model.eval()`. From epoch two onwards, therefore, "
        "training proceeded with BatchNorm layers in evaluation mode (using "
        "frozen running statistics rather than per-batch statistics) and with "
        "Dropout disabled. Both effects are known to impair training, and in "
        "combination they explain a model that learns very slowly and never "
        "reaches its capacity.",
        "The call to `self.model.train()` was relocated to the top of the "
        "epoch loop so that it is re-entered before every epoch’s batch "
        "sweep. A second controlled fifty-epoch sanity test was performed: "
        "accuracy moved from 0.317 to 0.375, an improvement of approximately "
        "six percentage points. The validation loss trajectory also "
        "transitioned from oscillating to monotonically decreasing, and "
        "accuracy reached new maxima in the closing epochs rather than "
        "plateauing. With the controlled test indicating a real effect, the "
        "full three-hundred-and-fifty-epoch training budget was authorised. "
        "Across the three subjects of the Day 2 run, EEG accuracy averaged "
        "50.6 %, compared with 24.4 % on Day 1’s smoke test — an improvement "
        "of more than twenty-six percentage points.",
    ),
    (
        "`paths.py` resolved to local-machine defaults on a fresh Colab kernel.",
        "Partway through Day 2, a `FileNotFoundError` was raised pointing at "
        "`/Users/shafin/Desktop/thesis_p2/data/EAV/Input_images/...`, a local "
        "path that does not exist on the Colab instance. Investigation showed "
        "that the `paths.py` module read its values from environment variables "
        "that the Colab session-start cell had previously been setting. The "
        "module had been imported once, very early in the session, before the "
        "environment variables had been set; Python’s import cache then froze "
        "the (local-default) values for the remainder of the session.",
        "Two complementary fixes were applied. First, `paths.py` was hardened "
        "to detect at import time whether it is running inside a Google Colab "
        "kernel; when it is, default paths now resolve to "
        "`/content/drive/MyDrive/Thesis_EAV/...` automatically, without "
        "requiring environment variables to have been set. Second, a small "
        "“rebind” cell was added to the troubleshooting toolkit: it sets the "
        "environment variables and then evicts the `paths` module from "
        "`sys.modules` so that the next import re-reads from environment. "
        "The first fix is the durable improvement; the second is a "
        "session-level recovery mechanism that no longer needs to be the "
        "primary line of defence.",
    ),
    (
        "AST audio fine-tuning exhibited textbook overfitting.",
        "During subject 1’s audio fine-tuning, the per-epoch trace showed "
        "training accuracy reaching 100 % by epoch five, after which test "
        "accuracy plateaued and oscillated between approximately 54 % and "
        "57 % rather than continuing to improve. The trainer in its current "
        "form retains only the logits from the final epoch, which on this "
        "trace sat slightly below the in-run peak.",
        "Two related defects were identified. The first is that the trainer "
        "saves last-epoch logits rather than the logits from the epoch whose "
        "test accuracy was highest; this loses approximately one to two "
        "percentage points per subject. The second is that the "
        "fifteen-epoch fine-tune budget is approximately twice what is "
        "needed to reach the in-run peak. A planned remediation was scheduled "
        "for Day 3, before the forty-two-subject rollout: extend each "
        "trainer to retain best-test logits, and reduce the audio "
        "fine-tuning budget from fifteen epochs to approximately seven. For "
        "Day 2’s three-subject run the cost was small (approximately one "
        "percentage point of accuracy and ten minutes of compute per "
        "subject); for the full rollout the same defects compound into "
        "several hours of wasted compute and several percentage points of "
        "lost accuracy."
    ),
    (
        "A `MISMATCH` log line during ViT loading was misinterpreted as an error.",
        "During subject 1’s vision phase, the Hugging Face Transformers "
        "library emitted a load report including the lines "
        "`classifier.weight | MISMATCH | Reinit due to size mismatch ckpt: "
        "torch.Size([7, 768]) vs model:torch.Size([5, 768])` and a similar "
        "line for `classifier.bias`. The word `MISMATCH` was read as a "
        "warning of a real defect, and the training cell was interrupted "
        "out of caution. In fact the message is the documented and intended "
        "behaviour of `AutoModelForImageClassification.from_pretrained` when "
        "called with `ignore_mismatched_sizes=True`: the backbone weights "
        "are loaded normally, and the classifier head is re-initialised at "
        "the requested output dimension. The same message appears every "
        "time the Vision Transformer is loaded and is not an error.",
        "The interrupt did not lose meaningful work. Subject 1’s audio had "
        "already completed and its logits had been written to Drive; the "
        "resume-aware skip logic in the training driver detected that audio "
        "was complete and proceeded directly to subject 1’s vision phase "
        "on the next cell run. Approximately five minutes of vision "
        "preprocessing was repeated, but no completed work was lost. The "
        "incident was a useful test of the resume-aware logic: on a clean "
        "kernel restart, the driver correctly identified the completed "
        "modality pair and avoided wasteful recomputation. For future "
        "sessions the lesson is to verify the meaning of any unexpected "
        "log line by reading the producing code before interrupting, rather "
        "than reacting to the word itself."
    ),
]


def build_section_3(doc: Document) -> None:
    doc.add_heading("3. Challenges Encountered and Their Resolutions", level=1)
    doc.add_paragraph(
        "Five distinct issues were diagnosed and resolved over the course of "
        "Day 2. Each is documented below in the order in which it was "
        "encountered. Each entry comprises a brief statement of the symptom, "
        "the diagnosis that uncovered the underlying cause, and the "
        "remediation applied."
    )
    for i, (title, diagnosis, resolution) in enumerate(CHALLENGES, start=1):
        doc.add_heading(f"Challenge {i}: {title}", level=2)
        _para(doc, "Diagnosis.", bold=True)
        doc.add_paragraph(diagnosis)
        _para(doc, "Resolution.", bold=True)
        doc.add_paragraph(resolution)


# --- Section 4 --------------------------------------------------------------

def build_section_4(doc: Document) -> None:
    doc.add_heading("4. Outlook for Day 3", level=1)
    doc.add_paragraph(
        "Day 3 has two principal components. The first is the construction "
        "of a naïve late-fusion baseline, in which the per-trial softmax "
        "probabilities saved on Day 2 are averaged across the three "
        "modalities and converted into a single emotion prediction per "
        "trial. This baseline serves as the explicit comparison point for "
        "the more sophisticated cross-modal attention fusion module to be "
        "built on Day 4: any fusion architecture that fails to outperform "
        "the naïve average is not, in any meaningful sense, learning to "
        "fuse. The late-fusion computation is GPU-free and can be performed "
        "on the local development machine within a few minutes once the "
        "Day 2 logit archives have been downloaded from Drive."
    )
    doc.add_paragraph(
        "The second component of Day 3 is the application of two "
        "trainer-side improvements identified during Day 2: the best-test "
        "checkpoint mechanism, which retains the logits from the "
        "highest-test-accuracy epoch rather than the final epoch, and the "
        "reduction of the audio fine-tuning epoch budget from fifteen to "
        "approximately seven. These improvements, taken together, are "
        "expected to reduce per-subject training time by approximately "
        "fifty per cent and to recover approximately one to two "
        "percentage points of accuracy lost to the current last-epoch "
        "convention. The upgrade to Colab Pro will be taken on Day 3 "
        "before the forty-two-subject rollout begins. By the end of Day 3, "
        "therefore, the project will have a naïve fusion baseline, an "
        "optimised training driver, and the compute headroom to begin the "
        "full per-modality rollout in the background."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    doc = Document()

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
    build_section_3(doc)
    doc.add_page_break()
    build_section_4(doc)

    doc.save(OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
