"""Generate poster_presentation.docx — a 7-minute three-speaker script for
the poster defence, split across the three columns of thesis_poster.pptx.

  Column 1   (Foundation)         — Shafin Islam       — ~2 min 00 s
  Column 2   (Model + Results)    — Abir Dey           — ~2 min 30 s
  Column 3   (Analysis + Future)  — Md. Mushroor M. K. — ~2 min 30 s
                                       + buffer        +  0 min 30 s
                                            TOTAL      —  7 min 30 s

Each section has:
  - A header with name, columns owned, and timing budget.
  - The script as labelled beats with target seconds.
  - Italic [CUE] lines for transitions to the next speaker.
  - Plain-language phrasing throughout (moderate ML knowledge audience).
"""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT = "poster_presentation.docx"

NAVY  = RGBColor(0x1F, 0x3A, 0x68)
ORANGE = RGBColor(0xE6, 0x7E, 0x22)
GREY  = RGBColor(0x55, 0x55, 0x55)
RED   = RGBColor(0xB0, 0x1F, 0x1F)
GREEN = RGBColor(0x2E, 0x7D, 0x32)


def add_header(doc, title, subtitle=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(title)
    r.font.size = Pt(18)
    r.font.bold = True
    r.font.color.rgb = NAVY
    if subtitle:
        p2 = doc.add_paragraph()
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r2 = p2.add_run(subtitle)
        r2.font.size = Pt(11)
        r2.font.color.rgb = GREY
        r2.italic = True


def add_speaker_block(doc, speaker, role, time_str):
    p = doc.add_paragraph()
    r = p.add_run(speaker)
    r.font.size = Pt(15)
    r.font.bold = True
    r.font.color.rgb = NAVY

    p2 = doc.add_paragraph()
    r2 = p2.add_run(role)
    r2.font.size = Pt(11)
    r2.italic = True
    r2.font.color.rgb = ORANGE

    p3 = doc.add_paragraph()
    r3 = p3.add_run(f"Speaking time budget: {time_str}")
    r3.font.size = Pt(10)
    r3.font.color.rgb = GREY


def add_beat(doc, beat_title, seconds):
    p = doc.add_paragraph()
    r = p.add_run(f"▶  {beat_title}  ")
    r.font.size = Pt(11)
    r.font.bold = True
    r.font.color.rgb = NAVY
    r2 = p.add_run(f"(~{seconds}s)")
    r2.font.size = Pt(9.5)
    r2.italic = True
    r2.font.color.rgb = GREY


def add_speech(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.4)
    r = p.add_run(text)
    r.font.size = Pt(11)
    r.font.color.rgb = RGBColor(0x20, 0x20, 0x20)


def add_cue(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.4)
    r = p.add_run(f"[CUE — {text}]")
    r.font.size = Pt(10)
    r.italic = True
    r.bold = True
    r.font.color.rgb = RED


def add_tip(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(0.4)
    r = p.add_run(f"Delivery hint: {text}")
    r.font.size = Pt(9.5)
    r.italic = True
    r.font.color.rgb = GREEN


def add_divider(doc):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("──────────────")
    r.font.color.rgb = GREY


def build():
    doc = Document()

    for s in doc.sections:
        s.top_margin = Cm(1.8)
        s.bottom_margin = Cm(1.8)
        s.left_margin = Cm(2.2)
        s.right_margin = Cm(2.2)

    add_header(doc,
               "Pre-thesis 2 Poster Defence — 7-Minute Three-Speaker Script",
               "Cross-Modal Affective Coherence  ·  Shafin, Abir, "
               "Mushroor  ·  June 2026")

    # ---- Global delivery tips ----
    p = doc.add_paragraph()
    r = p.add_run("Before you start")
    r.bold = True
    r.font.size = Pt(13)
    r.font.color.rgb = NAVY

    tips = [
        "Stand on the side of YOUR column. Point to specific blocks while "
        "you speak.",
        "When you hand over, take a half-step back and the next speaker "
        "steps forward to their column.",
        "Total speaking time is ~6 min 30 s including transitions, leaving "
        "30 s of natural pauses inside the 7-minute slot.",
        "Practice once together with a phone stopwatch. If you over-run, "
        "trim the labelled OPTIONAL beats first.",
        "Q&A is separate — do not dip into it during the 7-minute slot.",
    ]
    for t in tips:
        bp = doc.add_paragraph(style="List Bullet")
        br = bp.add_run(t)
        br.font.size = Pt(10.5)

    doc.add_page_break()

    # =====================================================================
    # SPEAKER 1 — Shafin Islam (Column 1: Foundation)
    # =====================================================================
    add_speaker_block(
        doc, "Speaker 1   ·   Shafin Islam",
        "Column 1  —  Abstract  /  Research Objectives  /  Methodology  "
        "/  Data Preprocessing  /  Dataset Specs",
        "~2 minutes 00 seconds")
    add_divider(doc)

    add_beat(doc, "Title + Welcome", 15)
    add_speech(doc,
               "Good morning. Our project is called Cross-Modal Affective "
               "Coherence. The short version: we built a system that "
               "doesn't just classify what emotion a person is showing — "
               "it also detects when the emotion they're showing on the "
               "outside doesn't match what's happening inside their brain.")

    add_beat(doc, "Problem & Motivation  (point at ABSTRACT block)", 25)
    add_speech(doc,
               "Most emotion-recognition systems treat audio, video, and "
               "EEG as three noisy views of the same thing. But that's not "
               "quite right. Audio and video — voice, facial expression — "
               "are voluntary. People can fake those. EEG measures brain "
               "activity directly, which is much harder to fake. So when "
               "the outside and the inside disagree, that disagreement is "
               "actually a clinically meaningful signal. It is documented "
               "in depression, alexithymia, and PTSD. Our thesis builds "
               "the computational framework to detect it.")

    add_beat(doc, "Research Objectives  (point at OBJECTIVES block)", 25)
    add_speech(doc,
               "We had five research questions. Can each modality "
               "individually beat the dataset's own published baselines? "
               "Does fusing them help? Does a smart attention-based fusion "
               "beat naive averaging? When the modalities disagree, are "
               "the patterns of disagreement consistent across people? "
               "And finally — does the system still work in a realistic "
               "setting where we don't have an EEG headset?")

    add_beat(doc, "Methodology  (point at METHODOLOGY block)", 25)
    add_speech(doc,
               "Three sequential stages. First, we train per-modality "
               "encoders: AST for audio, ViT for video, EEGNet for EEG. "
               "Second, we fuse them — either with cross-attention or "
               "with a simpler concat-MLP, plus an important training "
               "trick called modality dropout. Third, we run a coherence "
               "analysis on the per-modality outputs to find disagreements.")

    add_beat(doc, "Dataset  (point at DATASET SPECS block)", 25)
    add_speech(doc,
               "Our data comes from the EAV dataset, published in "
               "Scientific Data 2024. Forty-two participants performing a "
               "cue-based conversation eliciting five emotions: Neutral, "
               "Anger, Happiness, Sadness, Calmness. Synchronised EEG, "
               "audio, and video. As far as we know, no fusion method has "
               "been evaluated on EAV before we started.")

    add_cue(doc,
            "Half-step left. Abir steps to Column 2 and begins on cue.")
    add_speech(doc,
               "With the problem and methodology set up, Abir will now "
               "walk you through the model architecture and the headline "
               "results.")

    add_tip(doc,
            "Section 1 is dense — speak slowly, slightly louder than "
            "normal. The audience is forming first impressions here.")

    doc.add_page_break()

    # =====================================================================
    # SPEAKER 2 — Abir Dey (Column 2: Model + Results)
    # =====================================================================
    add_speaker_block(
        doc, "Speaker 2   ·   Abir Dey",
        "Column 2  —  Proposed Model Architecture  /  Training Results  "
        "/  Prior Fusion Methods on EAV",
        "~2 minutes 30 seconds")
    add_divider(doc)

    add_beat(doc, "Architecture overview  (point at ARCHITECTURE block)", 30)
    add_speech(doc,
               "This is our trimodal fusion model. Audio, video, and EEG "
               "each go through their own encoder. The two transformer "
               "encoders, AST and ViT, are pre-trained on giant audio and "
               "image datasets respectively. EEGNet is a small CNN trained "
               "from scratch on EEG, because there is no large pre-trained "
               "model for brain signals at this scale.")

    add_beat(doc, "Fusion mechanism  (point at TrimodalAttentionFusion)", 30)
    add_speech(doc,
               "Each encoder produces a feature vector. We project all "
               "three to the same 256-dimensional space and treat them as "
               "three tokens. A two-layer transformer with self-attention "
               "lets the modalities look at each other and decide how much "
               "weight to give each one. Then we mean-pool and pass through "
               "a small MLP head to get the five-class prediction. The "
               "whole fusion module is 2.28 million parameters.")

    add_beat(doc, "Modality dropout — the key training trick", 25)
    add_speech(doc,
               "Crucially, during training we randomly zero out one of "
               "the three modalities at each step. This is called softhard "
               "modality dropout. Two reasons. First, it makes the model "
               "robust to missing modalities at inference — important "
               "because in any real deployment we won't have an EEG "
               "headset. Second, it acts as a regulariser, forcing each "
               "modality to carry self-sufficient evidence.")

    add_beat(doc, "Training results  (point at TRAINING RESULTS chart)", 35)
    add_speech(doc,
               "Looking at the bar chart: vision alone reaches about 75 "
               "percent. Audio about 57. EEG, the hardest modality, about "
               "44. When we fuse — any kind of learned fusion — accuracy "
               "jumps significantly: naive averaging to 77.5, cross-"
               "attention to 80.2, and concat-MLP to 81.7. Then, with "
               "modality dropout layered on top, we reach 84.7 percent. "
               "And the demo path — where we run inference with the EEG "
               "input zeroed out — still gives us 81.5 percent. All seven "
               "of these comparisons are statistically significant by "
               "paired Wilcoxon tests.")

    add_beat(doc, "Comparison vs prior EAV work  (point at PRIOR FUSION chart)", 30)
    add_speech(doc,
               "On the right, this is how we compare to prior trimodal "
               "fusion work on EAV. AMERL got 70.86 percent. Hyper-MML, "
               "the previous state of the art, got 76.65 percent. EEG-MoCE "
               "got 75.88. Our cross-attention alone reaches 80.2 percent. "
               "With modality dropout, 84.7. That is 8.05 percentage "
               "points above the previous state of the art on the same "
               "per-subject evaluation protocol.")

    add_cue(doc, "Step half-left. Mushroor steps to Column 3.")
    add_speech(doc,
               "So we beat the previous state of the art by over eight "
               "points. But classification accuracy is only one half of "
               "our contribution. Mushroor will now walk you through the "
               "analysis side — including the suppression matrix, which "
               "is the novel scientific contribution of the thesis.")

    add_tip(doc,
            "Section 2 has the most numbers. Pause briefly after each "
            "key percentage to let it land. Do NOT rush through the "
            "84.7 / 81.5 / 8.05 sequence — those are the headline numbers.")

    doc.add_page_break()

    # =====================================================================
    # SPEAKER 3 — Md. Mushroor Muttakin Khan (Column 3)
    # =====================================================================
    add_speaker_block(
        doc, "Speaker 3   ·   Md. Mushroor Muttakin Khan",
        "Column 3  —  Per-Class Metrics  /  Suppression Matrix  /  "
        "Research Outcomes  /  Future Work",
        "~2 minutes 30 seconds")
    add_divider(doc)

    add_beat(doc, "Per-class performance  (point at CLASSWISE chart)", 20)
    add_speech(doc,
               "Quickly, on per-class performance: macro-F1 is 0.846. "
               "Happiness is the easiest to classify, almost 90 F1. "
               "Calmness is the hardest — it gets confused with Neutral "
               "about 14 percent of the time, because both are low-arousal "
               "states.")

    add_beat(doc, "Suppression matrix — the novel contribution", 50)
    add_speech(doc,
               "Now for the most novel part. The suppression matrix. We "
               "look at every test trial and apply three filters: audio "
               "and video must agree on an emotion; EEG must disagree; "
               "and EEG must be confident, with at least 50 percent "
               "softmax probability on its prediction. When all three "
               "filters pass, we count it as a suppression event — an "
               "event where the externally displayed emotion does not "
               "match what the brain is indicating internally. Across "
               "five thousand and forty test trials, we find 401 such "
               "events — eight percent. The dominant pattern is "
               "bidirectional Anger and Happiness confusion. These are "
               "both high-arousal states, and the EEG often can't tell "
               "them apart by valence. Sadness and Calmness, both low-"
               "arousal states, tend to look like Neutral at the EEG "
               "level. All of these patterns are population-wide, not "
               "driven by one or two outlier subjects.")

    add_beat(doc, "Research outcomes  (point at OUTCOMES block)", 25)
    add_speech(doc,
               "So to summarise our outcomes: we set a new state of the "
               "art on EAV. We show that modality dropout is the dominant "
               "training gain and works regardless of which fusion you "
               "use. We demonstrate that the audio-visual demo path "
               "actually exceeds the original full-modality model. And "
               "we provide the first cross-modal coherence analysis on "
               "EAV.")

    add_beat(doc, "Future work  (point at FUTURE WORK block)", 25)
    add_speech(doc,
               "For Phase 2, four directions. Clinical validation on "
               "alexithymia or depression populations. A primary audio-"
               "visual dataset that we collect ourselves and merge with "
               "EAV by two-stage fine-tuning. In-the-wild temporal "
               "tracking on DFEW or MAFW. And MERCL contrastive pre-"
               "training to improve cross-modal alignment.")

    add_beat(doc, "Closing", 10)
    add_speech(doc,
               "And that is our thesis. Thank you for your attention — "
               "we welcome your questions.")

    add_tip(doc,
            "The suppression matrix beat is the longest in the whole "
            "presentation. Slow down on it. This is the novel scientific "
            "contribution — let it land. Make eye contact while explaining "
            "the bidirectional Anger-Happiness pattern.")

    # ---- Final delivery cheatsheet ----
    doc.add_page_break()

    add_header(doc, "If You Run Out Of Time", "Cut these beats first")

    cuts = [
        ("Speaker 1", "Drop the 'cue-based conversation' detail from the "
                      "Dataset beat. The audience just needs to hear "
                      "'42 subjects, 5 emotions'."),
        ("Speaker 2", "Drop the 'fusion module is 2.28 million parameters' "
                      "line. The exact count doesn't matter for a 7-minute "
                      "talk."),
        ("Speaker 3", "Drop the Future Work beat. Go straight from Research "
                      "Outcomes to Closing. Phase 2 directions live on "
                      "the poster — readers can find them there."),
    ]
    for who, what in cuts:
        p = doc.add_paragraph()
        r1 = p.add_run(f"{who}:  ")
        r1.bold = True
        r1.font.color.rgb = NAVY
        r1.font.size = Pt(11)
        r2 = p.add_run(what)
        r2.font.size = Pt(11)

    add_header(doc, "If You Have Extra Time",
               "Use these beats only if you're ahead of schedule")

    extras = [
        ("Speaker 2",
         "Mention that the concat-MLP at 81.7% slightly beat our cross-"
         "attention at 80.2%. We keep cross-attention because of two "
         "structural properties: it accommodates modality dropout cleanly "
         "and exposes per-modality embeddings for future temporal work."),
        ("Speaker 3",
         "If asked about clinical generalisation: the suppression matrix "
         "is computed from independently-trained per-modality classifiers, "
         "so it doesn't depend on which fusion architecture we used. "
         "That decoupling is what makes the methodology portable to "
         "future studies."),
    ]
    for who, what in extras:
        p = doc.add_paragraph()
        r1 = p.add_run(f"{who}:  ")
        r1.bold = True
        r1.font.color.rgb = ORANGE
        r1.font.size = Pt(11)
        r2 = p.add_run(what)
        r2.font.size = Pt(11)

    doc.save(OUT)
    print(f"Wrote {OUT}")
    print(f"Paragraphs: {len(doc.paragraphs)}")


if __name__ == "__main__":
    build()
