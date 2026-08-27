"""Generate thesis_speech.docx — a slide-by-slide spoken script for a
~10-minute Pre-thesis 2 presentation, written in plain language.
"""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT = "thesis_speech.docx"

NAVY = RGBColor(0x1F, 0x3A, 0x68)
GOLD = RGBColor(0xC8, 0x96, 0x1B)
GREY = RGBColor(0x55, 0x55, 0x55)
RED = RGBColor(0xB0, 0x1F, 0x1F)


def add_section(doc, slide_num: int, title: str, time_est: str):
    p = doc.add_paragraph()
    r = p.add_run(f"Slide {slide_num}  —  {title}")
    r.bold = True
    r.font.size = Pt(14)
    r.font.color.rgb = NAVY

    p2 = doc.add_paragraph()
    r2 = p2.add_run(f"[ approx {time_est} ]")
    r2.italic = True
    r2.font.size = Pt(10)
    r2.font.color.rgb = GREY


def add_speech(doc, text: str):
    for para in text.strip().split("\n\n"):
        p = doc.add_paragraph()
        run = p.add_run(para.strip())
        run.font.size = Pt(12)
        p.paragraph_format.space_after = Pt(8)
    # spacer between sections
    doc.add_paragraph()


def add_cue(doc, text: str):
    p = doc.add_paragraph()
    r = p.add_run(f"[CUE — {text}]")
    r.bold = True
    r.italic = True
    r.font.size = Pt(10)
    r.font.color.rgb = RED


def build():
    doc = Document()

    for section in doc.sections:
        section.top_margin = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin = Cm(2.2)
        section.right_margin = Cm(2.2)

    # Title
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("Pre-thesis 2 — Spoken Script")
    r.bold = True
    r.font.size = Pt(20)
    r.font.color.rgb = NAVY

    st = doc.add_paragraph()
    st.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sr = st.add_run("Cross-Modal Affective Coherence Using EEG, Audio and Video on EAV")
    sr.italic = True
    sr.font.size = Pt(12)
    sr.font.color.rgb = GREY

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    mr = meta.add_run("13 slides  ·  approx 10 minutes  ·  spoken in plain language")
    mr.font.size = Pt(10)
    mr.font.color.rgb = GREY

    doc.add_paragraph()

    # Delivery tips
    p = doc.add_paragraph()
    r = p.add_run("Delivery tips")
    r.bold = True
    r.font.size = Pt(12)
    r.font.color.rgb = NAVY

    tips = [
        "Read once aloud and time yourself. Aim for 10–11 minutes including the demo video.",
        "Slow down on slides 8, 9 and 10 — these are the contributions. Speed up on slides 3, 4 and 11.",
        "When you reach the demo (slide 12), stop talking, play the video, then resume.",
        "If you go over time, drop the dataset details on slide 4 and the per-class details on slide 11.",
        "Sentences are short on purpose. Do not over-rehearse, or it will sound robotic.",
    ]
    for tip in tips:
        bp = doc.add_paragraph(style="List Bullet")
        br = bp.add_run(tip)
        br.font.size = Pt(11)

    doc.add_page_break()

    # =====================================================================
    # Slide 1 — Title
    # =====================================================================
    add_section(doc, 1, "Title", "30 seconds")
    add_speech(doc, """
Good [morning / afternoon]. Thank you for being here.

My thesis is called "Cross-Modal Affective Coherence". In plain language, that
means I am trying to figure out when the emotion that a person is showing on
the outside does not match what is happening on the inside.

To do this, I use three things together: a video of the person's face, a
recording of their voice, and a measurement of their brain activity called
EEG. Over the next ten minutes I will explain why this problem matters, how I
built the system, what it can do, and how it compares with other published work.
""")

    # =====================================================================
    # Slide 2 — Problem & Motivation
    # =====================================================================
    add_section(doc, 2, "Problem & Motivation", "1 minute")
    add_speech(doc, """
Most emotion recognition systems answer one question: "what emotion is this
person showing?" That is a useful question, but it misses something
important. People can hide what they are really feeling. Somebody can smile
and look calm on the outside while feeling angry or sad on the inside.

So I want to answer a second question as well: "does the emotion the person is
showing actually match what they are feeling internally?" This is what I call
coherence — the agreement between the outside signal and the inside signal.

Why is this useful? The face and the voice are things people can control on
purpose. EEG, which measures electrical activity from the brain, is much
harder to fake. So if the face says one thing and the brain says another, the
disagreement itself is a signal worth studying. This kind of mismatch is
already known to be related to conditions like depression, anxiety, and PTSD.
My thesis builds the foundation for that — first on healthy subjects, in a
controlled lab setting.
""")

    # =====================================================================
    # Slide 3 — Research Questions
    # =====================================================================
    add_section(doc, 3, "Research Questions", "30 seconds")
    add_speech(doc, """
I had five concrete questions. First, can my individual audio, video and EEG
models perform better than the baselines the dataset authors published?
Second, does combining the three modalities together — what we call fusion —
beat the best single modality? Third, does a smart attention-based fusion
actually do better than a simple average of the three predictions? Fourth,
when the modalities disagree, are there patterns to the disagreement, or is
it just noise? And fifth, does the system still work if one modality, for
example the EEG, is missing at the time we use it in the real world?
""")

    # =====================================================================
    # Slide 4 — Dataset
    # =====================================================================
    add_section(doc, 4, "Dataset — EAV", "45 seconds")
    add_speech(doc, """
The dataset I use is called EAV. It was released in September 2024 by Lee and
colleagues at Nazarbayev University, and published in the journal Scientific
Data. It is, as far as I know, the only public dataset that combines all three
modalities — EEG, audio, and video — recorded from the same people at the same
time, for emotion recognition.

There are 42 participants. Each one took part in 200 short conversational
interactions, where they were prompted to express one of five emotions:
Neutral, Anger, Happiness, Sadness, or Calmness. So in total we have around
8,400 trials. The EEG was recorded from 30 channels. Audio at 16 kilohertz.
Video at 30 frames per second. The training and testing split is 280 trials
per subject for training and 120 for testing — done per person, not pooled.

The important thing for my thesis: when the EAV authors released the dataset,
they only published per-modality baselines. They did not provide a fusion model.
That is the gap I fill.
""")

    # =====================================================================
    # Slide 5 — Methodology
    # =====================================================================
    add_section(doc, 5, "Methodology — Three-Stage Pipeline", "1 minute")
    add_speech(doc, """
My pipeline has three stages.

Stage one: I train a separate model for each modality. For audio, I use a
model called AST, the Audio Spectrogram Transformer. For video, I use ViT, the
Vision Transformer, with a face crop on each frame. For EEG, I use EEGNet, a
small convolutional network designed specifically for brain signals. Each of
these models takes one modality and produces a compact summary vector that
represents what that signal looks like for that trial.

Stage two: I take the three summary vectors and feed them into a fusion module
based on attention. Attention is the same mechanism that powers large language
models — it lets each modality look at the others and decide how much to weigh
them.

Stage three: I do a coherence analysis. Once the system has predicted what
each modality says on its own, I count the trials where they disagree, and I
look for systematic patterns. That gives me a 5-by-5 table I call the
suppression matrix.

Two design rules I committed to from day one: the fusion model has to expose
the per-modality features (so I can analyse coherence) and it has to accept a
zero EEG input (so the demo works without an EEG sensor).
""")

    # =====================================================================
    # Slide 6 — Architecture
    # =====================================================================
    add_section(doc, 6, "Proposed Architecture", "1 minute")
    add_speech(doc, """
This is the architecture. Each modality produces one feature vector — audio
gives 768 numbers, video gives 768 numbers, EEG gives 960 numbers. I project
all three down to 256 numbers, so they live in the same space. Then I stack
them as a sequence of three tokens — one per modality — and pass them through
a small transformer with two layers and eight attention heads.

You might ask: why self-attention over three tokens, rather than the more
common pairwise cross-attention used in other multimodal papers? The answer
is practical. My per-modality encoders give me one summary vector per
modality, not a long sequence. When you only have one token per modality,
pairwise cross-attention degenerates into a simple linear projection — it
doesn't really do anything interesting. Self-attention over the three-token
sequence does the same job with fewer parameters and cleaner code. The whole
fusion module has only 2.28 million parameters, which is small.

The output of the transformer is a single fused vector that goes into a
small classifier head, which predicts the emotion.
""")

    # =====================================================================
    # Slide 7 — Training Results
    # =====================================================================
    add_section(doc, 7, "Training Results", "1 minute")
    add_speech(doc, """
Now the results. This is averaged over all 42 subjects.

Individually: audio reaches 57 percent, video reaches 75 percent, and EEG
reaches about 44 percent. All three of these are significantly higher than
the baselines the EAV paper reported — for example, video is up 22 percentage
points from their 53 percent number.

When I combine the three with naive late fusion — just averaging their
softmax outputs — I get 80 percent. When I use my cross-attention fusion, I
also get 80 percent. So these two are basically tied. That is an honest finding,
not a weakness. It says that on this dataset, when modalities are summarised
as single feature vectors, the choice of fusion operator doesn't matter very
much.

The interesting result is the next row: when I add a training technique called
softhard modality dropout — which I will explain on the next slide but one —
the fusion accuracy jumps to 84.7 percent. That is a 4.5 percentage point
gain, just from how I trained the model, not from changing the architecture.

The last row, 81.5 percent, is the demo path — running the system with the
EEG set to zero. We will come back to it.
""")

    # =====================================================================
    # Slide 8 — Comparison with Prior EAV Work
    # =====================================================================
    add_section(doc, 8, "Comparison with Prior EAV Work", "1 minute")
    add_speech(doc, """
This is the comparison table.

EAV is a new dataset, so the literature is still small. To my knowledge there
are three published trimodal fusion methods on it. The first is AMERL, from
Yin and colleagues in 2024 — they used tailored transformers with
self-attention fusion, and they reported 70.86 percent. The second is
Hyper-MML, from Kang and colleagues in 2025 — they used a hypergraph fusion
module and reported 76.65 percent, which is the previously published state
of the art. The third is EEG-MoCE, from 2026, which uses something called
hyperbolic geometry experts and reports 75.88 percent.

My cross-attention fusion alone reaches 80.2 percent — that is 3.5 percentage
points above the previous state of the art. My modality-dropout variant
reaches 84.7 percent — that is 8.1 percentage points above the previous state
of the art. So if the evaluation protocols line up — and I have checked that
they do for the open-source comparison, AMERL — these numbers represent the
new best published result on EAV.

Just as important, none of these three prior methods test missing-modality
robustness. None of them include a working demo. And none of them perform
any coherence analysis. So the comparison is not only about the headline
number, it is about three contributions that are unique to this work.
""")

    # =====================================================================
    # Slide 9 — Modality Dropout & Robustness
    # =====================================================================
    add_section(doc, 9, "Modality Dropout & Robustness", "1 minute")
    add_speech(doc, """
Now let me explain modality dropout. The idea is simple. During training, at
every batch, I randomly pick one of the three modalities and zero it out
before it goes into the fusion. So sometimes the model sees all three
modalities, and sometimes it has to make a prediction with only two. This
teaches the model that any modality might be missing at any time, and it
should still produce a sensible answer.

There are two reasons this is useful. The first is the engineering reason —
in a real-world deployment, EEG is hard to record. You need electrodes on the
scalp. Audio and video are easy. So the system has to work without EEG. The
second reason is more interesting: when I force the model to handle missing
modalities, it learns better representations across the board. It cannot rely
on the strongest modality and ignore the others. It has to make every modality
pull its own weight.

The table on this slide confirms both effects. Full modality is 84.7 percent.
Without EEG — which is the demo path — we get 81.5 percent. That is only 3
points down. Without audio is similar, at 80 percent. Without video drops to
54 percent, because video is the dominant modality on this dataset.

The line at the bottom of the slide is the headline. The system running with
zero EEG — which is what the demo does — is more accurate than the original
full-modality system trained without dropout. That is unusual, and it is
worth highlighting.
""")

    # =====================================================================
    # Slide 10 — Coherence Analysis (Suppression Matrix)
    # =====================================================================
    add_section(doc, 10, "Cross-Modal Coherence Analysis", "1 minute 30 seconds")
    add_speech(doc, """
This is the most novel part of my thesis.

I define a "suppression event" as a trial where the audio and video agree on
the same emotion, but the EEG disagrees with confidence. The intuition is that
audio and video together form what we see and hear from outside — the
person's external expression. EEG comes from inside the head — it is more
involuntary. So when the outside and the inside disagree, that is the kind
of disagreement worth studying.

Out of 5,040 test trials across 42 subjects, I find 401 suppression events,
which is a base rate of 8 percent. So this is not rare, but it is also not
the default — most of the time the modalities agree.

The 5-by-5 table on this slide is the suppression matrix. The rows are what
the outside says, the columns are what the EEG says. The dominant patterns
are clear. First, anger and happiness suppress each other in both directions —
65 events of anger-outside-but-happiness-inside, and 52 events of the
reverse. That is 117 events, which is a lot. These are the two high-energy
emotions, and the brain seems to confuse them often. Second, sadness and
calmness both show up as Neutral on the EEG side. That is consistent with
the idea that low-energy emotions look like a kind of baseline at the
brain level.

I checked one important thing. Earlier in the project, on just three subjects,
I worried that the EEG model was biased toward predicting Neutral too often.
At full 42-subject scale, that bias disappeared completely. So the matrix can
be read at face value.

What this means for the thesis: I now have a quantitative method to describe
cross-modal divergence, on a public dataset, with no prior baseline to compare
against. That is a genuine contribution.
""")

    # =====================================================================
    # Slide 11 — Per-Class Performance
    # =====================================================================
    add_section(doc, 11, "Per-Class Performance & Confusion Patterns", "45 seconds")
    add_speech(doc, """
This slide is a sanity check. I want to confirm that the patterns I see in
the coherence matrix also show up in standard classification metrics.

Per-emotion F1 scores range from 80 to 90 percent. Happiness is the easiest
to classify at almost 90. Calmness is the hardest at 81.

The confusion matrix on the right shows which classes the model mixes up. The
biggest confusion is between Neutral and Calmness — 14 percent of Neutral
samples get predicted as Calmness. And on the coherence side, the Calmness-
to-Neutral suppression count was 40 events. The same two emotions are
confusable in both analyses. This is what cross-validates the coherence
findings — the classification confusion and the coherence disagreement
point at the same blind spot in the model.
""")

    # =====================================================================
    # Slide 12 — Live Demonstration
    # =====================================================================
    add_section(doc, 12, "Live Demonstration", "1 minute 30 seconds")
    add_speech(doc, """
For the live demonstration, I recorded a short video of myself expressing
different emotions, and ran it through the deployed system. The system uses
only the audio and video — no EEG was recorded, the EEG input is just a
vector of zeros. This is possible because the model was trained with
modality dropout.

Let me play the video now.
""")
    add_cue(doc, "PLAY thesis_demo_overlay.mp4. Wait for it to finish before continuing.")
    add_speech(doc, """
What you saw: the system slides a 5-second window across the recording and
gives a prediction every second. The emotion label appears at the top, with
a confidence bar. The model is running on a laptop, in real time, with no
brain sensor required. The reported accuracy of this demo path on the held-
out EAV test set is 81.5 percent.
""")

    # =====================================================================
    # Slide 13 — Future Work & Conclusion
    # =====================================================================
    add_section(doc, 13, "Future Work & Conclusion", "45 seconds")
    add_speech(doc, """
To wrap up.

My contributions are: first, the first trimodal fusion model on the EAV
dataset, which is also a new state of the art on this benchmark. Second, a
modality-dropout training scheme that lets the system run without EEG —
which is what makes real-world deployment possible. And third, a coherence
analysis framework that measures, quantitatively, when the outside and the
inside of a person's expression disagree.

Future work has three directions. The first is clinical validation — testing
the coherence analysis on labelled depression or alexithymia data. The
second is in-the-wild deployment — using a temporal head over the per-
modality features to handle longer videos. And the third is to add the
MERCL contrastive pre-training step from Lee, Kim and Kim, which we
deliberately left out due to time constraints.

That is everything. Thank you for your attention. I am happy to take questions.
""")

    doc.save(OUT)
    print(f"Wrote {OUT}")
    print(f"Paragraphs: {len(doc.paragraphs)}")


if __name__ == "__main__":
    build()
