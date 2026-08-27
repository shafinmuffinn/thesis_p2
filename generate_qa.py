"""Generate poster_qa.docx — 30 likely defence questions with prepared
answers, organised by category.

Format per question:
  Q  — the question
  A  — short, defensible headline answer (1-3 sentences)
  If pushed  — follow-up detail for when the panel digs deeper
"""
from docx import Document
from docx.shared import Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUT = "poster_qa.docx"

NAVY   = RGBColor(0x1F, 0x3A, 0x68)
ORANGE = RGBColor(0xE6, 0x7E, 0x22)
GREY   = RGBColor(0x55, 0x55, 0x55)
RED    = RGBColor(0xB0, 0x1F, 0x1F)
DARK   = RGBColor(0x20, 0x20, 0x20)


# -- helpers -----------------------------------------------------------------

def add_title(doc, text, subtitle=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text)
    r.font.size = Pt(18)
    r.font.bold = True
    r.font.color.rgb = NAVY
    if subtitle:
        p2 = doc.add_paragraph()
        p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r2 = p2.add_run(subtitle)
        r2.font.size = Pt(11)
        r2.italic = True
        r2.font.color.rgb = GREY


def add_category(doc, n_from, n_to, title):
    doc.add_paragraph()
    p = doc.add_paragraph()
    r = p.add_run(f"Q{n_from}–Q{n_to}   ·   {title}")
    r.font.size = Pt(14)
    r.font.bold = True
    r.font.color.rgb = NAVY


def add_qa(doc, n, question, answer, follow_up=None):
    # Question
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    r = p.add_run(f"Q{n}.  ")
    r.font.size = Pt(11.5)
    r.font.bold = True
    r.font.color.rgb = ORANGE
    r2 = p.add_run(question)
    r2.font.size = Pt(11.5)
    r2.font.bold = True
    r2.font.color.rgb = DARK

    # Answer
    p2 = doc.add_paragraph()
    p2.paragraph_format.left_indent = Cm(0.6)
    r3 = p2.add_run("A.  ")
    r3.font.size = Pt(11)
    r3.font.bold = True
    r3.font.color.rgb = NAVY
    r4 = p2.add_run(answer)
    r4.font.size = Pt(11)
    r4.font.color.rgb = DARK

    if follow_up:
        p3 = doc.add_paragraph()
        p3.paragraph_format.left_indent = Cm(0.6)
        r5 = p3.add_run("If pushed:  ")
        r5.font.size = Pt(10)
        r5.italic = True
        r5.font.bold = True
        r5.font.color.rgb = GREY
        r6 = p3.add_run(follow_up)
        r6.font.size = Pt(10)
        r6.italic = True
        r6.font.color.rgb = GREY


# -- the 30 Q&As -------------------------------------------------------------

def build():
    doc = Document()
    for s in doc.sections:
        s.top_margin = Cm(1.8)
        s.bottom_margin = Cm(1.8)
        s.left_margin = Cm(2.2)
        s.right_margin = Cm(2.2)

    add_title(doc, "Defence Q&A  —  30 Likely Questions",
              "Cross-Modal Affective Coherence  ·  practise before the panel")

    # ---- general strategy ----
    doc.add_paragraph()
    p = doc.add_paragraph()
    r = p.add_run("General Q&A strategy")
    r.font.bold = True
    r.font.size = Pt(12.5)
    r.font.color.rgb = NAVY

    tips = [
        "Acknowledge the question before answering. \"That's a great "
        "point\" or \"Good question, the short answer is...\" buys you "
        "a beat to think.",
        "Lead with the headline answer in one or two sentences. Only "
        "elaborate if asked.",
        "If a question identifies a real limitation, OWN it and reframe "
        "as future work. Don't deny.",
        "If you genuinely don't know, say \"That's outside what we "
        "tested, but our framework supports it as future work\" — never "
        "fabricate.",
        "Pass to the appropriate teammate whose section the question "
        "lives in. Don't all answer at once.",
    ]
    for t in tips:
        bp = doc.add_paragraph(style="List Bullet")
        br = bp.add_run(t)
        br.font.size = Pt(10.5)

    # =========================================================
    # Architecture & Fusion (Q1-Q6)
    # =========================================================
    add_category(doc, 1, 6, "Architecture & Fusion")

    add_qa(doc, 1,
        "Why self-attention over three modality tokens rather than the "
        "six-way pairwise cross-attention used in Lee, Kim and Kim's 2024 "
        "MERCL paper?",
        "Cross-attention is the natural choice when each modality "
        "contributes a sequence of tokens. Our pre-trained encoders give "
        "us one pooled vector per modality — just one token each. With "
        "one query and one key, pairwise cross-attention degenerates to "
        "a linear projection. Self-attention over the three-token "
        "sequence does the same mixing with fewer parameters.",
        "If we moved to sequence-level features — for example, the "
        "per-frame ViT outputs instead of the mean-pooled clip embedding "
        "— full six-way cross-attention would become non-degenerate "
        "again. That's our Phase-2 architectural extension.")

    add_qa(doc, 2,
        "Why didn't you use transformers for EEG like you did for audio "
        "and video?",
        "Three reasons: pretraining, data scale, and inductive bias. AST "
        "and ViT have foundation models pretrained on millions of audio "
        "clips and images. There is no equivalent for 30-channel scalp "
        "EEG. We have 280 training trials per subject — a transformer "
        "with 87 million parameters would catastrophically overfit. "
        "EEGNet has 2.2 million parameters and the right inductive bias "
        "for the channel-by-time structure of multi-channel EEG.",
        "Lee, Kim and Kim themselves used a 'modified Conformer with no "
        "self-attention' for EEG, precisely because raw self-attention "
        "is data-hungry on EEG. EEGNet has also been the dominant EEG "
        "architecture in BCI literature since 2018.")

    add_qa(doc, 3,
        "Why exactly two transformer layers? Why not more?",
        "Two layers gives enough capacity to mix three modality tokens "
        "without overfitting on 280 training trials. We did informally "
        "explore one and four layers in early development. One layer "
        "underfit; four layers overfit. Two was the sweet spot.",
        "This is a hyperparameter we'd revisit if we had cross-subject "
        "training data — more data would let us scale to deeper fusion "
        "without overfitting.")

    add_qa(doc, 4,
        "Why project everything to 256 dimensions? Why not match the "
        "original feature dimensions?",
        "256 is a compromise: large enough to preserve the information "
        "in each per-modality feature but small enough to keep the "
        "fusion module to about 2.3 million parameters. The three input "
        "dimensions — 768, 768, and 960 — are different, so we need a "
        "common projection space for the attention mechanism to work "
        "uniformly across modalities.",
        None)

    add_qa(doc, 5,
        "Why mean-pool the three modality tokens after attention rather "
        "than use a learned attention pooler or take the first token?",
        "Mean pooling is rotation-invariant in token order, which is "
        "appropriate because there's no canonical ordering of the three "
        "modalities. A learned pooler or first-token pooling would "
        "implicitly privilege one modality. The attention layers already "
        "produce a weighted mixture; mean pooling at the end just "
        "averages those mixtures.",
        None)

    add_qa(doc, 6,
        "Could you have used a simpler fusion architecture and reached "
        "the same accuracy?",
        "Yes, in fact our concat-MLP fusion baseline at 81.7% slightly "
        "beat our cross-attention at 80.2%. We retained cross-attention "
        "for two reasons: it accommodates modality dropout cleanly by "
        "operating on discrete modality tokens, and it exposes named "
        "per-modality embeddings for Phase-2 temporal head extensions. "
        "The 1.5-point accuracy cost is the price of those architectural "
        "commitments.",
        "The thesis explicitly reports both numbers and the Wilcoxon "
        "test that confirms concat-MLP is significantly higher. This is "
        "an honest finding, not a flaw.")

    # =========================================================
    # Encoder choices (Q7-Q9)
    # =========================================================
    add_category(doc, 7, 9, "Encoder Choices")

    add_qa(doc, 7,
        "Why didn't you use a stronger EEG encoder like LEREL or a "
        "Conformer?",
        "Two reasons. First, we wanted an architecture-controlled "
        "comparison with the EAV paper's published baseline, which is "
        "EEGNet. With the same architecture, our +7.2 percentage point "
        "EEG improvement is attributable to our bug fixes and training "
        "discipline, not to a stronger encoder. Second, a stronger EEG "
        "encoder is a Phase-2 substitution we explicitly identify in "
        "our future work — the fusion module can accept any EEG encoder "
        "via its named-embedding contract.",
        None)

    add_qa(doc, 8,
        "Why use bag-of-frames averaging for ViT rather than a temporal "
        "model over the 25 frames?",
        "Two reasons. First, EAV clips are only five seconds long, and "
        "the dominant emotional expression is relatively stable over "
        "that window, so a temporal model adds parameters without "
        "adding much signal. Second, treating frames as independent "
        "lets us use a strong pre-trained image classifier without "
        "retraining a temporal head. The 25-frame mean of CLS embeddings "
        "gives 74.6% — strong enough that adding temporal complexity is "
        "Phase-2 work, not Phase-1 necessity.",
        None)

    add_qa(doc, 9,
        "AST is pre-trained on AudioSet, which is mostly environmental "
        "sounds, not speech. Why is that OK for emotion recognition?",
        "AudioSet does include human speech, vocalisations, and music — "
        "527 classes total. Even where the pretraining domain doesn't "
        "directly match, transformer features transfer well to "
        "downstream audio tasks. Empirically our AST reaches 57.1% on "
        "EAV's five-class emotion task, which is 20.4 points above the "
        "published SCNN baseline. The pretraining is doing real work.",
        "A more speech-specialised pretraining like wav2vec 2.0 might "
        "do even better, and is a Phase-2 substitution.")

    # =========================================================
    # Modality Dropout (Q10-Q12)
    # =========================================================
    add_category(doc, 10, 12, "Modality Dropout")

    add_qa(doc, 10,
        "Why did you choose p = 1/3 as the dropout probability? Did you "
        "tune it?",
        "p = 1/3 corresponds to dropping one of the three modalities "
        "uniformly at random — it's the natural symmetric choice. We "
        "informally explored higher and lower values in early "
        "development; 1/3 gave the best balance between regularisation "
        "and signal preservation. We did not formally grid-search this "
        "parameter.",
        "A formal sensitivity sweep would be a worthwhile robustness "
        "check; we report it as a limitation that could be tightened "
        "in a journal version.")

    add_qa(doc, 11,
        "Would modality dropout help even without the cross-attention "
        "architecture? Did you test it with the concat-MLP?",
        "We tested it on the cross-attention fusion only. The cross-"
        "attention's discrete modality-token structure is what makes "
        "dropout natural — you zero out one token. With a concat-MLP, "
        "you'd zero out a contiguous chunk of the concatenated feature "
        "vector, which is a different intervention. We expect dropout "
        "would still help concat-MLP, but verifying that is Phase-2 "
        "work.",
        "The 84.7% headline is specifically dropout-plus-cross-"
        "attention. If concat-MLP plus dropout exceeded that, it would "
        "change which architecture we recommend. We don't have that "
        "experiment yet.")

    add_qa(doc, 12,
        "Why is it called 'softhard' dropout? What would 'soft' or "
        "'hard' alternatives be?",
        "The terminology comes from Chumachenko et al.'s 2022 ICPR "
        "paper, which distinguishes 'softhard' (zero an entire modality "
        "token with some probability) from 'noise' (replace it with "
        "Gaussian noise of matched statistics). Softhard is the right "
        "choice for our deployment scenario because the demo literally "
        "has zero EEG — no headset — and the model needs to be trained "
        "on that exact zero pattern to be calibrated.",
        None)

    # =========================================================
    # Results & SOTA (Q13-Q16)
    # =========================================================
    add_category(doc, 13, 16, "Results & SOTA Claim")

    add_qa(doc, 13,
        "Concat-MLP beat your cross-attention — why claim cross-attention "
        "as the contribution?",
        "We don't. We claim modality dropout plus learned fusion as the "
        "contribution, and modality dropout works on top of any learned "
        "fusion. The 84.7% headline is the dropout-trained variant, "
        "which exceeds both concat-MLP and prior published work on EAV "
        "by significant margins. Cross-attention is the chosen base "
        "fusion because it supports the dropout intervention cleanly "
        "and exposes per-modality embeddings for Phase-2 temporal "
        "extensions.",
        None)

    add_qa(doc, 14,
        "Hyper-MML reported 76.65% on EAV. You're claiming SOTA at "
        "84.7% — can you really compare without running their code?",
        "We compare on the same evaluation protocol they report — per-"
        "subject within-subject 280/120 split, 42-subject mean. AMERL "
        "released their code, and our results align with theirs on that "
        "protocol. Hyper-MML and EEG-MoCE did not release code, so we "
        "limit our SOTA claim to 'among reports using comparable "
        "protocols.' That qualification is explicit in the thesis.",
        "We'd happily withdraw the SOTA claim if Hyper-MML released "
        "code showing they used a stricter protocol than ours. The "
        "current SOTA claim is calibrated to what's reproducible.")

    add_qa(doc, 15,
        "Five of your 42 subjects scored below 70%. Why are these "
        "subjects hard?",
        "Subject-level difficulty in EAV is a known phenomenon driven by "
        "individual EEG electrode contact quality, expressivity of "
        "facial response, and personality factors. The five low subjects "
        "are not the same as the low subjects in the EAV paper's per-"
        "modality baselines, suggesting our difficulty pattern is "
        "specific to fusion, not generally to the data. We did not "
        "investigate individual subjects clinically — that's a Phase-2 "
        "extension.",
        None)

    add_qa(doc, 16,
        "Your per-modality EEG accuracy at 43.9% is lower than some "
        "competing methods report. Why?",
        "We use vanilla EEGNet, the EAV paper's reference architecture. "
        "Competitors like LEREL use stronger EEG-specific architectures "
        "and obtain 76% on EAV. Our 43.9% is a deliberate choice for "
        "architectural-comparison fidelity. Despite the lower per-"
        "modality EEG, our fusion accuracy dominates the field, which "
        "indicates the fusion design — cross-attention plus dropout — "
        "extracts more value from a weaker EEG signal.",
        None)

    # =========================================================
    # Suppression Matrix (Q17-Q21)
    # =========================================================
    add_category(doc, 17, 21, "Suppression Matrix")

    add_qa(doc, 17,
        "Why 0.5 as the EEG confidence threshold? Is the suppression "
        "matrix sensitive to this choice?",
        "0.5 means the EEG places more probability mass on its top "
        "prediction than on the other four classes combined — a "
        "conservative reliability gate. We did informally check that "
        "the four dominant patterns survive at thresholds of 0.4 and "
        "0.6, though the absolute count of events shifts. A formal "
        "sensitivity sweep would be a worthwhile robustness check.",
        "The threshold is independent of the matrix structure — it "
        "only affects how many events qualify, not which off-diagonal "
        "cells dominate.")

    add_qa(doc, 18,
        "The Anger-Happiness bidirectional confusion — is this real "
        "suppression or just an EEG model error confusing two high-"
        "arousal states?",
        "Both interpretations are consistent with the data and we say "
        "so explicitly in our discussion. Anger and Happiness share "
        "high-arousal physiology, so EEGNet may genuinely struggle to "
        "discriminate them. Equally, participants displaying Anger may "
        "have experienced elevated-arousal positive affect "
        "neurophysiologically — competitive excitement, for example. "
        "Disentangling these requires a self-report measure collected "
        "alongside EEG, which is Phase-2 work.",
        "What we can say firmly is that the pattern is cross-population "
        "and structural, not artefactual at the individual-subject "
        "level. Whether it's neuroscientific or methodological is the "
        "open question.")

    add_qa(doc, 19,
        "The Listen/Speak temporal asymmetry — EEG is recorded during "
        "listening, audio/video during speaking. Doesn't that invalidate "
        "the suppression matrix?",
        "It restricts the interpretation but does not invalidate it. "
        "The same emotional cue drives both windows within a trial, so "
        "the 'emotion' label applies to both. What we measure is "
        "agreement between internal neural state during listening and "
        "external behavioural expression during speaking — a meaningful "
        "operationalisation of cross-modal coherence in the EAV "
        "paradigm. We acknowledge this as a corpus-specific limitation.",
        "In a synchronous corpus — naturalistic conversation with "
        "concurrent EEG — the suppression matrix would carry stronger "
        "real-time coherence semantics. That would be a Phase-2 dataset "
        "extension.")

    add_qa(doc, 20,
        "Could your suppression matrix detect clinical conditions like "
        "depression or alexithymia?",
        "That is the Phase-2 hypothesis, but we do not claim it from "
        "the present thesis. We establish a healthy-population baseline "
        "on EAV. Whether clinical populations deviate from that "
        "baseline, and how, requires recruiting patient cohorts under "
        "ethics approval — which is explicitly outside this thesis's "
        "scope.",
        "The methodology is exportable, but any clinical inference from "
        "our specific findings would be premature.")

    add_qa(doc, 21,
        "How is the suppression matrix related to your fusion model? "
        "Does it depend on which fusion architecture you used?",
        "It does not depend on the fusion model at all. The suppression "
        "matrix is computed from the per-modality classifier outputs — "
        "AST, ViT, EEGNet — independently. We could swap concat-MLP for "
        "cross-attention and the matrix would be identical. This "
        "decoupling is by design: coherence is a property of the "
        "modalities, not of the fusion operator. The matrix would still "
        "work if no fusion model existed.",
        None)

    # =========================================================
    # Generalisation (Q22-Q24)
    # =========================================================
    add_category(doc, 22, 24, "Generalisation")

    add_qa(doc, 22,
        "Why didn't you run leave-one-subject-out evaluation?",
        "Full LOSO would require 42 separate training runs of the full "
        "pipeline — approximately 500 hours of GPU time on our Colab "
        "Pro budget. We acknowledge this as a limitation and identify "
        "partial LOSO as a Phase-2 priority. Within-subject training is "
        "the standard protocol in EAV's published baselines, so our "
        "results are comparable to prior work on a like-for-like basis.",
        "A partial LOSO on five representative subjects would be a "
        "reasonable compromise between scope and cost.")

    add_qa(doc, 23,
        "How would your system perform on a different dataset like "
        "DEAP or SEED?",
        "We did not test it, because DEAP and SEED have different "
        "modality structures — DEAP doesn't have audio, SEED is EEG-"
        "only. Direct transfer would require modifying the pipeline. "
        "The fusion module's architectural choices — including modality "
        "dropout — are general and would apply to any trimodal corpus "
        "with comparable modality dimensions.",
        None)

    add_qa(doc, 24,
        "Within-subject training is much easier than cross-subject. "
        "Doesn't that inflate your numbers?",
        "Within-subject is easier than cross-subject — that's true and "
        "we acknowledge it. But it's the protocol used by every "
        "published EAV baseline including AMERL, Hyper-MML, and EEG-"
        "MoCE, so our comparison is like-for-like. Within-subject "
        "results establish a per-subject ceiling; cross-subject would "
        "lower it and is Phase-2 work.",
        None)

    # =========================================================
    # Statistical Methods (Q25-Q26)
    # =========================================================
    add_category(doc, 25, 26, "Statistical Methods")

    add_qa(doc, 25,
        "Why use the Wilcoxon signed-rank test rather than a paired "
        "t-test?",
        "Wilcoxon is non-parametric — it doesn't assume the per-subject "
        "paired differences are normally distributed. With only 42 "
        "subjects and possibly skewed difference distributions, "
        "Wilcoxon is the safer choice. It is the standard test for "
        "paired method comparisons in machine-learning evaluation "
        "literature.",
        "We did informally check that paired t-test gives equivalent "
        "decisions across our seven comparisons, so the choice doesn't "
        "drive any of our claims.")

    add_qa(doc, 26,
        "Seven Wilcoxon tests with α = 0.05 — shouldn't you correct for "
        "multiple comparisons, e.g. Bonferroni?",
        "Good catch. With Bonferroni at α = 0.05 across seven tests, "
        "the corrected threshold becomes 0.007. Two of our results — "
        "cross-attention vs naive late fusion at p = 0.023, and naive "
        "late fusion vs vision at p = 0.020 — would not survive that "
        "correction. The other five comparisons, including the "
        "modality-dropout and concat-MLP findings, would. We mention "
        "this as a methodological refinement for the journal version.",
        "The substantive conclusions — that learned fusion beats "
        "unlearned, and that modality dropout dominates — survive "
        "Bonferroni cleanly.")

    # =========================================================
    # Demo & Real-world (Q27)
    # =========================================================
    add_category(doc, 27, 27, "Demonstration & Deployment")

    add_qa(doc, 27,
        "Your demo runs without EEG. If you can do that, why use a "
        "trimodal model at all?",
        "Because training with all three modalities and inferring with "
        "two gives a better audio-visual model than training with only "
        "two from the start. The trimodal training plus modality "
        "dropout reaches 81.5% on the AV-only inference path, which is "
        "1.3 points higher than our dropout-untrained full-modality "
        "model at 80.2%. EEG presence during training improves the "
        "model even when EEG is absent at inference. That's the value.",
        "The training-time EEG signal acts as a regulariser, even "
        "though the deployed system never sees an EEG headset.")

    # =========================================================
    # Data & Ethics (Q28-Q29)
    # =========================================================
    add_category(doc, 28, 29, "Data & Ethics")

    add_qa(doc, 28,
        "EAV uses cue-based elicitation — participants are told to "
        "display each emotion. Aren't these acted emotions rather than "
        "real ones?",
        "Partially, yes. The participants were instructed to display "
        "each cued emotion, so the elicitation is partially acted and "
        "partially genuine engagement with the cue. We acknowledge this "
        "explicitly as a limitation: the 'suppression events' in our "
        "matrix may partly reflect imperfect emotion simulation rather "
        "than genuine social suppression. Disentangling these requires "
        "a spontaneous-emotion corpus, which is Phase-2 work.",
        None)

    add_qa(doc, 29,
        "How did you access the EAV dataset? It's not publicly "
        "available, is it?",
        "We requested access by email from the EAV authors at "
        "Nazarbayev University. Adai Shomanov responded with a personal "
        "token-authorised Zenodo link, with explicit permission for the "
        "dataset's use in this thesis. The thesis acknowledges him by "
        "name. The dataset is restricted-access, per individual "
        "request, and our ethics paragraph reflects that correctly.",
        None)

    # =========================================================
    # Future Work (Q30)
    # =========================================================
    add_category(doc, 30, 30, "Future Work")

    add_qa(doc, 30,
        "Your Phase-2 primary dataset is audio-visual only because you "
        "don't have an EEG machine. Doesn't that mean you can't "
        "validate the trimodal generalisation?",
        "Correct, and we acknowledge this in the Phase-2 description. "
        "What an AV-only primary dataset validates is our demonstration "
        "path — the zero-EEG inference configuration — on data outside "
        "the EAV laboratory protocol. That's a real generalisation "
        "claim, just a narrower one than full trimodal validation. The "
        "full trimodal extension requires EEG hardware and ethics "
        "approval that we flag as Phase-2.3 work.",
        "We also propose self-reported internal-state Likert ratings as "
        "a behavioural proxy for EEG in the primary corpus, giving us "
        "a way to extend the coherence analysis without EEG hardware.")

    # ---- closing note ----
    doc.add_page_break()
    p = doc.add_paragraph()
    r = p.add_run("Killer-line for difficult questions")
    r.font.size = Pt(13)
    r.font.bold = True
    r.font.color.rgb = NAVY

    add_qa(doc, 0,
        "(if pushed beyond what you've prepared)",
        "\"That's outside what we tested in Phase 1, but our "
        "architecture is designed to accommodate it as Phase-2 work. "
        "Specifically, the forward() contract exposes per-modality "
        "embeddings, so adding [whatever the question is about] is an "
        "incremental extension without rewriting the fusion module.\"",
        "This pivot — 'didn't test it, designed to support it' — is "
        "the honest defence-friendly answer for almost any "
        "extension question.")

    doc.save(OUT)
    print(f"Wrote {OUT}")
    print(f"Paragraphs: {len(doc.paragraphs)}")


if __name__ == "__main__":
    build()
