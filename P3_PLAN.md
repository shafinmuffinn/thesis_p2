# P3 Plan — submission end of Fri 26 Sep 2026

New contribution: **unsupervised subject calibration for trimodal fusion**
(replaced few-shot on 23 Sep after calibration_study held up), on top of
an integrity package (#1 leak-free rerun, #2 suppression audit, #7 stats) and the
identity audit (#3). Report = full thesis-length P3 document absorbing P2 + Ch 7.

Clock times are suggestions; shift blocks, keep the order and the freezes.
Priority: M = must, S = should, N = nice. DoD = definition of done.

---

## Tue 23 Sep — Block 1 (now → ~23:30, ~5h) — integrity quick wins

### Experiments
- [x] M 0.5h — Run `--subject-norm` Stage C, all folds. DoD: `cv5_fusion_subjectnorm.csv` exists; summary table sent.
- [x] M 0.5h — Alignment sequence check. DoD: per subject, does the EEG label sequence equal the audio and vision label sequences element-wise; mismatch count.
- [x] M 1.5h — Identity probe. DoD: subject-ID decoding accuracy per modality, fold-norm vs subject-norm features, with chance level.
- [x] M 2h — Leak-free within-subject rerun (validation split from the 280 train trials; cross_attn, concat_mlp, dropout). DoD: `p2_leakfree.csv` for 42 subjects.

**Checkpoint 23:** subject-norm, identity, alignment results in hand; leak-free rerun done or running.
**Sleep ~00:00–06:00**

## Wed 24 Sep

### Block 2 (06:30–12:00) — defensive analyses
- [x] M 1h — Leak-free results. DoD: corrected within-subject table, delta vs P2 per variant, Wilcoxon vs naive.
- [x] M 3h — Suppression audit. DoD: (a) % of events where AV consensus == cued label; (b) suppression matrix vs EEG confusion on AV-correct trials; (c) permutation null; one-paragraph verdict.
- [x] S 1h — Cross-subject suppression matrix. DoD: 5x5 + base rate, compared with within-subject.

### Block 3 (13:00–19:00) — consolidate the calibration contribution
- [x] M 1.5h — Head-vs-head Holm tests under subject-norm (concat vs dropout vs cross_attn). DoD: can state whether attention is tied with or below concat.
- [x] M 1h — Calibration-curve figure data (accuracy vs n, random + skewed, with CIs). DoD: CSV/plot ready for Ch 7.
- [x] M 1.5h — Catch-up slot for anything from Blocks 1–2 that slipped.
- [~] N 2h — Few-shot — CUT (24 Sep).

### Block 4 (20:00–23:30) — statistics
- [x] M 1h — Holm correction across every reported test family. DoD: every p in Ch 6/7 has an adjusted p.
- [x] S 1h — Bootstrap 95% CIs for headline means.
- [x] N 1.5h — Per-class cross-subject recall/F1.

**Checkpoint 24 (go/no-go):** all experiments done. If anything slipped, apply the fallback.
**Sleep ~00:00–06:00**

## Thu 25 Sep — EXPERIMENT FREEZE 12:00 (only bug reruns before noon)

### Block 5 (06:30–12:00) — report restructure
- [x] M 1h — Create `LateX_P3/` from the P2 source; P3 title page and date. DoD: compiles.
- [x] M 2.5h — Correct Ch 3/4/6 for leak-free numbers; remove "best-test-epoch"; soften SOTA claim. DoD: no test-selected number remains.
- [x] M 1.5h — Suppression-audit section, framed per the verdict. DoD: section written.

### Block 6 (13:00–19:00) — new content
- [x] M 2.5h — Ch 7 additions: identity audit, subject-norm, dynamic/calibrated fusion negative results.
- [x] M 2.5h — New chapter: unsupervised subject calibration (method, protocol, calibration curve, robustness, discussion).
- [x] M 1h — Abstract, contributions, RQ8, limitations, conclusion. DoD: consistent with every table.

### Block 7 (20:00–23:30) — figures + slides draft
- [~] M 2h — Figures: `make_p3_figures.py` written + smoke-tested locally; needs one Colab run (Drive data).
- [ ] M 1.5h — Slide skeleton (15–18 slides) with figures.

**Checkpoint 25:** full draft compiles, `verify_latex.py` passes, slide skeleton exists.
**Sleep ~00:00–06:00**

## Fri 26 Sep — FINAL BUFFER: writing, slides, rehearsal only. No experiments.

### Block 8 (06:30–12:00) — finish report
- [ ] M 2h — Trace every number to a CSV.
- [ ] M 1.5h — Engineering-challenges chapter, captions, bibliography.
- [ ] M 1h — Final compile, ToC, overfull boxes.

### Block 9 (13:00–18:00) — slides + defence prep
- [ ] M 2.5h — Finish slides + speaker notes.
- [ ] M 2h — Mock Q&A: 15 hard questions with 2–3 line answers.

### Block 10 (18:00–22:00) — rehearse + submit
- [ ] M 1.5h — Two timed run-throughs.
- [ ] M 1h — Package and submit; git tag.
- [ ] 1.5h slack.

---

## Fallback

**Cut in this order:** per-class analysis → bootstrap CIs (keep Holm) → cross-subject
suppression matrix → drop the skewed condition from the report (keep random) → slide polish.

**Never cut:** leak-free rerun (#1), suppression audit (a)+(b), identity probe, Holm.

**Minimum viable contribution:** first subject-independent evaluation on EAV
+ unsupervised subject calibration (20 unlabeled clips, +10-12pp over averaging)
+ identity probe + corrected leak-free within-subject baseline.

**Triggers:**
- Leak-free rerun not done by Wed 12:00 → run cross_attn + dropout only.
- Suppression audit shows matrix ≈ EEG confusion → reframe honestly as EEG error
  structure; do not drop the section.

## Progress log
- 23 Sep: plan set.
- 23 Sep: `--subject-norm` run DONE. Learned heads 58-59% -> 74-76% cross-subject
  (concat 75.9, dropout 75.0, cross_attn 73.9, dropout_av 73.6); naive unchanged
  at 62.2. Title-rescuing IF it survives: (1) transductive on scored trials,
  (2) hidden class-balance prior. `calibration_study.py` written to test both.

- 23 Sep: calibration_study DONE. Gain holds with DISJOINT calibration: random
  20 clips (~100 s, no labels) -> heads 72.5-73.8% vs naive 62.1 (+10-12pp, all
  Holm-sig); saturates ~50 clips. Neutral-only calibration FAILS (worse/tied).
  Skewed needs ~50-100 clips. skewed n=200 cell unreliable (rejected draws) --
  exclude/flag. concat_mlp still edges attention variants by ~0.8-1.9pp.
- 23 Sep: identity_probe.py written; below-chance artefact caught and fixed
  (normalisation stats now from probe-train half only).

- 23 Sep: identity_probe DONE. Identity linearly decodable: vision 100%, audio
  97%, EEG 82% (chance 2.4%); subject-norm -> chance in all three. Emotion
  cross-subject linear probe gains: vision +13.6, EEG +10.1, audio +9.2pp
  (40-41/42 improved, all p<1e-4). REVISE Ch7: identity is in ALL modalities
  (audio 97% yet transferred at parity) -> what matters is entanglement with
  the emotion decision; vision most entangled. "Removed to chance" is partly
  by construction (z-scoring removes 1st/2nd moments) -- say so.
- 23 Sep: p2_leakfree.py + label_sequence_check.py written and tested.

- 24 Sep: p2_leakfree DONE. Rerun reproduces P2 (naive 77.48; test-selected
  concat 81.65 vs P2 81.7). Selection inflation 5.3-7.5pp. Leak-free (final_full,
  primary, justified a priori: no selection, full 280): concat 79.96, dropout
  79.80, dropout_av 79.17, cross_attn 76.75, naive 77.48. Heads vs naive is
  FAMILY-DEPENDENT: concat/dropout +2.4pp, p_holm 0.025 (4 primary tests),
  0.031/0.034 (7-test family), 0.050/0.059 (8-test incl. val_selected).
  Report as MARGINAL, give both families. (Corrected 24 Sep.) SOTA claim FALLS. Survives:
  dropout > cross_attn +3.1pp p<1e-4; concat > cross_attn +3.2pp p=0.0017.
  SPINE: learned fusion beats averaging only with enough data AND identity-free
  features (within: tied; cross raw: -3 to -4.5; cross calibrated: +10 to +14).
- 24 Sep: suppression_audit.py written; validated (silent on independent
  errors, detects planted coupling).

- 24 Sep: alignment DONE: 84/84 splits match but 0/84 interleaved (all
  class-blocked) -> trial correspondence UNVERIFIED.
- 24 Sep: suppression_audit DONE. Within: reproduces 401 events; matrix vs EEG
  confident-error matrix r=0.956 (all) / 0.989 (AV-correct); NO cell exceeds
  the within-class permutation null; total events BELOW null (401 vs 437+/-12,
  p=0.005) -> modalities MORE coherent than independent errors (also evidence
  of real shared trial-level structure, not proof of exact alignment). Cross:
  621 events; Sadness->Neutral 69 vs null 51 (z=+4.5, p_holm=0.01) = only cell
  exceeding chance anywhere (candidate, not finding); Happiness->Anger below
  null. P2 headline patterns DO NOT survive -> reframe as methodological
  contribution (class-conditional permutation null for coherence claims).
- 24 Sep: p3_stats.py written (all families, Holm, bootstrap CIs, calibration
  curve data, per-class recall).

- 24 Sep: p3_stats DONE (tables in results/p3_stats/). Calibrated cross-subject:
  concat 75.93 [73.5,78.2], dropout 75.02, cross_attn 73.87, dropout_av 73.57,
  naive 62.15; all heads > naive +11.4..13.8pp (40-42/42), p_holm<2e-7.
  Head ranking SIGNIFICANT: concat > dropout (+0.92, p_holm .031) > cross_attn
  (+1.15, p_holm .006); dropout > dropout_av (+1.45). Per-class: calibration
  helps Calmness most (concat .365->.618), macro-F1 .582->.758.
- 24 Sep: ALL EXPERIMENTS DONE. Decision: freeze experiments now; few-shot cut.

## >>> RESUME HERE <<<
24 Sep: scripts for A/B/D/E written, CPU-tested on fake data, pushed (designs
pre-specified in docstrings BEFORE results). Next = ONE Colab session:
  git pull  (banner >= 7413f15)
  python calibrated_naive.py --selftest && python calibrated_naive.py            # A
  python cv_pipeline.py --stages C --subject-norm-only vision|audio|eeg          # D x3
  python attention_analysis.py --selftest && python attention_analysis.py       # E
  python fewshot_calibration.py                                                  # B
  python p3_stats.py ; python make_p3_figures.py
Paste: calibrated_naive output, p3_stats sections 1/3/7, attention output,
fewshot output; download 4 PNGs -> LateX_P3/images/.
Then: write A/B/D/E into Ch 8 (A: tab:cal_main row + sec:cal_fairness + Lim 3;
D: mechanism table; E: new section "Is the attention dynamic?" + cite Abnar &
Zuidema 2020, Jain & Wallace 2019; B: new section "Labelled vs unlabelled").
Cutoffs: B dropped if no sane output by Fri 25 Sep 16:00; ALL experiments
frozen Fri 25 Sep 20:00; Sat 26 = Turnitin rewrite of Ch1-5 + C1-C13 + Ch6
6.1-6.2, Overleaf compile, slides (8-10), submit by 23:00.
Supervisor not yet told about the retraction -> tell them.

## Decisions 24 Sep (user answers)
- Turnitin compares against OUR OWN P2; ~20% similarity allowed INCLUDING references
  and template pages -> body text must be essentially 0% verbatim P2.
- Full standalone thesis; Ch1-5 condensed in fresh wording; no page limit.
- Rewrite also C1-C13 (engineering) and Ch6 per-modality/naive sections. Cross-subject
  chapter was never submitted anywhere. P2 diagrams may be reused.
- Supervisor does NOT yet know about the SOTA retraction / suppression result.
- Defence talk 5-10 min with slides (-> ~8-10 slides, not 15-18); slides later.
- Experiments chosen: A calibrated-naive control, B labelled few-shot, D per-modality
  calibration, E attention-weight analysis. Order A -> D -> E -> B (B dropped first).
- Submission: 26 Sep 23:00 (Dhaka).

## REPLAN (CONFIRMED by result)
The new contribution is **unsupervised
subject calibration** (calibration-size curve + neutral-only + skewed robustness),
replacing few-shot as the headline. Few-shot drops to N: "labeled vs unlabeled
calibration" comparison, only if Wed Block 3 has room.
Block 1 order now: calibration_study (M) -> identity probe (M) -> alignment (M)
-> leak-free rerun (M).

- 24 Sep: REPORT DRAFTED in LateX_P3/. Found + fixed two more report/code
  mismatches: (1) Ch4 described softhard dropout as 4 sub-batches (4N); code
  zeroes one random modality per sample w.p. 0.5 and rescales 3/n_active (C20);
  (2) P2 claimed Sadness->Neutral was the pilot's most consistent pattern; the
  contemporaneous log says Calmness->Neutral (removed). Pilot EEG 24.4->50.6
  now attributed to bug fixes AND 50->350 epochs. Encoders confirmed NOT
  test-selected (EAV trainers keep final epoch) -> per-modality + naive stand.
  New local tests (p3_stats robustness family): dropout_av > cross_attn +2.42pp
  p_holm .0043; dropout_full - dropout_av +0.63 n.s.

- 24/25 Sep: A/B/D/E RESULTS (all pre-specified, pushed before running):
  A calibrated averaging (logit centring) = 74.87 transductive / 72.95 n=20 /
    73.98 n=50; heads vs it -1.30..+1.10pp, ALL n.s. (p_holm>=0.495). ->
    HEADLINE CHANGED: the 11-14pp is CALIBRATION, not learned fusion.
    Cal. averaging beats cal. audio by ~9pp (fusion still matters).
  D vision-only calibration recovers 67-72% of gain, audio-only 25-30%,
    EEG-only ~0-15% (only concat sig).
  E attention IS dynamic (within-person SD 0.05-0.11); up-weights audio when
    audio right (all 4 sig); DOWN-weights EEG when EEG right (all 4 sig);
    calibration moves weight EEG->audio, vision unchanged.
  B labelled fine-tune on same clips: +5.6..+7.4pp at n=50 (42/42), heads
    80.4-81.9%; exploratory: labelled dropout-attn > labelled concat +0.94pp
    p_holm .027 at n=50 only. Grid picked 50 steps in 13/15 cases (may underestimate).
  Written into Ch 8 (commit bf6d9e9). PENDING: labelled_naive.py (labelled
    averaging control) -> decides whether learned fusion uses labels better
    than averaging. Then rewrite abstract, Ch1 contributions, Ch7 RQ7/discussion,
    Ch11 summary/contrib/RQ table/Lim 3 around the new headline.

- 25 Sep: labelled_naive DONE. Labelled averaging (5 class biases) adds
  +1.4..+2.7pp over centred averaging (all sig); weights variant useless
  (lambda=0.1 every fold). Labelled heads vs labelled averaging: n=5 tie/worse
  (dropout_av -2.09 sig), n=10 tie, n=20 only dropout_full +2.02 sig, n=50 ALL
  heads +3.2..+4.8 sig (dropout_full +4.75, 40/42).
  FINAL HEADLINE: without labels, calibration (not the fusion operator) is what
  matters (calibrated averaging 74.9 = heads). With 20-50 labelled clips,
  learned heads beat labelled averaging; dropout-trained attention first/most.
  Written into abstract, Ch1, Ch6, Ch7, Ch8(cal), Ch11 (commit 22a652e).
  EXPERIMENTS FROZEN. Remaining \pend: Ch7 per-fold Stage A times (1).
  Next: Turnitin rewrite (Ch1-5 condensed fresh, C1-C13, Ch6 6.1-6.2; ToC),
  re-measure verbatim overlap, Overleaf compile, slides (8-10), supervisor.

