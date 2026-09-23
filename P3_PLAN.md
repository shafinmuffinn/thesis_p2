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
- [ ] M 0.5h — Alignment sequence check. DoD: per subject, does the EEG label sequence equal the audio and vision label sequences element-wise; mismatch count.
- [ ] M 1.5h — Identity probe. DoD: subject-ID decoding accuracy per modality, fold-norm vs subject-norm features, with chance level.
- [ ] M 2h — Leak-free within-subject rerun (validation split from the 280 train trials; cross_attn, concat_mlp, dropout). DoD: `p2_leakfree.csv` for 42 subjects.

**Checkpoint 23:** subject-norm, identity, alignment results in hand; leak-free rerun done or running.
**Sleep ~00:00–06:00**

## Wed 24 Sep

### Block 2 (06:30–12:00) — defensive analyses
- [ ] M 1h — Leak-free results. DoD: corrected within-subject table, delta vs P2 per variant, Wilcoxon vs naive.
- [ ] M 3h — Suppression audit. DoD: (a) % of events where AV consensus == cued label; (b) suppression matrix vs EEG confusion on AV-correct trials; (c) permutation null; one-paragraph verdict.
- [ ] S 1h — Cross-subject suppression matrix. DoD: 5x5 + base rate, compared with within-subject.

### Block 3 (13:00–19:00) — consolidate the calibration contribution
- [ ] M 1.5h — Head-vs-head Holm tests under subject-norm (concat vs dropout vs cross_attn). DoD: can state whether attention is tied with or below concat.
- [ ] M 1h — Calibration-curve figure data (accuracy vs n, random + skewed, with CIs). DoD: CSV/plot ready for Ch 7.
- [ ] M 1.5h — Catch-up slot for anything from Blocks 1–2 that slipped.
- [ ] N 2h — Few-shot (labeled vs unlabeled calibration) ONLY if ahead of schedule.

### Block 4 (20:00–23:30) — statistics
- [ ] M 1h — Holm correction across every reported test family. DoD: every p in Ch 6/7 has an adjusted p.
- [ ] S 1h — Bootstrap 95% CIs for headline means.
- [ ] N 1.5h — Per-class cross-subject recall/F1.

**Checkpoint 24 (go/no-go):** all experiments done. If anything slipped, apply the fallback.
**Sleep ~00:00–06:00**

## Thu 25 Sep — EXPERIMENT FREEZE 12:00 (only bug reruns before noon)

### Block 5 (06:30–12:00) — report restructure
- [ ] M 1h — Create `LateX_P3/` from the P2 source; P3 title page and date. DoD: compiles.
- [ ] M 2.5h — Correct Ch 3/4/6 for leak-free numbers; remove "best-test-epoch"; soften SOTA claim. DoD: no test-selected number remains.
- [ ] M 1.5h — Suppression-audit section, framed per the verdict. DoD: section written.

### Block 6 (13:00–19:00) — new content
- [ ] M 2.5h — Ch 7 additions: identity audit, subject-norm, dynamic/calibrated fusion negative results.
- [ ] M 2.5h — New chapter: unsupervised subject calibration (method, protocol, calibration curve, robustness, discussion).
- [ ] M 1h — Abstract, contributions, RQ8, limitations, conclusion. DoD: consistent with every table.

### Block 7 (20:00–23:30) — figures + slides draft
- [ ] M 2h — Figures: within-vs-cross per-modality; calibration curve; identity probe; suppression vs EEG confusion.
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

## REPLAN (CONFIRMED by result)
The new contribution is **unsupervised
subject calibration** (calibration-size curve + neutral-only + skewed robustness),
replacing few-shot as the headline. Few-shot drops to N: "labeled vs unlabeled
calibration" comparison, only if Wed Block 3 has room.
Block 1 order now: calibration_study (M) -> identity probe (M) -> alignment (M)
-> leak-free rerun (M).
