# Defence Q&A — 30 likely panel questions

Every number below is from the final report (`LateX_P3/`). Numbers marked *within* are within-subject; *cross* are subject-independent (5-fold, participants held out). "pp" = percentage points.

**How to use this.** Say the first sentence of each answer out loud as your headline, then give detail only if they ask for more. When a question exposes a weakness, admit it in the first sentence. Panels respect "yes, and here's what we did about it" far more than a defensive answer.

---

## A. The big picture

### 1. Summarise your thesis in one minute.

We built a trimodal emotion recogniser on the EAV dataset (EEG, speech, face; 42 people; 5 emotions) and asked one question: **when does learned, attention-based fusion actually beat simply averaging the three modalities' predictions?**

The answer has three parts:
- **On people the model has already seen (within-subject)**, learned fusion is only marginally better than averaging (80.0% vs 77.5%), and plain attention isn't better at all.
- **On new people (cross-subject)**, every learned method *loses* to averaging (57.6–58.8% vs 62.2%). The reason is that the features encode *who* the person is more strongly than *what they feel*.
- **A short unlabelled recording of the new user** removes that identity signal and lifts everything to ~74–76%, but then averaging ties the learned heads. **Learned fusion only wins when the new user also labels a few clips.** With 50 labelled clips, the learned heads beat averaging-with-the-same-labels by 3–5 pp, and attention trained with modality dropout is best, at 81.9%.

Alongside this, we audited our own earlier results, found that test-set selection had inflated them by 5–7.5 pp, and corrected everything.

> **Real-life example.** A new phone's Face ID asks you to enrol your face before it works well for you. Our system behaves the same way: out of the box it recognises people it was trained on, and a short enrolment makes it work for a stranger.

### 2. Your title says "Dynamic Attention-Based Fusion", but concat-MLP beats attention in several of your results. Does the title still hold?

It holds, but in a narrower and more precise sense than we originally expected.
- **Without labels,** attention offers no advantage. Concat-MLP edges it, and calibrated averaging ties both.
- **With labels,** it does: when a new user labels 20 clips, the attention model trained with modality dropout is the *only* head that significantly beats labelled averaging (+2.0 pp). At 50 clips it's the best head overall (81.9%), 0.94 pp above concat-MLP (p_Holm = 0.027). That comparison was added after we saw the results, so we report it as exploratory.
- **The attention is also measurably "dynamic":** its modality weights change from trial to trial, with a within-person standard deviation of 0.05–0.11.

So the thesis doesn't claim attention is always best. It identifies *when* dynamic attention earns its place: adapting to a new person from a few labelled examples. We state the limits openly (Chapter 7 and Limitation 3).

### 3. You call it cross-modal attention, but isn't it just self-attention over three tokens?

Yes, and we say so explicitly (Section 4.4.1). Each encoder outputs one pooled vector per clip, so each modality is a single token. Cross-attention from one single token to another reduces to a learned linear map. Lee et al.'s six directional cross-attention blocks would add parameters without adding expressiveness at this resolution. Self-attention over the three tokens lets every modality attend to every other modality, which is the cross-modal interaction we want.

Real pairwise cross-attention needs sequence-level features (per-frame vision, per-patch audio). That's future work, because it requires re-extracting features for all five folds and tuning a new architecture.

> **Real-life example.** If three people each hand you a one-line summary, "cross-attention between summaries" is just reading all three together. Cross-attention only makes a difference if each person gives you a whole transcript.

### 4. What is genuinely new here?

1. **The first subject-independent evaluation of trimodal fusion on EAV.** All three published EAV fusion papers (AMERL, Hyper-MML, EEG-MoCE) report within-subject results only.
2. **Subject calibration and few-shot adaptation, with fair controls.** Normalising each person's features with statistics from ~20 unlabelled clips turns learned fusion from worst to best on unseen people. The controls then establish what that does and doesn't buy: calibrated averaging ties the learned heads, and labels are what separate them.
3. **Mechanistic evidence:** participant identity is linearly decodable at 82–100% from every modality. Calibration removes it, and vision accounts for ~70% of the gain.
4. **An integrity audit** with three reusable tests, which overturned our own earlier headline claims.

### 5. Why EAV rather than a larger dataset such as DEAP or SEED?

EAV is the only public dataset with all three modalities we needed recorded together (30-channel EEG, speech audio and face video) in a conversational task. DEAP and SEED have EEG but no conversational speech; DEAP has facial video but no speech. EAV also shipped with per-modality baselines and loaders but **no fusion**, which left a clear gap. Its limitations are real, and we state them: it's cued (acted) emotion, it comes from one lab, and it has 42 people.

---

## B. Integrity and corrections

### 6. Your P2 report claimed 84.7%. Now you report 79.8%. What happened, and why should we trust anything in this report?

In P2 we evaluated each fusion head on the test set after every one of its 80 training epochs and reported the best epoch. That makes each reported number the *maximum of 80 test measurements*, which is biased upward. Naive averaging has no epochs, so it got no such boost, and the comparison was tilted towards learned fusion.

In this phase we re-ran exactly the same recipe and read one training run four ways. The difference between "best on test" and "best on validation" in the same run is the inflation itself: **5.3–7.5 pp**. Every learned-fusion number in this report is now leak-free, and we withdrew the state-of-the-art claim.

**Why trust it now?**
- Every later experiment had its design, primary measure and test family written into the script and committed to git **before** it was run, so the commit timestamps prove the analysis wasn't chosen after seeing the results.
- Every number traces to a results file.
- Every family of tests is Holm-corrected.
- We found this ourselves and reported it, rather than hoping nobody would notice.

> **Real-life example.** It's like a student taking a practice exam 80 times and reporting their best score as their ability. The fair number is how they do on an exam they haven't seen.

### 7. For the corrected figure you chose "final epoch on all training data". Isn't that also a choice you could have tuned?

It's the choice that involves **no selection at all**. The recipe was fixed in P2 (80 epochs, cosine schedule that anneals the learning rate to zero), so the final epoch is its natural end point. We designated it as the primary readout before seeing the numbers.

The alternative, picking the epoch on a 56-trial validation split, turned out to be *worse* than just stopping at the end, for every head. A validation set that small is too noisy to choose epochs well. So final-epoch is not only the principled choice; it also didn't favour us.

### 8. Were your per-modality encoders also selected on test data?

No. The EAV trainers train for a fixed number of epochs and keep the final weights; test accuracy is only printed. So the per-modality results (57.1 / 74.6 / 43.9%) and naive averaging (77.5%) are not test-selected.

One weaker exposure we do disclose: we reduced the epoch budgets (AST 5+5, ViT 3+2) after looking at three pilot participants' training curves. That's three hyperparameters chosen once, not a per-person choice.

### 9. You've withdrawn your suppression-matrix claims. What's left of the "affective coherence" idea?

Honestly, what's left is a **methodological result**, not a psychological one. Three tests showed the problem:
- In 87.8% of "suppression events", the audio-video consensus was simply the correct, cued emotion.
- The matrix correlates at r = 0.989 with the EEG classifier's own error pattern.
- Most decisively, when we shuffled EEG predictions within each person and emotion (which keeps every classifier's error rates but breaks the trial pairing), **no cell exceeded chance** within-subject.

So the "Anger↔Happiness suppression" was EEGNet confusing two high-arousal emotions. What survives:
- **A reusable test:** any coherence claim built from classifier disagreement must beat a class-conditional permutation null.
- **One candidate pattern:** Sadness→Neutral exceeds the null cross-subject (69 events vs 51 expected, p_Holm = 0.01). It's one cell of 40 tested and doesn't replicate within-subject, so it's a hypothesis for future work, not a finding.

> **Real-life example.** If a smoke detector often beeps when you cook, a log of its beeps is mainly a record of the detector's false alarms, not of fires. You have to compare it with how often it would beep by chance before you conclude anything about fires.

### 10. You say trial alignment across modalities is "unverified". Doesn't that invalidate the whole trimodal system?

No, but it limits one kind of claim.
- **What we checked:** all 84 train/test label sequences match element by element across the three modalities, but every sequence is sorted into class blocks. Sorted sequences with equal class counts *always* match, so the check can't distinguish correct alignment from any within-class shuffle. The released files contain no trial IDs.
- **What's unaffected:** the classification results. A model trained and tested on "same-emotion" triplets of clips is still a valid classifier of those triplets.
- **What's affected:** claims about a single moment of one person's behaviour, which is mainly the coherence analysis.
- **Indirect evidence the pairing isn't arbitrary:** in the permutation test, the modalities were *more* often right or wrong on the same trials than chance predicts (401 events vs 437 expected, p = 0.005).

> **Real-life example.** If two decks of cards are both sorted by suit, "the 5th card of each deck is a heart" tells you nothing about whether it's the same card.

### 11. In EAV, EEG is recorded while the participant *listens* and the audio-video while they *speak*. Why fuse signals from different moments?

Because both come from the same cued emotional interaction: the EEG captures the brain's response to the emotional prompt, and the audio-video capture the person's expressive reply. For *classifying the cued emotion*, both are legitimate evidence, and the fusion results show they're complementary (calibrated averaging beats the best single calibrated modality by ~9 pp).

What the asymmetry rules out is interpreting EEG-vs-face disagreement as "what they felt vs what they showed at that instant". That's one more reason we withdrew the suppression interpretation.

---

## C. Cross-subject evaluation

### 12. Why 5-fold subject-wise cross-validation instead of leave-one-subject-out?

The cost. The leakage rule forces us to **retrain all three encoders for every fold**, and one fold took about 2.7 GPU-hours (fold 1: 164 min). Five folds took roughly 14 GPU-hours; LOSO would need at least 113. Five-fold keeps the essential property: every person is tested exactly once as a complete stranger. That still gives 42 per-person accuracies for paired tests.

### 13. Why did vision collapse from 74.6% to 41.2% on new people?

Because within-subject, the vision model largely learned *each person's face*, not emotion in general. We have three pieces of evidence:
1. A linear probe identifies which of the 42 people a clip came from with **100%** accuracy from the vision features.
2. Training the vision encoder *longer* made cross-subject accuracy *worse* (45.8% → 41.9% on fold 0): the signature of memorising the training people.
3. Removing each person's feature offset improves cross-subject emotion decoding from vision the most of any modality (+13.6 pp). Calibrating vision alone recovers ~70% of the full calibration gain.

Audio, by contrast, transferred almost perfectly (57.1% → 57.5%).

> **Real-life example.** You can read your best friend's mood from a tiny change in their smile, but the same change on a stranger's face means nothing to you, because you learned *their* face, not faces in general.

### 14. Why does learned fusion lose to plain averaging on new people?

A learned head is trained on features where each person occupies their own characteristic region, their "offset". A new person arrives with an offset the head has never seen, so their inputs fall somewhere uninformative, and the head is misled. Averaging combines class *probabilities*, not features, so it has nothing that the offset can mislead.

We tested an alternative explanation, that learned heads fit the wrong modality weights for new people. Per-trial confidence weighting (no fitted weights at all) *also* lost to averaging, while per-person normalisation (no change to any weighting) fixed the heads. So the problem is the input, not the weighting.

### 15. How do you know your cross-subject numbers aren't contaminated by leakage?

Three safeguards:
1. **Encoders are retrained from scratch for every fold** on the training people only. Reusing the P2 per-person encoders would have been the easy shortcut, and it would have been silently wrong: those encoders had already been trained on the "held-out" person.
2. **Epochs are chosen on three held-aside *training* participants,** never on the test fold.
3. **Folds are a deterministic partition** (9/9/8/8/8), checked by a test so that train and test people never overlap.

> **Real-life example.** A drug trial must test the drug on patients who weren't used to develop it. If a doctor who had already treated a patient then "predicts" that patient's outcome, the prediction proves nothing.

### 16. 62% on new people. Is that useful?

On its own, it's a clear signal (chance is 20%) but not deployable. That's why the calibration chapter matters: a 100-second unlabelled recording takes it to ~73–76%, and 50 labelled clips take it to ~81%. That's higher than the within-subject result we originally reported for known people after correction (80.0%), though the protocols differ.

The honest message is that **new-user performance depends on what the new user provides**, and we quantify that trade-off precisely.

---

## D. Calibration and few-shot adaptation

### 17. Isn't calibration cheating? You use the test person's data.

No labels are used, and we tested the cheating concern directly.
- **The simplest version** computes the statistics from all 400 of the person's clips, including the scored ones. So we re-ran it with calibration clips **disjoint** from the scored clips. With only 20 separate clips, every head still beat uncalibrated averaging by 9.7–11.7 pp.
- **We also stress-tested the hidden assumption** that the calibration clips are balanced across emotions:
  - *neutral-only* calibration **fails*;
  - *heavily skewed* sessions need 50–100 clips.

It's a realistic deployment step: a short enrolment recording, like voice-assistant training. It must be reported as an assumption, and we do that (Limitation 4).

> **Real-life example.** Before you weigh anything, you zero the kitchen scale with the empty bowl on it. That uses the bowl, not the ingredients. Calibration is "zeroing" each person's baseline without knowing their emotions.

### 18. How is this different from existing domain-adaptation work, such as AdaBN?

The idea is closely related, and we cite it: AdaBN re-estimates batch-normalisation statistics on a new domain without labels. We apply that idea at the level of **each individual person** and to **fused multimodal features**. What's new is:
1. measuring how much calibration data is needed (~20 clips for significance, saturating at ~50);
2. showing which kinds of recordings break it (neutral-only);
3. linking it mechanistically to identity removal, per modality;
4. and, crucially, testing whether it helps *fusion* specifically, which it doesn't, because averaging gains just as much.

### 19. If calibrated averaging ties the learned heads, why build a learned fusion model at all?

Because the learned heads can do something averaging structurally cannot: **adapt to a person from labels.** With 50 labelled clips from a new user, the learned heads reach 80.4–81.9%, **3.2–4.8 pp above** averaging that uses the same labels to fit its class biases (77.2%). With 20 clips, only dropout-trained attention is significantly ahead (+2.0 pp).

So the recommendation is concrete:
- **no labels** → use calibrated averaging (simpler, just as good);
- **a few dozen labels** → use dropout-trained attention.

> **Real-life example.** Autocorrect with no history works about as well as any simple dictionary; after you've typed a few hundred words, the personalised model pulls ahead.

### 20. Is it realistic to expect a new user to label 50 clips?

Fifty 5-second clips is about four minutes of guided recording ("say something as if you're angry"), comparable to fingerprint or voice-assistant enrolment. In clinical or long-term monitoring settings, a one-time four-minute setup is reasonable.

That said, our labels are EAV's **cued** emotions. Self-labelled real emotions would be noisier, and we haven't tested that. The honest claim is "a short cued enrolment helps"; real-world labelling is future work.

### 21. Why does neutral-only calibration fail?

Calibration subtracts the person's *average* feature vector. If every calibration clip is Neutral, you subtract the *Neutral average* instead. Every emotional clip is then shifted by its emotion's difference from Neutral, which is exactly the signal you want to keep. So a relaxed "sit still and look at the camera" enrolment is the wrong design; the session needs varied emotional content, such as a natural conversation.

> **Real-life example.** If you zero a thermometer in an ice bath and assume that's "room temperature", every later reading is shifted by the gap.

### 22. You say identity "drops to chance" after calibration. Isn't that true by construction?

Partly, and we say so in the report. Per-person z-scoring removes each person's mean and variance, which are exactly what a linear identity classifier relies on. So "removed to chance" means **the linearly available first- and second-order identity signal** is gone. A non-linear probe, or identity carried in correlations between dimensions, could survive.

The substantive evidence isn't the drop to chance itself. It's that removing that signal **improves cross-subject emotion decoding** in every modality (+9–14 pp, 40–41 of 42 people), and most for vision, which depended on identity most.

### 23. Calibrating vision alone gives ~70% of the gain. What does that imply, and does it raise privacy concerns?

It confirms the face is the main carrier of identity that hurts transfer. Audio still contributes ~27%: audio transferred fine *on its own*, but its identity offset misleads a fusion head that reads all modalities together. EEG contributes little.

On privacy: yes. Emotion features that identify the person with 97–100% accuracy are effectively **biometric data**. Any deployment must store them as personal data, and the enrolment recording needs consent (Chapter 3, Ethics).

> **Real-life example.** Even a "mood-only" app that stores these features would effectively hold a voiceprint and a faceprint of every user.

---

## E. Attention, EEG and modality dropout

### 24. You measured attention weights. What did they show?

1. **The attention is dynamic:** each modality's weight varies from trial to trial (within-person SD 0.05–0.11 on means of 0.19–0.42).
2. **It's partly sensible:** it gives audio more weight when audio's own classifier is right, in all four settings.
3. **It's not always sensible:** it gives EEG *less* weight when EEG is right, in all four settings. It doesn't recognise when EEG is reliable.
4. **Calibration shifts weight from EEG to audio;** vision's weight doesn't change. So calibration helps vision by making its token informative, not by attending to it less.

Caveats: the differences are small (0.01–0.04), rollout is a heuristic, and attention weights aren't a full explanation (Jain & Wallace, 2019).

### 25. EEG is the weakest modality and costs money to record. Does it help at all?

It helps a little, and more once identity is removed.
- **Within-subject,** removing EEG from the dropout model costs only 0.63 pp (not significant).
- **Cross-subject without calibration,** it costs 0.89 pp (not significant).
- **With calibration,** it costs **1.45 pp, and that's significant.**

So EEG contributes measurable but modest information, and the system is designed to run without it: the zero-EEG path, trained with modality dropout, is the realistic deployment for anyone without a headset. Our honest position is that on EAV, EEG is a useful optional extra, not essential.

### 26. Your EEG accuracy (43.9%) is lower than AMERL's (53.5%) or EEG-MoCE's (62.7%). Why?

We deliberately kept **vanilla EEGNet** (74,933 parameters) as the EEG branch, trained from scratch on 280 trials per person, so that any fusion effect couldn't be credited to a stronger EEG model. The other papers use EEG-specific architectures.

Our 43.9% still beats the EAV paper's own EEGNet baseline (36.7%), and getting there required fixing two defects in the public EEGNet code that stopped it from training at all. A stronger EEG encoder is an obvious improvement and is listed as future work.

### 27. What does modality dropout do, and why does it help?

During training, for each sample, with probability 0.5, one random modality is zeroed and the others are scaled up. The model learns to make good predictions from any two modalities.

That buys two things:
- **Robustness:** it still works with EEG missing (79.2% within-subject, versus 79.8% with EEG).
- **Regularisation:** within-subject it lifts plain attention from 76.8% to 79.8% (+3.1 pp, p_Holm < 0.001). With labelled few-shot data, the dropout-trained model adapts best of all heads.

> **Real-life example.** Pilots train with simulated instrument failures so they don't over-rely on any single gauge; the training makes them better pilots even when everything works.

---

## F. Statistics, comparison and future work

### 28. Why Wilcoxon tests and Holm correction, and aren't your choices of test "family" arbitrary?

- **Wilcoxon signed-rank:** each method produces one accuracy per person on the same 42 people. The data are paired and not guaranteed to be normal, so the Wilcoxon signed-rank test is the standard choice.
- **Holm correction:** because we run several tests per question, Holm controls the chance of any false positive in that family. It's never weaker than Bonferroni and makes no independence assumption.
- **Families:** each is defined by the question a table answers. Where the conclusion *depends* on the family, we report every reasonable family rather than the most favourable one. The within-subject "concat beats naive" result is significant in two families and not in a third, so we call it **marginal**.

For the new experiments, the families were written into the code before running.

> **Real-life example.** Buy 20 lottery tickets and one will probably "win" something. Holm correction is how you stop counting luck as a discovery.

### 29. Are you better than the published state of the art on EAV (Hyper-MML, 76.65%)?

We make no state-of-the-art claim. Our leak-free within-subject figures (80.0% concat, 79.8% dropout-attention) are numerically higher. But none of the competing papers states how its reported epoch or checkpoint was chosen, and we showed that choice alone can move results by 5–7.5 pp. We also showed that within-subject accuracy on EAV is heavily identity-driven, so within-subject rankings are a weak basis for comparing methods.

Our distinctive contributions are elsewhere: the first subject-independent numbers, calibration and few-shot results, and a zero-EEG path. None of the other papers reports these.

### 30. What are the main limitations, and what would you do with six more months?

**Limitations:**
1. one laboratory corpus, with acted (cued) emotions;
2. calibration clips come from the same session;
3. the few-shot tuning grid chose its larger step count in 13 of 15 cases, so the gains may be underestimated, and the attention-vs-concat comparison is exploratory;
4. pooled features limit what attention can do;
5. trial alignment is unverified.

**Six more months:**
1. **Cross-session calibration:** record the same people twice and calibrate on one day, test on another.
2. **A pre-registered, wider few-shot study,** to confirm whether dropout-attention's lead is real.
3. **Sequence-level features with true pairwise cross-modal attention,** plus contrastive pre-training (MERCL), especially an identity-invariant version that could remove the need for enrolment.
4. **A small new recording study with self-reported emotions,** the only way to revisit the suppression question properly.

---

## Five questions to rehearse hardest

1. **Q6** (84.7% → 79.8%): lead with the correction, not an excuse.
2. **Q2** (the title): "attention earns its place when adapting from a few labels".
3. **Q17** (is calibration cheating?): lead with "disjoint clips, 20 are enough, no labels".
4. **Q19** (why learn fusion if averaging ties?): "labels — +3.2–4.8 pp at 50 clips".
5. **Q9** (the suppression claims): "classifier error explains it; the permutation null is the contribution".

## Items to confirm before the defence

- **The Pan et al. (2024) description:** P2 described it two contradictory ways; check it against the paper.
- **Your supervisor knows about the retraction** before the panel reads it.
- **Per-fold Stage A times:** Q12 uses fold 1's time only.
