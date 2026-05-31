# Thesis Context — Multimodal Emotion Recognition (Audio + Video + EEG)

This file is loaded into Claude's context every session in this directory. Keep concise; update as work evolves.

## Student & thesis
- **Topic:** Trimodal emotion detection — audio + video + EEG.
- **Deadline:** 10 days to tangible output (as of 2026-05-19).
- **Dataset:** EAV (EEG-Audio-Video) — Lee et al. 2024, *Scientific Data*. 42 subjects, 30-channel EEG + audio + video, 5 emotions (Neutral, Anger, Happiness, Sadness, Calmness), 200 interactions/subject (listen/speak), pseudo-random cue-based conversation. Dataset hosted on [Zenodo (DOI 10.5281/zenodo.10205702)](https://doi.org/10.5281/zenodo.10205702).
- **Compute:** Google Colab (T4 GPU confirmed in `personal_work.ipynb`); EAV pickles live at `/content/drive/MyDrive/Thesis_EAV/Input_images/{Audio,Vision,EEG}/subject_XX_*.pkl`.
- **Local dev machine:** MacBook M1 Air — code-only, no local training (no CUDA). All experiments run on Colab.

## Workflow (Drive + GitHub hybrid, decided Day 1)
- **Code** → private GitHub repo `thesis_p2`. Push from Mac, `git pull` (or fresh clone) in Colab each session via fine-grained PAT. Saves history, enables diffs, never lost.
- **Data (EAV pickles), checkpoints, results, HF model cache** → Google Drive at `MyDrive/Thesis_EAV/`. Mounted in Colab at `/content/drive/MyDrive/...`. Files in this tree survive Colab disconnects.
- **Notebooks** → kept under the GitHub repo so changes are committed; opened in Colab via File → Open notebook → GitHub.
- **Path config** → single source of truth in [paths.py](paths.py). All paths come from env vars (`THESIS_ROOT`, `EAV_PICKLES`, `CHECKPOINTS`, `RESULTS`, `PRETRAINED`) set in the Colab setup cell, with Mac defaults baked in.
- **Hard-coded paths in EAV / Chumachenko repos** → not edited. Every hard-coded string is in an `if __name__ == "__main__"` driver block; all library classes already accept `parent_directory` / `model_path` kwargs. Our own driver passes paths from [paths.py](paths.py).

## Google Drive layout (canonical)
```
MyDrive/Thesis_EAV/
├── Input_images/{Audio,Vision,EEG}/subject_XX_{aud,vis,eeg}.pkl   # 42 subjects × 3 modalities
├── raw/                          # optional: raw EAV from Zenodo
├── pretrained/                   # HF model cache (HF_HOME points here)
├── checkpoints/                  # per-subject per-model .pth, saved every epoch
├── results/                      # CSVs: day1_smoketest.csv, etc.
├── logs/                         # wandb / tensorboard
└── .github_token                 # fine-grained PAT (gitignored on Mac)
```

## Colab session start ritual (`< 2 min`)
1. Mount Drive: `drive.mount('/content/drive')`.
2. `git clone` (or `git pull`) the GitHub repo into `/content/thesis_p2` using token.
3. Set env vars: `THESIS_ROOT`, `EAV_PICKLES`, `CHECKPOINTS`, `RESULTS`, `LOGS`, `PRETRAINED`, `HF_HOME`.
4. `pip install -q transformers wandb` (the rest is pre-installed on Colab). **Do not install facenet-pytorch via plain pip** — it pins `numpy<2` and old torch, which silently downgrades Colab's torch/torchaudio and breaks AST. Only needed for raw-video MTCNN preprocessing; if used, install with `pip install -q --no-deps facenet-pytorch`. EAV's `requirements.txt` is otherwise mostly redundant on Colab (numpy/scipy/torch/librosa/opencv all pre-shipped).
5. `from paths import ensure_dirs, summary; ensure_dirs(); print(summary())`.
6. `assert torch.cuda.is_available()`.

The full setup cell is documented in the Day 1 chat thread and should be the first cell of every notebook.

## Honest Colab budget
- Free tier: ~12 h max session, disconnect after ~90 min idle, T4 (16 GB). Workable but fragile.
- Per-subject training time at *paper-faithful* hyperparams (Day 2+): AST ~25 min + ViT ~60 min + EEGNet ~5 min ≈ **90 min/subject**. 42 subjects ≈ 63 h total — must spread across many sessions, save checkpoints aggressively to Drive.
- Colab Pro ($10/mo) gives 24 h sessions + fewer disconnects + occasional V100/A100. **Recommended for the 10-day window.**

## Restart status (2026-05-31)
- **Project resumed** after a week-long break. New hard deadline: **2026-06-09**. Nine working days available.
- **Raw EAV data not available locally** (user deleted it after pickle generation; only 3 subjects' pickles cached on Drive currently). Re-preprocessing all 42 subjects requires re-downloading ~46 GB from Zenodo, which is feasible but expensive.
- **Compute**: no 4080 access tomorrow (2026-06-01). Plan is to buy Colab Pro for tomorrow's training and use 4080 later in the week when it becomes available. Pro on V100 is comparable to 4080; Pro on T4 is 2–3× slower.
- **Methodology simplification — Listen-vs-Speak analysis is dropped.** EAV's preprocessing pipeline did not preserve the per-trial task tag, and we don't have the raw data on hand to regenerate it. The headline coherence analysis is the **suppression matrix** instead — a 5×5 contingency table of (external AV emotion, internal EEG emotion) on trials where AV and EEG disagree and EEG is confident. This works on the current pickles without re-preprocessing. Behavioural validation via Listen-vs-Speak is recorded as Phase 2 future work.
- **Tonight's deliverable** (running on Mac, no GPU): `suppression_matrix.py` produces the thesis's headline figure from the existing 3-subject `day2_logits/*.npz` files. Validates the methodology in advance of the 42-subject rollout.
- **Trial alignment check (2026-05-31)**: per-class counts match across all three modalities for subjects 1–3 (80/80/80/80/80 each, total 400). Necessary condition for trial alignment met; treated as defensible to proceed without re-preprocessing.

## Day 4 status (2026-05-22)
- **Compute pivot**: 4080 GPU (16 GB VRAM) now available for local training. Workflow is: develop and train locally on the workstation, keep Drive only for archival of Day-2 artefacts. Colab Pro becomes optional, not load-bearing.
- **`fusion/` module created** at the project root, separate from the EAV repo (which we did not fork into our contribution).
  - [fusion/trimodal_attention.py](fusion/trimodal_attention.py) — `TrimodalAttentionFusion` class: 2.28 M trainable parameters, self-attention over 3 modality tokens, common embedding dim D=256, 2 transformer encoder layers, 8 heads, GELU, LayerNorm pre-norm.
  - `forward()` returns a dict with keys `logits`, `audio_embed`, `vision_embed`, `eeg_embed`, `fused_embed`. This is a stable contract for the Phase 2 temporal head and must not be broken by later edits.
  - `forward()` with `x_eeg = torch.zeros(B, 960)` produces finite output (logits norm ~0.99). Architecture permits zero-EEG inference; Day 7's modality dropout will make it meaningful.
- **Design decision**: 3-token self-attention rather than Lee et al.'s 6-way pairwise cross-attention. With single-token-per-modality pooled features (which is what our trained encoders produce), pairwise cross-attentions degenerate to learned projections. Self-attention across the modality-token sequence does the same work with fewer parameters and cleaner code. Going to sequence-level features and full 6-way CMA is a localised swap for Day 6 if MERCL stretch-goal is taken on.
- **Day 4 explicitly excludes**: connection to trained encoders (Day 5), training on EAV (Day 5), modality dropout (Day 7). Today is architecture only.

## Day 2 status (2026-05-20)
- **Softmax removed from `EEGNet_tor.forward`** (and the `nn.Softmax` instance from `__init__`). `CrossEntropyLoss` consumes raw logits and applies log-softmax internally; the previous double-softmax flattened gradients somewhat but, on a 50-epoch sanity test, the improvement was only ~1 pp (0.308 → 0.317 for Subject 1). Useful for cleanliness, not the dominant blocker.
- **Fixed `Trainer_uni.train()` mode bug**: the upstream EAV repo called `self.model.train()` once before the epoch loop, but `validate()` switches the model to `eval()` mode at the end of every epoch. From epoch 2 onwards, training happened with BatchNorm in eval mode (frozen running stats) and Dropout disabled — silently crippling convergence. Fix: moved `self.model.train()` inside the epoch loop. This is the more likely dominant cause of slow EEG learning.
- **Re-calibrated EEG expectations.** Per-subject EAV is a genuine low-data regime (280 train trials × 30 channels, no pretraining). EAV's published EEGNet baseline is 36.7 %, and Lee et al.'s EEG-alone numbers in their MERCL paper sit in the 30–45 % band on other datasets. A 40–55 % EEG-alone result at 350 epochs would be a good outcome; the thesis value is in *fusion gain*, not EEG-alone strength.
- **Procedure**: re-run the 50-epoch sanity test with the train-mode fix before triggering the full 350-epoch Day-2 run. Only commit GPU to the full run when the sanity number visibly moves.
- **Epoch budgets set to EAV-repo defaults**: AST audio 10 frozen + 15 fine-tune; ViT vision 10 frozen + 5 fine-tune; EEGNet 350 epochs.
- **Test logits now saved per (subject, modality) to Drive** under `MyDrive/Thesis_EAV/checkpoints/day2_logits/sub{NN}_{modality}.npz`. Required by Day 3 (late fusion) and Day 4 (cross-attention fusion) so encoders never need to be re-run.
- **Colab Pro: decided to purchase on Day 3.** Day 2 stays on Free for the first batch of subjects (~3 h on T4 is tolerable per session). From Day 3 onward, Pro for 24 h sessions, fewer disconnects, and occasional V100/A100.
- **Day 2 driver script**: cell-paste version lives in the working notebook; reads/writes to `RESULTS / "day2_full.csv"` (append-mode, incremental), so a kernel kill mid-session preserves completed rows.
- **Per-subject wall-clock estimate** (T4 Free, EAV defaults): audio ~45 min + vision ~25 min + EEG ~70 s ≈ **70–75 min/subject**. For 42 subjects this is ~50 GPU-hours, hence the Pro decision before broad rollout.

## Day 1 status (2026-05-19)
- Created [paths.py](paths.py) with env-var-driven path config and [.gitignore](.gitignore) excluding data/checkpoints/pretrained.
- Decided: no edits to EAV repo library files — all hard-coded paths live in `__main__` blocks only; our own driver passes paths.
- Smoke-test config for today: 3 subjects × {AST 2+2 ep, ViT 1+1 ep, EEGNet 50 ep} ≈ 15 min/subject.
- Output: `Thesis_EAV/results/day1_smoketest.csv` (columns: subject, modality, test_acc, seconds).
- Open todo (carry to Day 2): full-epoch training across all 42 subjects.
- Gotcha hit on Day 1: `pip install facenet-pytorch` downgraded numpy and torch on Colab, breaking torchaudio. Resolution: delete runtime, reconnect, install only `transformers wandb`. facenet-pytorch is only required for raw MTCNN preprocessing, which we skip because pickles are pre-processed.

## Project layout
```
thesis_p2/
├── multimodal-emotion-recognition/   # Reference only (gitignored). Not a dependency.
│                                     # Source for two snippets we copy on Day 4:
│                                     #   models/transformer_timm.py (Attention/AttentionBlock)
│                                     #   train.py:28-53 (softhard modality dropout)
├── EAV/                              # Primary repo: official EAV dataset code (3 modalities, NO fusion)
│   ├── Dataload_{audio,vision,eeg}.py   # per-modality preprocessors → pickles
│   ├── EAV_datasplit.py              # stratified train/test split (h_idx=56 → 70/30)
│   ├── Transformer_torch/            # AST (audio), ViT (vision), ShallowConvNet+Transformer (EEG)
│   ├── CNN_torch/                    # ResNet50 (vision), 1D-CNN (audio), EEGNet (EEG)
│   └── CNN_tensorflow/               # TF variants
├── personal_work.ipynb               # Colab inference demo: 3 frozen models, naive softmax average
└── CLAUDE.md                         # this file
```

## Repo #1 — Chumachenko et al. ICPR 2022 (audio–visual only)
- Paper: *Self-attention fusion for audiovisual emotion recognition with incomplete data* ([arXiv 2201.11095](https://arxiv.org/abs/2201.11095)).
- Visual: EfficientFace (ShuffleNetV2-style) pre-trained on AffectNet7; 15 frames → 1D-conv temporal stack.
- Audio: 10 MFCCs → 4 stacked 1D-conv blocks.
- Three cross-attention fusion variants in [models/multimodalcnn.py](multimodal-emotion-recognition/models/multimodalcnn.py): `lt` late-transformer, `it` intermediate-transformer (residual cross-attn), `ia` intermediate-attention (multiplicative gate).
- Modality dropout (`softhard`/`noise`) for missing-modality robustness.
- **Use to thesis:** copy the `Attention`/`AttentionBlock` modules from [models/transformer_timm.py](multimodal-emotion-recognition/models/transformer_timm.py) and the `softhard` dropout from [train.py](multimodal-emotion-recognition/train.py); ignore everything else.

## Repo #2 — nubcico/EAV (official EAV-dataset repo)
- Authors: Lee, Shomanov, Kabidenova, Yazici (Nazarbayev University) — same group as the dataset paper.
- **What it provides:** working preprocessing + per-modality trainers on EAV.
  - **Audio** ([Dataload_audio.py](EAV/Dataload_audio.py), [Transformer_torch/Transformer_Audio.py](EAV/Transformer_torch/Transformer_Audio.py)): resample to 16 kHz, 5 s segments → AST (`MIT/ast-finetuned-audioset-10-10-0.4593`) fine-tune to 5 classes.
  - **Vision** ([Dataload_vision.py](EAV/Dataload_vision.py), [Transformer_torch/Transformer_Vision.py](EAV/Transformer_torch/Transformer_Vision.py)): 25 frames per 5 s clip, optional MTCNN face crop (56×56 or 224×224) → ViT (`dima806/facial_emotions_image_detection`) fine-tune.
  - **EEG** ([Dataload_eeg.py](EAV/Dataload_eeg.py), [CNN_torch/EEGNet_tor.py](EAV/CNN_torch/EEGNet_tor.py)): 500 → 100 Hz downsample, Butterworth bandpass (0.5–45 or 5–30 Hz), 5 s windows → EEGNet (F1=8, D=8, F2=64, kernLength=300, Samples=500) or custom ShallowConvNet+Transformer.
  - **Split:** stratified per class, `h_idx=56` ⇒ 280 train / 120 test per subject ([EAV_datasplit.py](EAV/EAV_datasplit.py)).
- **Reported baselines (per-subject avg):** DeepFace=52.8% (vision), SCNN=36.7% (audio), EEGNet=36.7% (EEG). All low → headroom for improvement.
- **Critical gap:** no fusion. Each modality trained independently; no joint training, no contrastive alignment, no cross-attention. The repo's roadmap explicitly lists missing demo / inference / fusion files.
- **Known code issues:** hard-coded Windows paths (`C:\Users\minho.lee\...`), `Trainer_uni._loader` missing `self` in [Transformer_EEG.py](EAV/Transformer_torch/Transformer_EEG.py), [CNN_torch/CNN_EEG.py](EAV/CNN_torch/CNN_EEG.py) imports a nonexistent `Fusion.VIT_audio.Transformer_audio` module.

## Paper — Lee, Kim & Kim, *Bioengineering* 11(10) 997 (2024)
- **Authors:** Ju-Hwan Lee, Jin-Young Kim, Hyoung-Gook Kim (Chonnam / Kwangwoon) — *different* group than the EAV dataset authors.
- **Modalities:** trimodal (audio + video + EEG) — exactly the thesis target.
- **Architecture (3 stages):**
  1. **Modality encoders** (spatial → temporal):
     - Vision: ViT per frame → Residual-TCN.
     - Audio: 50 ms Hamming + STFT + 20 log-Mel filters → Vggish → Residual-TCN.
     - EEG: 5-band PSD (δ θ α β γ) → modified Conformer (no self-attention) → Residual-TCN.
     - Residual-TCN: 4 blocks, dilations 1/2/4/8, dilated conv + BN + ELU + dropout(0.5) + residual.
  2. **Pre-training with MERCL** (multimodal emotion recognition contrastive learning):
     - **AMCL** — supervised contrastive within each modality (intra-modal class clustering).
     - **EMCL** — supervised contrastive across modalities (inter-modal class clustering, 2N positives, 2M negatives).
     - **SMCL** — sample-wise positive-only alignment with margin α (closes modality gap).
     - `L_MERCL = λ₁·L_AMCL + λ₂·L_EMCL + λ₃·L_SMCL`.
     - Audio-energy-based sample filtering excludes silent clips (+1% F1).
  3. **Fine-tuning with cross-modal attention (CMA):** 6 directional pairwise multi-head attentions (A→V, V→A, V→E, E→V, A→E, E→A) → concat → MLP classifier. Encoders frozen during this stage in the paper's algorithm.
- **Datasets used:** DEAP, SEED, DEHBA (private), MTIY (private). **Not evaluated on EAV.**
- **Results (highlights):** DEAP 4-class 83.2% / SEED 3-class 90.9% / DEHBA 4-class **96.5%** / MTIY 4-class 91.6% (4-fold CV). Ablation: removing contrastive learning −4.04%, removing CMA −2.19%, removing both −5.28%.
- **Code:** **none released.** No GitHub link in paper or by author search.
- **Acknowledged limitations:** model complexity hurts interpretability; small datasets risk overfitting; high compute cost limits real-time use.

## Decided thesis direction
**Implement the Lee et al. 2024 methodology (MERCL + CMA fusion) on the EAV dataset using nubcico/EAV as the data + per-modality-encoder base.**

Why this works:
- The repo and the paper are *complementary*: repo has EAV data loaders + strong per-modality encoders; paper has the fusion architecture + training recipe but no code and no EAV evaluation.
- This produces a clean, citable thesis novelty: **first reproduction of the MERCL+CMA pipeline on the EAV benchmark.**
- The previous `personal_work.ipynb` already loads the exact same pickle layout — work is incremental, not from scratch.

## 10-Day Roadmap (2026-05-19 → 2026-05-28)

| Day | Goal | Concrete output |
|---|---|---|
| **1 (today)** | Setup + reproduce EAV repo baselines | Fix hard-coded paths; preprocessing pipelines run end-to-end on Colab; per-modality baseline numbers for ≥3 subjects |
| **2** | All-subject unimodal baselines | AST/ViT/EEGNet trained on all 42 subjects; results CSV + summary stats |
| **3** | Naive late-fusion baseline | Softmax-averaging across modalities reproduced from `personal_work.ipynb`; weighted-average + majority-vote variants |
| **4** | Trimodal cross-attention fusion module | Port `AttentionBlock` from Chumachenko repo; build 6-directional CMA + concat + MLP head; end-to-end trainable |
| **5** | Train and evaluate cross-attention fusion | Joint fine-tuning; per-subject results + comparison table vs late-fusion |
| **6** | MERCL contrastive pre-training (stretch) | Implement AMCL + EMCL + SMCL; pre-train encoders 5–10 epochs; then fine-tune CMA |
| **7** | Robustness + per-class analysis | Modality dropout (softhard); degraded-modality eval; confusion matrices; per-emotion F1 |
| **8** | Subject-independent eval (stretch) | LOSO or 5-fold across subjects on the best model |
| **9** | Writing — methods + results | Draft chapters: methods (figure + equations), experiments, results tables |
| **10** | Polish + **emotion-overlay demo** + defense prep | Architecture diagram, final tables, limitations section, slide deck, `demo_inference.py`/`demo_overlay.py`, rendered demo MP4 |

**MVP at Day 10 (if everything stretches):**
- Trimodal cross-attention model on EAV with results for at least one fold/split.
- 4 comparable numbers: unimodal (×3) → naive late fusion → cross-attention fusion.
- One confusion matrix per emotion, one per-modality contribution table.
- 15-page draft of methods + experiments + results.

**Cut-list under pressure:** MERCL (Day 6), LOSO (Day 8), Chumachenko-AV comparison.

## Foundational concepts — shortest path
- **Cross-attention** — read [models/transformer_timm.py](multimodal-emotion-recognition/models/transformer_timm.py) line-by-line (~1 h).
- **Supervised contrastive learning** — Khosla et al. 2020, just §3 (loss formula) (~2 h).
- **EEGNet** — Lawhern et al. 2018, §2 (~1 h).
- **AST** — Gong et al. 2021, §2 (~1 h).
- **ViT** — Dosovitskiy et al. 2021, §3 (~1 h).
- **Residual-TCN** — Bai et al. 2018 (TCN paper) + Lee 2024 §3.1.2 (~1 h).

## Tools / libraries to install immediately
```bash
conda create -n eav python=3.10 -y && conda activate eav
pip install torch torchaudio torchvision transformers facenet-pytorch \
            librosa scipy scikit-learn opencv-python pandas numpy \
            matplotlib seaborn tqdm mne wandb
```
- Use **wandb** or tensorboard from Day 1; tracking is non-optional once experiments multiply.
- Mount Google Drive in Colab; keep all pickles + checkpoints under `MyDrive/Thesis_EAV/`.

## Known pitfalls (from the codebases)
- `Trainer_uni._loader` in [Transformer_EEG.py](EAV/Transformer_torch/Transformer_EEG.py) is missing `self` — fix before use.
- [CNN_torch/CNN_EEG.py](EAV/CNN_torch/CNN_EEG.py) imports `Fusion.VIT_audio.Transformer_audio` which does not exist.
- **Fixed Day 1**: [CNN_torch/EEGNet_tor.py](EAV/CNN_torch/EEGNet_tor.py) had the same broken `from Fusion.VIT_audio.Transformer_audio import Trainer_uni` import on line 4. `Trainer_uni` is defined locally further down in the same file, so the line was dead. Removed and committed. **Side-effect**: that broken import was *silently* pulling in `TensorDataset`/`DataLoader`; removing it caused a `NameError` later. Fix: added `from torch.utils.data import DataLoader, TensorDataset` explicitly. If `CNN_EEG.py` ever needs to be imported, apply the same two-line fix (delete the bad line, add the explicit import).
- **Fixed Day 1**: Same file had a real semantic bug — the max-norm forward hooks were lambdas like `lambda module, inputs, outputs: module.weight.data.renorm_(p=2, dim=0, maxnorm=norm_rate)`. `renorm_` is in-place but returns the tensor; lambdas implicitly return their expression value; PyTorch's `register_forward_hook` *replaces the layer output* with any returned tensor. Result: `depthwiseConv`'s output was silently substituted with the layer's weight tensor of shape `(64, 1, 30, 1)`, which crashed `depthwiseBN(64)` with `running_mean should contain 1 elements not 64`. Same trap on `self.dense`. Fixed by extracting a module-level helper `_make_max_norm_hook(norm_rate)` that returns a real (non-lambda) function whose implicit return is `None`. **Note**: this is a real EAV-repo bug; their reported EEGNet numbers (36.7%) likely come from the TensorFlow variant, not this PyTorch path. Worth verifying.
- **Cosmetic in same file**: pre-existing unused imports (`weight_norm`, `torch.nn.functional as F`, `spectral_norm`) and a dead duplicate `_prepare_dataloader` at module scope. Pylance flags them; safe to ignore for now, clean up if/when refactoring.
- **Loss-vs-softmax mismatch (not yet fixed)**: `EEGNet_tor.forward` ends with `nn.Softmax(dim=1)` while `Trainer_uni` applies `nn.CrossEntropyLoss` on top. CE expects raw logits — applying it to softmax probabilities trains a degenerate model (vanishing gradients once any class becomes confident). Day 2 fix: drop the final `softmax` from `forward` (CE will softmax internally).
- **Fixed Day 1**: [Transformer_torch/Transformer_Vision.py](EAV/Transformer_torch/Transformer_Vision.py) had a head-size bug — replaced `model.classifier` with a 5-output `nn.Linear` and set `model.num_labels = 5`, but never updated `model.config.num_labels`. When the trainer passed `labels=` to the model, transformers' internal CE loss tried to `.view(-1, 7)` on (B, 5) logits and crashed (`RuntimeError: shape '[-1, 7]' is invalid for input of size 160`). Fixed by using `AutoModelForImageClassification.from_pretrained(model_path, num_labels=5, ignore_mismatched_sizes=True)`, which re-inits the head AND updates the config.
- **Potential future bite**: [Transformer_torch/Transformer_Audio.py](EAV/Transformer_torch/Transformer_Audio.py) has the same structural issue — replaces `classifier.dense` but doesn't update `config.num_labels`. It currently works only because the trainer computes the loss externally (`logits = model(x).logits; loss = nn.CrossEntropyLoss()(logits, t)`), never passing `labels=` to the model. If anyone refactors to use the model's internal loss path, apply the same `from_pretrained(num_labels=..., ignore_mismatched_sizes=True)` fix.
- **Fixed Day 1**: [Transformer_torch/Transformer_Vision.py](EAV/Transformer_torch/Transformer_Vision.py) `preprocess_images` ran the HF processor one image at a time over ~10k images per subject, built a 4 GB list, `torch.stack`'d it (peak ~8 GB CPU RAM), then `.to(device)` dumped 4 GB onto GPU. Colab Free (12.7 GB CPU RAM) OOM'd silently during the stack with no progress prints, so it looked like a hang. Fixed: batch processor calls (64 imgs at a time, ~10× faster), keep result on CPU (DataLoader moves per-batch), print progress every 20 batches.
- Vision preprocessing only keeps **Speaking** clips ([Dataload_vision.py:47](EAV/Dataload_vision.py#L47)) — Listening videos are silently skipped.
- EEG label selection uses `SELECTED_CLASSES = [1,3,5,7,9]` (Listening only) in [Dataload_eeg.py:33](EAV/Dataload_eeg.py#L33) — pairs with Speaking-only video selection? Verify alignment before fusion.
- Audio emotion-to-index map is non-contiguous: `{Neutral:0, Sadness:1, Anger:2, Happiness:3, Calmness:4}` — same in vision; **must** match across modalities.
- AST `feature_extractor` allocates large tensors — keep batch size ≤ 8 on T4.
- Chumachenko's preprocessing scripts use hard-coded `/lustre/scratch/...` paths — ignore.

## Methodology framing — Phase 1 vs Phase 2 (decided 2026-05-22)
- **Phase 1 = this thesis.** Trimodal (A + V + E) clip-level emotion recognition trained and evaluated on EAV. Deliverable: a cross-attention fusion architecture that beats naïve late fusion on EAV per-subject and per-LOSO metrics, robust to missing EEG via modality-dropout training. Demo: sliding-window inference (5 s window, 1 s stride) on a personal video clip, with EEG zeroed and temporal smoothing applied post-hoc.
- **Phase 2 = explicit future work, out of scope.** Per-second continuous emotion tracking on in-the-wild video. Requires: temporally-annotated in-the-wild dataset (DFEW / MAFW / custom); sequence-to-sequence temporal head (LSTM / Residual-TCN / temporal transformer) on top of frozen per-modality encoders; robust face detection, identity tracking, and cut detection in the visual preprocessor; voice activity detection and optional source separation in the audio preprocessor. None of these are part of the 10-day thesis.
- **Architectural commitments to keep Phase 2 cheap to start.**
  - The Day 4 cross-attention fusion module **must** expose per-modality pre-classifier embeddings as named outputs of `forward()`. A future temporal head reads these without any architectural rewrite.
  - `forward(x_audio, x_visual, x_eeg=torch.zeros_like(...))` **must** be a valid call producing finite outputs. The Day 7 modality-dropout training step (per Chumachenko's `softhard` scheme) is what makes this true at the parameter level, and is a first-class design requirement rather than a robustness ablation.
- **Defence framing.** The honest story is: "Phase 1 architecture validated on EAV under controlled conditions, with inference-time degradation to missing EEG demonstrated; closing the domain gap with a temporally-annotated in-the-wild dataset is Phase 2." Do not oversell what the EAV-trained model will do on movie input.

## Demo plan (decided 2026-05-19)
- **Form factor: file-based, not live.** Input is a recorded MP4; output is a copy of the same MP4 with a per-segment emotion-label overlay rendered on top. Live webcam/mic was evaluated and rejected — high engineering cost (rolling buffers, on-device MTCNN, M1 has no CUDA, audio/video sync) for negligible additional value over a recorded clip.
- **EEG at demo time: zeros, with modality-dropout-trained fusion.** Demo machine has no EEG hardware. The Day-7 modality-dropout / softhard training step is therefore load-bearing for the demo — it teaches the fusion model to make sensible predictions when one modality is zeroed. At demo time the EEG branch receives a zero tensor of the correct shape and the model degrades gracefully. The overlay carries a small "EEG unavailable; prediction uses audio + video only" caption. **Do not** feed dummy EEG into a model trained without modality dropout — predictions become indefensible.
- **Prediction granularity.** Model issues one prediction per 5-second window. Use overlapping windows at 1-second stride for smoother label transitions on screen (fallback: non-overlapping 5-second segments if time is short).
- **Output rendering stack.** OpenCV (`cv2`) for frame-by-frame overlay; FFmpeg to re-attach the original audio (cv2.VideoWriter produces silent video). Optionally PIL for nicer text rendering. Overlay shows: coloured emotion pill (top-left), five-bar confidence row (bottom), timestamp (bottom-right).
- **Time budget**: ~2 focused days of engineering total. Slotted into Day 10. Dependencies: fusion model (Day 5), modality dropout training (Day 7), thesis draft (Day 9).
- **Minimum viable fallback** if Day 10 is tight: skip the overlay video; pick 3 EAV test clips, run prediction, render a static probability-bar image per clip. Defensible, finishable in 2–3 hours.
- **Scripts to write**: `demo_inference.py` (video → list of (start_time, emotion, confidence)) and `demo_overlay.py` (renderer that consumes that list plus the original video).

## Reference papers (prioritized for this thesis)
1. **Lee, Kim, Kim** — *Emotion Recognition Using EEG Signals and Audiovisual Features with Contrastive Learning*, Bioengineering 11(10) 997, 2024 — primary methodology.
2. **Lee et al.** — *EAV: EEG-Audio-Video Dataset for Emotion Recognition in Conversational Contexts*, Scientific Data 2024 — dataset paper, [Nature link](https://www.nature.com/articles/s41597-024-03838-4).
3. **Chumachenko et al.** — *Self-attention fusion for audiovisual emotion recognition with incomplete data*, ICPR 2022 — fusion + modality-dropout reference code.
4. **Khosla et al.** — *Supervised Contrastive Learning*, NeurIPS 2020 — for MERCL implementation.
5. **Lawhern et al.** — *EEGNet*, J. Neural Engineering 2018.
6. **Gong et al.** — *AST: Audio Spectrogram Transformer*, Interspeech 2021.
7. **Dosovitskiy et al.** — *ViT*, ICLR 2021.
8. Survey: *EEG-based Multimodal Emotion Recognition: Recent Progress, Challenges, and Future Directions*, ACM TOMM 2025 — lit-review anchor.
