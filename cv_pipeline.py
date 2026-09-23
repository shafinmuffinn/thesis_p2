"""Track A -- 5-fold subject-wise cross-validation for all five fusion variants.

What this retires
-----------------
Limitation 2 of the P2 thesis: every reported accuracy was within-subject, so
none of them supports a generalisation claim. This pipeline re-runs the whole
stack under a subject-disjoint protocol and produces one accuracy per subject
per variant, directly pairable with the P2 within-subject numbers.

Why the Day-5 caches cannot be reused
-------------------------------------
`day5_state_dicts/sub{NN}_{modality}.pt` was fine-tuned on subject NN's own
trials, and `day5_features/sub{NN}.npz` was produced by that encoder. Training
a fusion head on subjects 1-41 and testing on subject 42 using those caches
would test on features produced by a model that had already seen subject 42.
So Stage A retrains the encoders from scratch, once per fold, on training
subjects only. This is the expensive part, and it is not optional.

Protocol
--------
    5 folds, subject-disjoint (see cv_split.py):  9/9/8/8/8 held out.
    For fold k:
      Stage A  train AST + ViT + EEGNet on train_subjects(k) only
      Stage B  extract features AND per-modality logits for all 42 subjects
               using fold k's encoders
      Stage C  train each fusion variant on train_subjects(k) features,
               evaluate on each held-out subject separately

    Every subject is held out exactly once, so a full 5-fold run yields 42
    per-subject accuracies per variant -- the same shape as the P2 results
    table, enabling a paired Wilcoxon test of protocol effect.

    Model selection (best-epoch checkpointing, early stopping) uses an INNER
    validation split carved from the training subjects, never the held-out
    fold. Selecting on the test fold would be a second, subtler leak.

Variants
--------
    audio_only      the AST encoder's own head        (no training)
    vision_only     the ViT encoder's own head        (no training)
    eeg_only        the EEGNet encoder's own head     (no training)
    naive_late      mean of per-modality softmax      (no training)
    cross_attn      TrimodalAttentionFusion
    concat_mlp      ConcatMLP
    dropout_full    TrimodalAttentionFusion + softhard dropout, all modalities
    dropout_av      the same trained model, evaluated with EEG zeroed

Running it
----------
    python cv_pipeline.py --folds 0            # calibrate on one fold first
    python cv_pipeline.py --folds 0,1 --stages A,B,C
    python cv_pipeline.py --stages C           # re-run fusion only, all folds

Folds are independent, so three people on three GPUs can split them:
person 1 --folds 0,1   person 2 --folds 2,3   person 3 --folds 4

Every stage is cache-checked and resumable; a killed session loses at most the
artefact currently being written.
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from cv_split import N_FOLDS, SUBJECTS, test_subjects, train_subjects, validate
from paths import CHECKPOINTS, EAV_PICKLES, HF_AUDIO_MODEL, HF_VISION_MODEL, RESULTS

sys.path.insert(0, str(Path(__file__).resolve().parent / "EAV"))

from concat_mlp_baseline import ConcatMLP                      # noqa: E402
from day7_modality_dropout import eval_mode, softhard_dropout  # noqa: E402
from fusion.trimodal_attention import TrimodalAttentionFusion  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FOLDER = {"audio": "Audio", "vision": "Vision", "eeg": "EEG"}
SUFFIX = {"audio": "aud", "vision": "vis", "eeg": "eeg"}
MODALITIES = ["audio", "vision", "eeg"]
N_CLASSES = 5
FEAT_DIMS = {"audio": 768, "vision": 768, "eeg": 960}

# Encoder budgets. Per-subject training saw 280 trials; a fold sees ~13,200
# (47x more), so far fewer passes are needed. These are ESTIMATES -- run
# --folds 0 first and check the inner-validation curves before committing all
# five folds to a queue.
AUD_EPOCHS_FROZEN, AUD_EPOCHS_FT = 2, 2
# Raised from 1+1 after fold 0 put vision at 0.4583 cross-subject against
# 0.746 within-subject, while audio held at 0.5697 vs 0.571. At 1+1 -- one
# head-only epoch then one backbone epoch at lr=5e-6 -- the ViT is close to a
# frozen-backbone linear probe, so underfitting is a live alternative to
# "facial features do not transfer". Only the epoch counts change; the
# learning rates stay as in P2 so any improvement is attributable.
VIS_EPOCHS_FROZEN, VIS_EPOCHS_FT = 3, 2
# EEGNet overfits well before these budgets at fold scale: at 60 epochs
# inner-val was loss 1.4469 / acc 0.3833, at 200 it was 1.6389 / 0.3567 while
# training loss kept falling. Rather than tune the count to one noisy reading,
# EEG_EPOCHS is now an UPPER BOUND and the best inner-validation epoch is kept
# (see _train_eegnet_with_selection). Patience stops the run once inner-val
# has not improved for EEG_PATIENCE epochs. The cap is 150 because the
# observed peak sits near epoch 90 (acc 0.3942) with a noisy, non-monotonic
# curve -- 60 (0.3833) and 200 (0.3567) both sit below it, so no fixed count
# is reliable and the cap only needs enough headroom past the peak.
EEG_EPOCHS = 150
EEG_PATIENCE = 25

# Fusion head budgets.
FUSION_EPOCHS = 40
FUSION_BATCH = 64
FUSION_LR = 1e-3
FUSION_WD = 1e-4
DROP_P = 0.5

# Subjects carved from the training set for model selection. Kept at the
# SUBJECT level so inner validation matches the outer protocol.
INNER_VAL_SUBJECTS = 3

# Vision RAM guard.
#
# The binding constraint is NOT the raw uint8 frames -- it is what
# ImageClassifierTrainer.preprocess_images holds. That method resizes every
# frame to 224x224 float32 and keeps the entire result in one CPU tensor:
#
#     224 * 224 * 3 * 4 bytes = 602 KB per frame, ~64x the 9.4 KB raw frame
#
# Per-subject training was 7,000 frames = ~4 GB, which is the OOM documented
# in Chapter 7. A fold is ~33 subjects; every frame would be 330k frames =
# ~190 GB. So the budget below is applied to the PREPROCESSED size, and is
# met by subsampling frames per clip (and, if still too large, clips per
# subject). The ViT trains per-frame, so frames are near-interchangeable
# training examples; Stage B extraction always uses all 25 frames.
#
# The budget is the size of the RESULT. preprocess_images used to peak at 2x
# that (a chunk list plus its concatenation) and killed fold 2 at 37,200
# frames; it now fills a preallocated tensor, so peak == result. Both the
# training and inner-validation tensors are resident at once, so leave headroom.
#
# Raise this on a high-RAM runtime via --vis-budget-gb to train on more frames.
# Keep it IDENTICAL across folds: changing it changes how much data the vision
# encoder sees, and folds trained on different amounts are not comparable.
PREPROC_FRAME_BYTES = 224 * 224 * 3 * 4
VIS_PREPROC_BUDGET_GB = 8.0
CLIPS_PER_SUBJECT = 400

STATE_DIR = CHECKPOINTS / "cv5_state_dicts"
FEAT_DIR = CHECKPOINTS / "cv5_features"
LOGITS_DIR = CHECKPOINTS / "cv5_fusion_logits"
FUSION_STATE_DIR = CHECKPOINTS / "cv5_fusion_heads"
RESULTS_CSV = RESULTS / "cv5_fusion.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

CSV_FIELDS = ["fold", "subject", "variant", "test_acc", "n_trials", "standardized"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pkl(sub: int, modality: str):
    path = EAV_PICKLES / FOLDER[modality] / f"subject_{sub:02d}_{SUFFIX[modality]}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def load_all_trials(sub: int, modality: str) -> tuple[np.ndarray, np.ndarray]:
    """All 400 trials for a subject, train and test halves concatenated.

    The within-subject 280/120 split is meaningless under a subject-wise
    protocol: a training subject contributes everything they have, and a
    held-out subject is evaluated on everything they have.
    """
    tr_x, tr_y, te_x, te_y = load_pkl(sub, modality)
    x = np.concatenate([np.asarray(tr_x), np.asarray(te_x)], axis=0)
    y = np.concatenate([np.asarray(tr_y), np.asarray(te_y)], axis=0)
    return x, y


def _labels_1d(y: np.ndarray) -> np.ndarray:
    """EAV pickles store labels one-hot in some modalities, dense in others."""
    y = np.asarray(y)
    return y.argmax(axis=1) if y.ndim == 2 else y.astype(np.int64)


def plan_vision_sampling(n_subjects: int, budget_gb: float,
                         clips_per_subject: int = CLIPS_PER_SUBJECT
                         ) -> tuple[int, int]:
    """Choose (frames_per_clip, clips_per_subject) fitting the CPU RAM budget.

    Budgets the PREPROCESSED tensor (224x224 float32), which is what actually
    OOMs -- see the PREPROC_FRAME_BYTES note above. Returns a plan that must be
    applied identically to the fit and inner-validation sets: the trainer reads
    frame_per_sample off tr_x.shape[1] and applies it to both, so mismatched
    frame counts desynchronise the validation labels.
    """
    max_frames = max(int(budget_gb * (1024 ** 3) / PREPROC_FRAME_BYTES), 1)
    total_clips = max(n_subjects * clips_per_subject, 1)
    frames_per_clip = int(np.clip(max_frames // total_clips, 1, 25))
    keep_clips = int(np.clip((max_frames // frames_per_clip) // max(n_subjects, 1),
                             1, clips_per_subject))
    return frames_per_clip, keep_clips


def pool_subjects(subs: list[int], modality: str,
                  frames_per_clip: int | None = None,
                  clips_per_subject: int | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate every trial from `subs` for one modality.

    Vision keeps its (n_clips, n_frames, H, W, 3) structure and per-CLIP
    labels: ImageClassifierTrainer flattens clips into frames itself and
    repeats the labels by frame_per_sample. Pre-flattening here makes it
    iterate rows of a frame instead.
    """
    xs, ys = [], []
    for i, sub in enumerate(subs):
        x, y = load_all_trials(sub, modality)
        y = _labels_1d(y)

        if modality == "vision":
            if clips_per_subject is not None and clips_per_subject < len(x):
                keep = np.linspace(0, len(x) - 1, clips_per_subject).astype(int)
                x, y = x[keep], y[keep]
            if frames_per_clip is not None and frames_per_clip < x.shape[1]:
                idx = np.linspace(0, x.shape[1] - 1, frames_per_clip).astype(int)
                x = x[:, idx]

        xs.append(x)
        ys.append(y)
        if (i + 1) % 10 == 0:
            print(f"    loaded {i + 1}/{len(subs)} subjects")

    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


# ---------------------------------------------------------------------------
# Stage A -- fold-wise encoder training
# ---------------------------------------------------------------------------

def _inner_split(subs: list[int]) -> tuple[list[int], list[int]]:
    """Split training subjects into (fit, inner-validation) subject lists."""
    rng = np.random.default_rng(SEED)
    shuffled = list(subs)
    rng.shuffle(shuffled)
    return sorted(shuffled[INNER_VAL_SUBJECTS:]), sorted(shuffled[:INNER_VAL_SUBJECTS])


def train_fold_audio(fold: int, fit: list[int], val: list[int]) -> Path:
    out = STATE_DIR / f"fold{fold}_audio.pt"
    if out.exists():
        print(f"  [audio] cache hit: {out.name}")
        return out
    from Transformer_torch.Transformer_Audio import AudioModelTrainer

    print(f"  [audio] pooling {len(fit)} fit + {len(val)} val subjects...")
    tr_x, tr_y = pool_subjects(fit, "audio")
    va_x, va_y = pool_subjects(val, "audio")
    print(f"  [audio] training on {len(tr_x):,} trials...")

    trainer = AudioModelTrainer(
        [tr_x, tr_y, va_x, va_y], model_path=HF_AUDIO_MODEL,
        sub=f"fold{fold}", num_classes=N_CLASSES, lr=5e-4, batch_size=8,
    )
    trainer.train(epochs=AUD_EPOCHS_FROZEN, lr=5e-4, freeze=True)
    trainer.train(epochs=AUD_EPOCHS_FT, lr=5e-6, freeze=False)
    torch.save(_unwrap(trainer.model), out)
    print(f"  [audio] saved {out.name}")
    return out


@torch.no_grad()
def _eval_vision_trainer(trainer) -> float:
    """Inner-validation accuracy of a vision trainer's current weights.

    ImageClassifierTrainer computes this internally each epoch but neither
    returns it nor keeps the best state, so train() leaves whatever the final
    epoch produced. Recomputed here so the epoch loop can select.
    """
    model = trainer.model
    model.eval()
    correct = total = 0
    for batch in trainer.test_dataloader:
        px, y = [b.to(trainer.device) for b in batch]
        out = model(px)
        logits = out.logits if hasattr(out, "logits") else out
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def _train_vision_with_selection(trainer, phases: list[tuple[int, float, bool]]) -> dict:
    """Run the trainer one epoch at a time, keeping the best inner-val weights.

    `phases` is [(n_epochs, lr, freeze), ...]. Calling train(epochs=1, ...)
    repeatedly is equivalent to one multi-epoch call -- the optimizer lives on
    the trainer, and each call just re-applies the lr and freeze flags -- but it
    hands us the epoch boundary, which is where selection has to happen.

    Selection ranges over FINE-TUNE epochs only. The frozen phase trains just
    the re-initialised classifier head on a fixed backbone, so a model from it
    is a linear probe -- a different architecture from the one under
    evaluation, not merely an earlier epoch of it. Letting it compete is what
    made fold 0 save a frozen-backbone encoder whose inner-validation accuracy
    beat the fine-tuned one by 0.25 pp (inside the ~0.8 pp noise floor of a
    3,600-frame validation set) while scoring 10 pp worse on held-out subjects.

    Returns the best state dict (already unwrapped from DataParallel).
    """
    best_acc, best_where = -1.0, None
    best_state = None
    for n_epochs, lr, freeze in phases:
        tag = "frozen" if freeze else "finetune"
        for e in range(n_epochs):
            trainer.train(epochs=1, lr=lr, freeze=freeze)
            acc = _eval_vision_trainer(trainer)
            where = f"{tag} {e + 1}/{n_epochs}"

            if freeze:
                # Warmup: reported for visibility, never selectable.
                print(f"  [vision] {where}: inner-val acc {acc:.4f}  (warmup)",
                      flush=True)
                continue

            marker = ""
            if acc > best_acc:
                best_acc, best_where = acc, where
                best_state = {k: v.detach().clone()
                              for k, v in _unwrap(trainer.model).items()}
                marker = "  <- best"
            print(f"  [vision] {where}: inner-val acc {acc:.4f}{marker}", flush=True)

    if best_state is None:
        # No fine-tune phase ran; fall back to the current weights.
        best_state = {k: v.detach().clone() for k, v in _unwrap(trainer.model).items()}
        best_where, best_acc = "final (no fine-tune phase)", _eval_vision_trainer(trainer)
    print(f"  [vision] selected {best_where} (inner-val acc {best_acc:.4f})")
    return best_state


def train_fold_vision(fold: int, fit: list[int], val: list[int]) -> Path:
    out = STATE_DIR / f"fold{fold}_vision.pt"
    if out.exists():
        print(f"  [vision] cache hit: {out.name}")
        return out
    from Transformer_torch.Transformer_Vision import ImageClassifierTrainer

    # One sampling plan, applied to fit AND val: the trainer derives
    # frame_per_sample from tr_x.shape[1] and reuses it for the val labels.
    fpc, cps = plan_vision_sampling(len(fit), VIS_PREPROC_BUDGET_GB)
    n_frames = len(fit) * cps * fpc
    print(f"  [vision] RAM plan: {fpc}/25 frames per clip, {cps}/{CLIPS_PER_SUBJECT} "
          f"clips per subject")
    print(f"           -> {n_frames:,} training frames, "
          f"~{n_frames * PREPROC_FRAME_BYTES / 1024**3:.1f} GB preprocessed "
          f"(budget {VIS_PREPROC_BUDGET_GB:.1f} GB)")

    print(f"  [vision] pooling {len(fit)} fit + {len(val)} val subjects...")
    tr_x, tr_y = pool_subjects(fit, "vision", frames_per_clip=fpc, clips_per_subject=cps)
    va_x, va_y = pool_subjects(val, "vision", frames_per_clip=fpc, clips_per_subject=cps)
    print(f"  [vision] training on {len(tr_x):,} clips x {fpc} frames "
          f"= {len(tr_x) * fpc:,} frames...")

    trainer = ImageClassifierTrainer(
        [tr_x, tr_y, va_x, va_y], model_path=HF_VISION_MODEL,
        sub=f"fold{fold}", num_labels=N_CLASSES, lr=5e-5, batch_size=32,
    )
    best_state = _train_vision_with_selection(trainer, [
        (VIS_EPOCHS_FROZEN, 5e-4, True),
        (VIS_EPOCHS_FT, 5e-6, False),
    ])
    torch.save(best_state, out)
    print(f"  [vision] saved {out.name}")
    return out


def _train_eegnet_with_selection(model, tr_x, tr_y, va_x, va_y,
                                 epochs: int, lr: float = 1e-5,
                                 batch_size: int = 32):
    """Train EEGNet, keeping the best inner-validation epoch.

    Replicates Trainer_uni's recipe exactly -- Adam at `lr`, CrossEntropyLoss,
    the same batch size, and model.train() re-entered every epoch (the Day-2
    fix for validate() leaving the model in eval mode). The one difference is
    that Trainer_uni.validate() only prints, and train_fold_eeg then saved
    whatever state the FINAL epoch left behind. At fold scale EEGNet overfits
    long before the budget, so the final epoch is the wrong model to keep.

    Selection uses inner-validation SUBJECTS, the same ones held out of this
    encoder's training -- matching the discipline used for the fusion heads.
    """
    from torch.utils.data import DataLoader, TensorDataset

    tr_ds = TensorDataset(tr_x, torch.as_tensor(tr_y, dtype=torch.long))
    va_ds = TensorDataset(va_x, torch.as_tensor(va_y, dtype=torch.long))
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False)

    model = model.to(DEVICE)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=lr)

    best_acc, best_loss, best_epoch, since_best = -1.0, float("inf"), -1, 0
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for epoch in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        correct, total, vloss = 0, 0, 0.0
        with torch.no_grad():
            for xb, yb in va_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                vloss += crit(out, yb).item() * yb.size(0)
                correct += (out.argmax(dim=1) == yb).sum().item()
                total += yb.size(0)
        acc, vloss = correct / max(total, 1), vloss / max(total, 1)

        if acc > best_acc:
            best_acc, best_loss, best_epoch, since_best = acc, vloss, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since_best += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    [eeg] epoch {epoch + 1:3d}/{epochs}  "
                  f"val_loss={vloss:.4f}  val_acc={acc:.4f}  (best {best_acc:.4f})")

        if since_best >= EEG_PATIENCE:
            print(f"    [eeg] early stop at epoch {epoch + 1}: "
                  f"no improvement for {EEG_PATIENCE} epochs")
            break

    model.load_state_dict(best_state)
    print(f"  [eeg] selected epoch {best_epoch}/{epochs}  "
          f"(inner-val acc {best_acc:.4f}, loss {best_loss:.4f})")
    return model


def train_fold_eeg(fold: int, fit: list[int], val: list[int]) -> Path:
    out = STATE_DIR / f"fold{fold}_eeg.pt"
    if out.exists():
        print(f"  [eeg] cache hit: {out.name}")
        return out
    from CNN_torch.EEGNet_tor import EEGNet_tor

    print(f"  [eeg] pooling {len(fit)} fit + {len(val)} val subjects...")
    tr_x, tr_y = pool_subjects(fit, "eeg")
    va_x, va_y = pool_subjects(val, "eeg")
    tr_x = torch.from_numpy(tr_x).float().unsqueeze(1)
    va_x = torch.from_numpy(va_x).float().unsqueeze(1)
    print(f"  [eeg] training on {len(tr_x):,} trials...")

    model = EEGNet_tor(nb_classes=N_CLASSES, D=8, F2=64, Chans=30,
                       kernLength=300, Samples=500, dropoutRate=0.5)
    model = _train_eegnet_with_selection(model, tr_x, tr_y, va_x, va_y,
                                         epochs=EEG_EPOCHS)
    torch.save(_unwrap(model), out)
    print(f"  [eeg] saved {out.name}")
    return out


def _unwrap(model) -> dict:
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def stage_a(fold: int) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    fit, val = _inner_split(train_subjects(fold))
    print(f"\n--- Stage A: fold {fold} encoders ---")
    print(f"  fit subjects   : {len(fit)}  {fit}")
    print(f"  inner-val      : {len(val)}  {val}  (model selection only)")
    print(f"  held-out (test): {test_subjects(fold)}  -- never touched here")
    train_fold_audio(fold, fit, val)
    train_fold_vision(fold, fit, val)
    train_fold_eeg(fold, fit, val)


# ---------------------------------------------------------------------------
# Stage B -- feature + logit extraction with fold-k encoders
# ---------------------------------------------------------------------------

def stage_b(fold: int) -> None:
    from fusion.feature_extractors import (AudioFeatureExtractor,
                                           EEGFeatureExtractor,
                                           VisionFeatureExtractor)
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n--- Stage B: fold {fold} feature extraction ---")

    todo = [s for s in SUBJECTS if not (FEAT_DIR / f"fold{fold}_sub{s:02d}.npz").exists()]
    if not todo:
        print("  all 42 subjects cached; nothing to do")
        return
    print(f"  {len(todo)} subjects to extract")

    aud = AudioFeatureExtractor(HF_AUDIO_MODEL,
                                state_dict_path=STATE_DIR / f"fold{fold}_audio.pt")
    vis = VisionFeatureExtractor(HF_VISION_MODEL,
                                 state_dict_path=STATE_DIR / f"fold{fold}_vision.pt")
    eeg = EEGFeatureExtractor(state_dict_path=STATE_DIR / f"fold{fold}_eeg.pt")

    for i, sub in enumerate(todo, 1):
        t0 = time.time()
        out = FEAT_DIR / f"fold{fold}_sub{sub:02d}.npz"

        xa, ya = load_all_trials(sub, "audio")
        xv, _ = load_all_trials(sub, "vision")
        xe, _ = load_all_trials(sub, "eeg")

        fa, la = aud.extract(xa, return_logits=True)
        fv, lv = vis.extract(xv, return_logits=True)
        fe, le = eeg.extract(xe, return_logits=True)

        np.savez(out,
                 audio=fa, vision=fv, eeg=fe,
                 logits_audio=la, logits_vision=lv, logits_eeg=le,
                 y=_labels_1d(ya))
        print(f"  [{i}/{len(todo)}] sub{sub:02d} -> {out.name} "
              f"({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------
# Stage C -- fusion variants
# ---------------------------------------------------------------------------

class ConcatMLPWrapped(nn.Module):
    """ConcatMLP returns a bare tensor; give it the dict contract the rest of
    the pipeline (and eval_mode) expects."""

    def __init__(self, **dims):
        super().__init__()
        self.net = ConcatMLP(**dims)

    def forward(self, xa, xv, xe):
        return {"logits": self.net(xa, xv, xe)}


def load_fold_features(fold: int, subs: list[int]) -> dict[str, np.ndarray]:
    """Stack cached features/logits/labels over `subs`, plus a subject index."""
    acc: dict[str, list] = {k: [] for k in
                            ["audio", "vision", "eeg", "logits_audio",
                             "logits_vision", "logits_eeg", "y", "subject"]}
    for sub in subs:
        with np.load(FEAT_DIR / f"fold{fold}_sub{sub:02d}.npz") as z:
            for k in acc:
                if k == "subject":
                    acc[k].append(np.full(len(z["y"]), sub))
                else:
                    acc[k].append(z[k])
    return {k: np.concatenate(v, axis=0) for k, v in acc.items()}


def softmax_np(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def fit_standardizer(train: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-modality mean/std fitted on TRAINING SUBJECTS ONLY."""
    stats = {}
    for m in MODALITIES:
        mu = train[m].mean(axis=0, keepdims=True)
        sd = train[m].std(axis=0, keepdims=True) + 1e-6
        stats[m] = (mu, sd)
    return stats


def apply_standardizer(d: dict, stats: dict) -> dict[str, np.ndarray]:
    return {m: (d[m] - stats[m][0]) / stats[m][1] for m in MODALITIES}


def subject_standardize(d: dict, modalities: tuple[str, ...] = tuple(MODALITIES)
                        ) -> dict[str, np.ndarray]:
    """Z-score each listed modality's features within each participant separately.

    The default standardiser fits one mean and variance on the training
    participants and applies it to everyone, so a held-out participant is
    normalised by other people's statistics and their own offset survives into
    the fusion input. That offset is subject identity, which Chapter 7 shows is
    the component that does not transfer.

    Normalising each participant by their own statistics removes that offset
    without using any labels. It is applied to training and held-out
    participants alike, so both sit in the same space.

    NOTE: this is TRANSDUCTIVE. Computing a participant's statistics requires a
    batch of their trials, so the method assumes a short unlabeled calibration
    session per new user rather than one-shot inference on a single clip. The
    assumption is standard in cross-subject EEG work but is weaker than the
    default protocol and must be reported as such.

    `modalities` restricts the per-participant treatment to a subset; the
    per-modality ablation (--subject-norm-only) uses it to ask whose identity
    offset is responsible. Only the listed modalities are returned.
    """
    out = {}
    for m in modalities:
        X = d[m].astype(np.float32, copy=True)
        for s in np.unique(d["subject"]):
            mask = d["subject"] == s
            mu = X[mask].mean(axis=0, keepdims=True)
            sd = X[mask].std(axis=0, keepdims=True) + 1e-6
            X[mask] = (X[mask] - mu) / sd
        out[m] = X
    return out


def to_tensors(feats: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {m: torch.from_numpy(feats[m]).float().to(DEVICE) for m in MODALITIES}


def train_fusion_head(model: nn.Module, train_x: dict, train_y: torch.Tensor,
                      val_x: dict, val_y: torch.Tensor,
                      use_dropout: bool) -> nn.Module:
    """Train one fusion head, selecting the best epoch on inner-validation.

    The validation subjects are the same ones Stage A held out of encoder
    training, so their features were produced by encoders that never saw them
    -- exactly how the test fold's features are produced. Validating on the
    fit subjects instead would be optimistically biased, because their
    features come from encoders trained on them.

    Without this selection the head simply ran FUSION_EPOCHS to completion and
    drove training loss to ~0. Cross-subject, that overfits the training
    subjects' feature geometry and penalises every trained variant relative to
    parameter-free late fusion, which has nothing to overfit.

    Validation is always full-modality, even for the dropout variant: dropout
    is a training-time regulariser, not part of the selection criterion.
    """
    model = model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=FUSION_LR, weight_decay=FUSION_WD)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FUSION_EPOCHS)
    crit = nn.CrossEntropyLoss()
    n = train_y.size(0)

    best_acc, best_epoch = -1.0, -1
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    for epoch in range(FUSION_EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        total = 0.0
        for i in range(0, n, FUSION_BATCH):
            idx = perm[i:i + FUSION_BATCH]
            xa, xv, xe = (train_x["audio"][idx], train_x["vision"][idx],
                          train_x["eeg"][idx])
            if use_dropout:
                xa, xv, xe = softhard_dropout(xa, xv, xe, p=DROP_P)
            opt.zero_grad()
            loss = crit(model(xa, xv, xe)["logits"], train_y[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()

        val_acc, _, _ = eval_mode(model, val_x, val_y, False, False, False)
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"      epoch {epoch + 1:3d}/{FUSION_EPOCHS}  "
                  f"loss={total / n:.4f}  val={val_acc:.4f}")

    model.load_state_dict(best_state)
    print(f"      selected epoch {best_epoch}/{FUSION_EPOCHS} "
          f"(inner-val acc {best_acc:.4f})")
    return model


def stage_c(fold: int, standardize: bool, rows: list[dict],
              subject_norm: bool = False,
              subject_norm_only: tuple[str, ...] = ()) -> None:
    """Train and score every fusion variant for one fold.

    subject_norm        every modality standardised per participant.
    subject_norm_only   per-modality ablation: only these modalities are
                        standardised per participant; the rest keep the default
                        per-fold statistics. Mutually exclusive with subject_norm.
    """
    if subject_norm and subject_norm_only:
        raise ValueError("subject_norm and subject_norm_only are mutually exclusive")
    print(f"\n--- Stage C: fold {fold} fusion variants ---")
    tr_subs, te_subs = train_subjects(fold), test_subjects(fold)
    fit_subs, val_subs = _inner_split(tr_subs)
    train = load_fold_features(fold, tr_subs)
    test = load_fold_features(fold, te_subs)
    print(f"  fit   {len(fit_subs)} subjects   inner-val {len(val_subs)} {val_subs}")
    print(f"  train {len(train['y']):,} trials / {len(tr_subs)} subjects")
    print(f"  test  {len(test['y']):,} trials / {len(te_subs)} subjects")

    stats = None
    if subject_norm_only:
        stats = fit_standardizer(train)
        train_f, test_f = apply_standardizer(train, stats), apply_standardizer(test, stats)
        train_f.update(subject_standardize(train, subject_norm_only))
        test_f.update(subject_standardize(test, subject_norm_only))
        print(f"  features standardized PER PARTICIPANT for {list(subject_norm_only)}; "
              f"per fold for the rest")
    elif subject_norm:
        train_f, test_f = subject_standardize(train), subject_standardize(test)
        print("  features standardized PER PARTICIPANT (transductive; see "
              "subject_standardize docstring)")
    elif standardize:
        # Statistics come from the whole training fold -- standardisation is
        # preprocessing, not model selection, and the test fold is untouched.
        stats = fit_standardizer(train)
        train_f, test_f = apply_standardizer(train, stats), apply_standardizer(test, stats)
        print("  features standardized (statistics fitted on train subjects only)")
    else:
        train_f = {m: train[m] for m in MODALITIES}
        test_f = {m: test[m] for m in MODALITIES}

    train_all, test_x = to_tensors(train_f), to_tensors(test_f)
    train_all_y = torch.from_numpy(train["y"]).long().to(DEVICE)
    test_y = torch.from_numpy(test["y"]).long().to(DEVICE)

    # Split the training fold into fit / inner-validation by SUBJECT, matching
    # the Stage A split so validation features come from encoders that never
    # saw those subjects.
    fit_m = torch.from_numpy(np.isin(train["subject"], fit_subs)).to(DEVICE)
    val_m = torch.from_numpy(np.isin(train["subject"], val_subs)).to(DEVICE)
    fit_x = {m: train_all[m][fit_m] for m in MODALITIES}
    val_x = {m: train_all[m][val_m] for m in MODALITIES}
    fit_y, val_y = train_all_y[fit_m], train_all_y[val_m]
    print(f"        -> fit {fit_y.numel():,} trials, inner-val {val_y.numel():,} trials")

    def record(variant: str, preds: np.ndarray) -> None:
        for sub in te_subs:
            m = test["subject"] == sub
            rows.append({
                "fold": fold, "subject": sub, "variant": variant,
                "test_acc": float((preds[m] == test["y"][m]).mean()),
                "n_trials": int(m.sum()),
                "standardized": 3 if subject_norm_only else (2 if subject_norm
                                                             else int(standardize)),
            })
        acc = float((preds == test["y"]).mean())
        print(f"    {variant:14s} pooled acc = {acc:.4f}")

    # --- Per-modality baselines (no training; the encoders' own heads) ---
    # Cross-subject answer to RQ1, and the diagnostic for which modality is
    # carrying the fusion result. Free: the logits are already cached.
    for m in MODALITIES:
        record(f"{m}_only", test[f"logits_{m}"].argmax(axis=-1))

    # --- V1: naive late fusion (no training) ---
    probs = np.stack([softmax_np(test[f"logits_{m}"]) for m in MODALITIES])
    record("naive_late", probs.mean(axis=0).argmax(axis=-1))

    dims = {"audio_dim": FEAT_DIMS["audio"], "vision_dim": FEAT_DIMS["vision"],
            "eeg_dim": FEAT_DIMS["eeg"]}

    # --- V2: cross-attention ---
    print("    training cross_attn...")
    torch.manual_seed(SEED)
    m2 = train_fusion_head(TrimodalAttentionFusion(**dims), fit_x, fit_y,
                           val_x, val_y, False)
    record("cross_attn", eval_mode(m2, test_x, test_y, False, False, False)[1])

    # --- V3: concat-MLP ---
    print("    training concat_mlp...")
    torch.manual_seed(SEED)
    m3 = train_fusion_head(ConcatMLPWrapped(**dims), fit_x, fit_y,
                           val_x, val_y, False)
    record("concat_mlp", eval_mode(m3, test_x, test_y, False, False, False)[1])

    # --- V4/V5: softhard modality dropout, evaluated full and EEG-zeroed ---
    print("    training dropout (softhard)...")
    torch.manual_seed(SEED)
    m4 = train_fusion_head(TrimodalAttentionFusion(**dims), fit_x, fit_y,
                           val_x, val_y, True)
    record("dropout_full", eval_mode(m4, test_x, test_y, False, False, False)[1])
    record("dropout_av", eval_mode(m4, test_x, test_y, False, False, True)[1])

    # Persist per-head logits AND the trained heads themselves.
    #
    # The logits previously went out with m4 (the dropout model) stored under
    # the key "cross_attn", so any consumer reading that key silently got the
    # wrong model. Each head now writes under its own name.
    #
    # The heads are saved because downstream analyses -- the cross-subject
    # suppression matrix, and the cross-modal conflict experiment, which feeds
    # the model mismatched modality pairs -- need to run the trained fusion
    # models on inputs that are not the test set. Re-training them later would
    # not reproduce these exact weights.
    LOGITS_DIR.mkdir(parents=True, exist_ok=True)
    FUSION_STATE_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"fold{fold}" + ("_subjnorm" if subject_norm else "") + (
        f"_subjnorm_{'+'.join(subject_norm_only)}" if subject_norm_only else "")

    heads = {"cross_attn": m2, "concat_mlp": m3, "dropout": m4}
    payload = {"y": test["y"], "subject": test["subject"]}
    with torch.no_grad():
        for name, model in heads.items():
            model.eval()
            out = model(test_x["audio"], test_x["vision"], test_x["eeg"])
            payload[name] = out["logits"].cpu().numpy()
        # The zero-EEG demo path, from the dropout-trained head.
        payload["dropout_av"] = m4(
            test_x["audio"], test_x["vision"],
            torch.zeros_like(test_x["eeg"]))["logits"].cpu().numpy()
    for m in MODALITIES:
        payload[f"logits_{m}"] = test[f"logits_{m}"]

    np.savez(LOGITS_DIR / f"{tag}.npz", **payload)
    for name, model in heads.items():
        torch.save(model.state_dict(), FUSION_STATE_DIR / f"{tag}_{name}.pt")
    if stats is not None:
        # Conflict trials must be standardised with the same statistics.
        np.savez(FUSION_STATE_DIR / f"{tag}_standardizer.npz",
                 **{f"{m}_{k}": v for m in MODALITIES
                    for k, v in zip(("mu", "sd"), stats[m])})
    print(f"    saved heads + logits -> {tag}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def append_rows(rows: list[dict]) -> None:
    if not rows:
        return
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"\n  wrote {len(rows)} rows -> {RESULTS_CSV}")


def summarise() -> None:
    if not RESULTS_CSV.exists():
        return
    with open(RESULTS_CSV) as f:
        rows = list(csv.DictReader(f))

    # The CSV is append-only, so re-running a fold leaves the superseded rows
    # behind. Keep the most recent row per (fold, subject, variant) and say so
    # -- silently averaging both protocols together would be a wrong number
    # that looks perfectly plausible.
    latest: dict[tuple, dict] = {}
    for r in rows:
        latest[(r["fold"], r["subject"], r["variant"])] = r
    n_superseded = len(rows) - len(latest)
    if n_superseded:
        print(f"\nNOTE: {n_superseded} superseded row(s) ignored "
              f"(re-run folds); using the most recent per subject/variant.")

    by: dict[str, list[float]] = {}
    for r in latest.values():
        by.setdefault(r["variant"], []).append(float(r["test_acc"]))

    print("\n" + "=" * 64)
    print("CROSS-SUBJECT RESULTS (5-fold, subject-wise)")
    print("=" * 64)
    print(f"  {'variant':16s} {'n':>4s} {'mean':>8s} {'std':>8s}")
    for v, accs in sorted(by.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"  {v:16s} {len(accs):4d} {np.mean(accs):8.4f} {np.std(accs):8.4f}")
    n = max(len(a) for a in by.values())
    if n < len(SUBJECTS):
        print(f"\n  NOTE: {n}/{len(SUBJECTS)} subjects done -- partial run.")


def _code_version() -> str:
    """Short git SHA of the running code, plus a dirty flag.

    Printed at startup because this pipeline is authored locally and executed
    in a remote runtime that pulls from GitHub. A fix that is committed but not
    pushed, or pushed but not pulled, reproduces the original failure exactly
    -- which has cost real GPU time. Comparing this line against `git log -1`
    locally answers "am I running the fix?" without reading log formats.
    """
    import subprocess
    root = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        if sha.returncode != 0:
            return "unknown (not a git checkout)"
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=10)
        flag = " +local-changes" if dirty.stdout.strip() else ""
        return sha.stdout.strip() + flag
    except Exception:
        return "unknown"


def main() -> int:
    global VIS_PREPROC_BUDGET_GB

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", default=",".join(str(k) for k in range(N_FOLDS)),
                    help="comma-separated fold indices, e.g. 0,1")
    ap.add_argument("--stages", default="A,B,C", help="subset of A,B,C")
    ap.add_argument("--no-standardize", action="store_true",
                    help="skip per-fold feature standardization")
    ap.add_argument("--vis-budget-gb", type=float, default=VIS_PREPROC_BUDGET_GB,
                    help="CPU RAM budget for the preprocessed ViT training "
                         "tensor (224x224 float32). Raise on a high-RAM runtime "
                         "to train vision on more frames.")
    ap.add_argument("--subject-norm", action="store_true",
                    help="standardize features per participant instead of per fold "
                         "(transductive; writes to a separate results CSV so the "
                         "default-protocol numbers are left intact)")
    ap.add_argument("--subject-norm-only", default="",
                    help="per-modality ablation: comma list of modalities "
                         "(audio,vision,eeg) standardized per participant; the rest "
                         "per fold. Writes cv5_fusion_subjectnorm_<mods>.csv")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    VIS_PREPROC_BUDGET_GB = args.vis_budget_gb
    only = tuple(m.strip() for m in args.subject_norm_only.split(",") if m.strip())
    if set(only) - set(MODALITIES):
        ap.error(f"--subject-norm-only: unknown modality in {only}")
    if only and args.subject_norm:
        ap.error("--subject-norm and --subject-norm-only are mutually exclusive")

    global RESULTS_CSV
    if args.subject_norm:
        # A separate file is mandatory, not cosmetic: rows would otherwise share
        # (fold, subject, variant) keys with the default run and the supersede
        # rule in summarise() would silently discard one protocol.
        RESULTS_CSV = RESULTS / "cv5_fusion_subjectnorm.csv"
        print(f"subject-norm protocol -> {RESULTS_CSV.name}")
    elif only:
        RESULTS_CSV = RESULTS / f"cv5_fusion_subjectnorm_{'+'.join(only)}.csv"
        print(f"per-modality subject-norm {list(only)} -> {RESULTS_CSV.name}")

    validate()
    if args.summary_only:
        summarise()
        return 0

    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    stages = {s.strip().upper() for s in args.stages.split(",")}

    print(f"code version: {_code_version()}")
    print(f"device={DEVICE}  folds={folds}  stages={sorted(stages)}")
    print(f"standardize={not args.no_standardize}")
    if DEVICE == "cpu" and stages & {"A", "B"}:
        print("\nWARNING: stages A and B need a GPU. On CPU this will not finish.")

    rows: list[dict] = []
    for fold in folds:
        t0 = time.time()
        print(f"\n{'=' * 64}\nFOLD {fold}\n{'=' * 64}")
        if "A" in stages:
            stage_a(fold)
        if "B" in stages:
            stage_b(fold)
        if "C" in stages:
            stage_c(fold, not args.no_standardize, rows,
                    subject_norm=args.subject_norm, subject_norm_only=only)
            append_rows(rows)
            rows = []
        print(f"\nfold {fold} done in {(time.time() - t0) / 60:.1f} min")

    summarise()
    return 0


if __name__ == "__main__":
    sys.exit(main())
