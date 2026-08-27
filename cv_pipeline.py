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
    naive_late      mean of per-modality softmax  (no training)
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
VIS_EPOCHS_FROZEN, VIS_EPOCHS_FT = 1, 1
EEG_EPOCHS = 60

# Fusion head budgets.
FUSION_EPOCHS = 40
FUSION_BATCH = 64
FUSION_LR = 1e-3
FUSION_WD = 1e-4
DROP_P = 0.5

# Subjects carved from the training set for model selection. Kept at the
# SUBJECT level so inner validation matches the outer protocol.
INNER_VAL_SUBJECTS = 3

# Vision RAM guard. Pooling every frame of ~33 subjects is 330k frames; at
# 224x224 that is ~49 GB and will kill the process. The ViT trains per-frame,
# so evenly-spaced subsampling costs little: even 4 frames/clip gives ~53k
# training frames, 5x more than the per-subject model ever saw. Feature
# extraction in Stage B always uses all 25 frames.
VIS_TRAIN_RAM_BUDGET_GB = 6.0
VIS_MAX_TRAIN_FRAMES = 60_000

STATE_DIR = CHECKPOINTS / "cv5_state_dicts"
FEAT_DIR = CHECKPOINTS / "cv5_features"
LOGITS_DIR = CHECKPOINTS / "cv5_fusion_logits"
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


def choose_frames_per_clip(n_clips: int, frame_nbytes: int) -> int:
    """Pick how many frames per clip to keep for ViT training, RAM-bounded."""
    budget_frames = int(VIS_TRAIN_RAM_BUDGET_GB * (1024 ** 3) / max(frame_nbytes, 1))
    allowed = min(budget_frames, VIS_MAX_TRAIN_FRAMES)
    return int(np.clip(allowed // max(n_clips, 1), 1, 25))


def pool_subjects(subs: list[int], modality: str,
                  subsample_vision: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate every trial from `subs` for one modality."""
    xs, ys = [], []
    frames_per_clip = None

    for i, sub in enumerate(subs):
        x, y = load_all_trials(sub, modality)

        if modality == "vision" and subsample_vision:
            if frames_per_clip is None:
                per_frame = int(np.asarray(x[0][0]).nbytes)
                frames_per_clip = choose_frames_per_clip(len(subs) * len(x), per_frame)
                total = len(subs) * len(x) * frames_per_clip
                print(f"    [vision] RAM guard: keeping {frames_per_clip}/25 frames "
                      f"per clip -> ~{total:,} training frames "
                      f"(~{total * per_frame / 1024**3:.1f} GB)")
            idx = np.linspace(0, x.shape[1] - 1, frames_per_clip).astype(int)
            x = x[:, idx]
            # Frame-level training: every frame is its own example.
            n, f = x.shape[0], x.shape[1]
            x = x.reshape(n * f, *x.shape[2:])
            y = np.repeat(_labels_1d(y), f)
        else:
            y = _labels_1d(y)

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


def train_fold_vision(fold: int, fit: list[int], val: list[int]) -> Path:
    out = STATE_DIR / f"fold{fold}_vision.pt"
    if out.exists():
        print(f"  [vision] cache hit: {out.name}")
        return out
    from Transformer_torch.Transformer_Vision import ImageClassifierTrainer

    print(f"  [vision] pooling {len(fit)} fit + {len(val)} val subjects...")
    tr_x, tr_y = pool_subjects(fit, "vision", subsample_vision=True)
    va_x, va_y = pool_subjects(val, "vision", subsample_vision=True)
    print(f"  [vision] training on {len(tr_x):,} frames...")

    trainer = ImageClassifierTrainer(
        [tr_x, tr_y, va_x, va_y], model_path=HF_VISION_MODEL,
        sub=f"fold{fold}", num_labels=N_CLASSES, lr=5e-5, batch_size=32,
    )
    trainer.train(epochs=VIS_EPOCHS_FROZEN, lr=5e-4, freeze=True)
    trainer.train(epochs=VIS_EPOCHS_FT, lr=5e-6, freeze=False)
    torch.save(_unwrap(trainer.model), out)
    print(f"  [vision] saved {out.name}")
    return out


def train_fold_eeg(fold: int, fit: list[int], val: list[int]) -> Path:
    out = STATE_DIR / f"fold{fold}_eeg.pt"
    if out.exists():
        print(f"  [eeg] cache hit: {out.name}")
        return out
    from CNN_torch.EEGNet_tor import EEGNet_tor, Trainer_uni

    print(f"  [eeg] pooling {len(fit)} fit + {len(val)} val subjects...")
    tr_x, tr_y = pool_subjects(fit, "eeg")
    va_x, va_y = pool_subjects(val, "eeg")
    tr_x = torch.from_numpy(tr_x).float().unsqueeze(1)
    va_x = torch.from_numpy(va_x).float().unsqueeze(1)
    print(f"  [eeg] training on {len(tr_x):,} trials...")

    model = EEGNet_tor(nb_classes=N_CLASSES, D=8, F2=64, Chans=30,
                       kernLength=300, Samples=500, dropoutRate=0.5)
    trainer = Trainer_uni(model=model, data=[tr_x, tr_y, va_x, va_y],
                          lr=1e-5, batch_size=32, num_epochs=EEG_EPOCHS)
    trainer.train()
    torch.save(_unwrap(trainer.model), out)
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


def to_tensors(feats: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {m: torch.from_numpy(feats[m]).float().to(DEVICE) for m in MODALITIES}


def train_fusion_head(model: nn.Module, train_x: dict, train_y: torch.Tensor,
                      use_dropout: bool) -> nn.Module:
    """Train one fusion head. No test-fold signal is used anywhere here."""
    model = model.to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=FUSION_LR, weight_decay=FUSION_WD)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=FUSION_EPOCHS)
    crit = nn.CrossEntropyLoss()
    n = train_y.size(0)

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
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"      epoch {epoch + 1:3d}/{FUSION_EPOCHS}  loss={total / n:.4f}")
    return model


def stage_c(fold: int, standardize: bool, rows: list[dict]) -> None:
    print(f"\n--- Stage C: fold {fold} fusion variants ---")
    tr_subs, te_subs = train_subjects(fold), test_subjects(fold)
    train = load_fold_features(fold, tr_subs)
    test = load_fold_features(fold, te_subs)
    print(f"  train {len(train['y']):,} trials / {len(tr_subs)} subjects")
    print(f"  test  {len(test['y']):,} trials / {len(te_subs)} subjects")

    if standardize:
        stats = fit_standardizer(train)
        train_f, test_f = apply_standardizer(train, stats), apply_standardizer(test, stats)
        print("  features standardized (statistics fitted on train subjects only)")
    else:
        train_f = {m: train[m] for m in MODALITIES}
        test_f = {m: test[m] for m in MODALITIES}

    train_x, test_x = to_tensors(train_f), to_tensors(test_f)
    train_y = torch.from_numpy(train["y"]).long().to(DEVICE)
    test_y = torch.from_numpy(test["y"]).long().to(DEVICE)

    def record(variant: str, preds: np.ndarray) -> None:
        for sub in te_subs:
            m = test["subject"] == sub
            rows.append({
                "fold": fold, "subject": sub, "variant": variant,
                "test_acc": float((preds[m] == test["y"][m]).mean()),
                "n_trials": int(m.sum()), "standardized": int(standardize),
            })
        acc = float((preds == test["y"]).mean())
        print(f"    {variant:14s} pooled acc = {acc:.4f}")

    # --- V1: naive late fusion (no training) ---
    probs = np.stack([softmax_np(test[f"logits_{m}"]) for m in MODALITIES])
    record("naive_late", probs.mean(axis=0).argmax(axis=-1))

    dims = {"audio_dim": FEAT_DIMS["audio"], "vision_dim": FEAT_DIMS["vision"],
            "eeg_dim": FEAT_DIMS["eeg"]}

    # --- V2: cross-attention ---
    print("    training cross_attn...")
    torch.manual_seed(SEED)
    m2 = train_fusion_head(TrimodalAttentionFusion(**dims), train_x, train_y, False)
    record("cross_attn", eval_mode(m2, test_x, test_y, False, False, False)[1])

    # --- V3: concat-MLP ---
    print("    training concat_mlp...")
    torch.manual_seed(SEED)
    m3 = train_fusion_head(ConcatMLPWrapped(**dims), train_x, train_y, False)
    record("concat_mlp", eval_mode(m3, test_x, test_y, False, False, False)[1])

    # --- V4/V5: softhard modality dropout, evaluated full and EEG-zeroed ---
    print("    training dropout (softhard)...")
    torch.manual_seed(SEED)
    m4 = train_fusion_head(TrimodalAttentionFusion(**dims), train_x, train_y, True)
    record("dropout_full", eval_mode(m4, test_x, test_y, False, False, False)[1])
    record("dropout_av", eval_mode(m4, test_x, test_y, False, False, True)[1])

    LOGITS_DIR.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        np.savez(LOGITS_DIR / f"fold{fold}.npz",
                 cross_attn=m4(test_x["audio"], test_x["vision"],
                               test_x["eeg"])["logits"].cpu().numpy(),
                 y=test["y"], subject=test["subject"])


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
    by: dict[str, list[float]] = {}
    for r in rows:
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", default=",".join(str(k) for k in range(N_FOLDS)),
                    help="comma-separated fold indices, e.g. 0,1")
    ap.add_argument("--stages", default="A,B,C", help="subset of A,B,C")
    ap.add_argument("--no-standardize", action="store_true",
                    help="skip per-fold feature standardization")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    validate()
    if args.summary_only:
        summarise()
        return 0

    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    stages = {s.strip().upper() for s in args.stages.split(",")}

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
            stage_c(fold, not args.no_standardize, rows)
            append_rows(rows)
            rows = []
        print(f"\nfold {fold} done in {(time.time() - t0) / 60:.1f} min")

    summarise()
    return 0


if __name__ == "__main__":
    sys.exit(main())
