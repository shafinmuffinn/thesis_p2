"""Concat-MLP fusion baseline (Tier-2 extra experiment).

Train a 2-layer MLP on the concatenated per-modality features cached at
CHECKPOINTS/day5_features/sub{NN}.npz, evaluated across the same 42-subject
within-subject protocol as TrimodalAttentionFusion (day5_pipeline.py).

Outputs RESULTS/concat_mlp_fusion.csv with one row per subject:
    subject,test_acc

Resume-safe: skips any subject already present in the CSV. Designed to be
launched on Colab and left to run unattended (~3-5 min per subject on L4).
"""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# Path config (auto-detect Colab vs local)
#
# NOTE: we cannot rely on `"google.colab" in sys.modules` here because this
# script may be launched with `!python concat_mlp_baseline.py` from a Colab
# cell, which spawns a fresh Python subprocess where google.colab is not
# imported. We instead probe for the mounted Drive folder directly.
# ---------------------------------------------------------------------------
DRIVE_ROOT = Path("/content/drive/MyDrive/Thesis_EAV")
_ON_COLAB_DRIVE = DRIVE_ROOT.exists()

if _ON_COLAB_DRIVE:
    CHECKPOINTS = Path(os.environ.get("CHECKPOINTS", DRIVE_ROOT / "checkpoints"))
    RESULTS = Path(os.environ.get("RESULTS", DRIVE_ROOT / "results"))
else:
    # Local fallback (matches the rest of the project's paths.py convention)
    HERE = Path(__file__).resolve().parent
    CHECKPOINTS = Path(os.environ.get("CHECKPOINTS", HERE / "checkpoints"))
    RESULTS = Path(os.environ.get("RESULTS", HERE / "results"))

FEAT_DIR = CHECKPOINTS / "day5_features"
RESULTS_CSV = RESULTS / "concat_mlp_fusion.csv"

# ---------------------------------------------------------------------------
# Hyperparameters (deliberately match day5_pipeline.py for parity)
# ---------------------------------------------------------------------------
SUBJECTS = list(range(1, 43))
EPOCHS = 80
BATCH = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.1

# MLP hidden sizes (about 1.34 M params, ~60% of TrimodalAttentionFusion)
HIDDEN_1 = 512
HIDDEN_2 = 128
N_CLASSES = 5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class ConcatMLP(nn.Module):
    """Simple feature-concatenation MLP fusion baseline.

    Concatenates audio (768) + vision (768) + EEG (960) = 2496 features,
    passes through two hidden layers with GELU and dropout, and projects
    to the 5-class logit space.
    """

    def __init__(self, audio_dim: int, vision_dim: int, eeg_dim: int):
        super().__init__()
        in_dim = audio_dim + vision_dim + eeg_dim
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, HIDDEN_1),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_2, N_CLASSES),
        )

    def forward(self, x_audio, x_vision, x_eeg):
        x = torch.cat([x_audio, x_vision, x_eeg], dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Training driver
# ---------------------------------------------------------------------------
def train_subject(sub: int) -> dict | None:
    feat_path = FEAT_DIR / f"sub{sub:02d}.npz"
    if not feat_path.exists():
        print(f"  [sub{sub:02d}] feature file missing: {feat_path}  -- skipping")
        return None

    z = np.load(feat_path)
    to_t = lambda arr, dtype=torch.float32: torch.from_numpy(arr).to(
        dtype=dtype, device=DEVICE)

    train_a = to_t(z["train_audio"])
    train_v = to_t(z["train_vision"])
    train_e = to_t(z["train_eeg"])
    train_y = to_t(z["train_y"], dtype=torch.long)

    test_a = to_t(z["test_audio"])
    test_v = to_t(z["test_vision"])
    test_e = to_t(z["test_eeg"])
    test_y = to_t(z["test_y"], dtype=torch.long)

    torch.manual_seed(SEED)
    model = ConcatMLP(
        audio_dim=train_a.shape[1],
        vision_dim=train_v.shape[1],
        eeg_dim=train_e.shape[1],
    ).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    n_train = len(train_y)
    best_test_acc = 0.0
    best_test_epoch = -1
    t0 = time.time()

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        epoch_loss = 0.0
        for i in range(0, n_train, BATCH):
            idx = perm[i:i + BATCH]
            optimizer.zero_grad()
            logits = model(train_a[idx], train_v[idx], train_e[idx])
            loss = criterion(logits, train_y[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        scheduler.step()
        epoch_loss /= n_train

        model.eval()
        with torch.no_grad():
            logits = model(test_a, test_v, test_e)
            preds = logits.argmax(dim=-1)
            test_acc = (preds == test_y).float().mean().item()

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_epoch = epoch + 1

        if (epoch + 1) % 20 == 0 or epoch == 0 or epoch == EPOCHS - 1:
            print(f"    epoch {epoch+1:3d}/{EPOCHS}: "
                  f"loss={epoch_loss:.4f}  test_acc={test_acc:.3f}  "
                  f"(best {best_test_acc:.3f} @ epoch {best_test_epoch})")

    dt = time.time() - t0
    print(f"  [sub{sub:02d}] done  best_test_acc={best_test_acc:.4f}  "
          f"({dt:.0f}s)")
    return {"sub": sub, "test_acc": best_test_acc}


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------
def load_done_subjects() -> set[int]:
    if not RESULTS_CSV.exists():
        return set()
    done = set()
    with open(RESULTS_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                done.add(int(row["subject"]))
            except (KeyError, ValueError):
                continue
    return done


def append_row(sub: int, test_acc: float) -> None:
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    new_file = not RESULTS_CSV.exists()
    with open(RESULTS_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["subject", "test_acc"])
        writer.writerow([sub, f"{test_acc:.6f}"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print(f"Concat-MLP fusion baseline")
    print(f"  Device       : {DEVICE}")
    print(f"  Feature dir  : {FEAT_DIR}")
    print(f"  Output CSV   : {RESULTS_CSV}")
    print(f"  Architecture : 2496 -> {HIDDEN_1} -> {HIDDEN_2} -> {N_CLASSES}")
    print(f"  Epochs       : {EPOCHS}, batch {BATCH}, AdamW lr={LR}")
    print()

    done = load_done_subjects()
    if done:
        print(f"  Resuming: {len(done)} subjects already in CSV "
              f"(skipping those)")
    print()

    n_total = len(SUBJECTS)
    n_done_at_start = sum(1 for s in SUBJECTS if s in done)

    for i, sub in enumerate(SUBJECTS, start=1):
        if sub in done:
            print(f"[{i}/{n_total}] sub{sub:02d}  already done, skipping")
            continue

        print(f"[{i}/{n_total}] sub{sub:02d}  training ...")
        result = train_subject(sub)
        if result is None:
            continue
        append_row(result["sub"], result["test_acc"])
        print(f"  saved row to {RESULTS_CSV.name}")
        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    accs = []
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    accs.append(float(row["test_acc"]))
                except (KeyError, ValueError):
                    continue
    if accs:
        accs = np.array(accs)
        print(f"  Subjects scored: {len(accs)}/{n_total}")
        print(f"  Mean test acc : {accs.mean():.4f}")
        print(f"  Std test acc  : {accs.std(ddof=0):.4f}")
        print(f"  Min / Max     : {accs.min():.4f} / {accs.max():.4f}")
    else:
        print("  No rows in CSV yet.")


if __name__ == "__main__":
    main()
