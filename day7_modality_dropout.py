"""Day 7 — Fusion re-training with softhard modality dropout.

Reads the cached per-modality features from Day 5 Stage B
(CHECKPOINTS/day5_features/sub{NN}.npz) and re-trains TrimodalAttentionFusion
with softhard modality dropout applied at the feature level during training.

Softhard scheme (Chumachenko et al., ICPR 2022):
    For each sample in a training batch, with probability DROP_P one modality
    is chosen uniformly at random and its feature vector is zeroed (hard).
    The two surviving modalities are scaled by 3/2 (soft), preserving total
    feature energy on average. This teaches the model to produce sensible
    predictions with any subset of the three modalities — including
    audio+vision only, which is the demo inference path (no EEG hardware).

Stage A and B from day5_pipeline.py need not be re-run; their outputs
(per-modality state_dicts and cached features) are reused verbatim.

Evaluation modes after each subject's training:
    full   — all three modalities (standard metric)
    av     — audio + vision, EEG zeroed   ← demo path
    ae     — audio + EEG,   vision zeroed
    ve     — vision + EEG,  audio zeroed

Outputs (all on Drive):
    RESULTS/day7_fusion_dropout.csv             per-subject accuracy + F1
    CHECKPOINTS/day7_fusion_logits/sub{NN}.npz  best-epoch logits + confusion matrix
    CHECKPOINTS/day7_state_dicts/sub{NN}_fusion.pt  best fusion checkpoint (used by demo)

Idempotent: subjects already written to the CSV are skipped on re-run.
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "EAV"))

from paths import CHECKPOINTS, RESULTS, IDX_TO_EMOTION
from fusion import TrimodalAttentionFusion

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SUBJECTS = list(range(1, 43))
FEAT_DIR = CHECKPOINTS / "day5_features"   # reuse Day-5 cached features

FUSION_EPOCHS = 80
FUSION_BATCH  = 32
FUSION_LR     = 1e-3
FUSION_WD     = 1e-4
DROP_P        = 0.5   # per-sample probability that one modality is dropped

LOGITS_DIR  = CHECKPOINTS / "day7_fusion_logits"
STATE_DIR   = CHECKPOINTS / "day7_state_dicts"
RESULTS_CSV = RESULTS / "day7_fusion_dropout.csv"

DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
EMOTIONS = [IDX_TO_EMOTION[i] for i in range(5)]

# Keys: display name. Values: (zero_audio, zero_vision, zero_eeg)
EVAL_MODES: dict[str, tuple[bool, bool, bool]] = {
    "full": (False, False, False),
    "av":   (False, False, True),
    "ae":   (False, True,  False),
    "ve":   (True,  False, False),
}


# ---------------------------------------------------------------------------
# Softhard modality dropout
# ---------------------------------------------------------------------------

def softhard_dropout(
    x_audio:  torch.Tensor,
    x_vision: torch.Tensor,
    x_eeg:    torch.Tensor,
    p: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-sample random modality zeroing with soft energy rescaling.

    For each sample independently:
      - with probability p, pick one modality uniformly at random and zero it
      - scale the remaining modalities by 3 / n_active so expected feature
        energy per sample is preserved regardless of how many are dropped

    Only applied during training. Caller must not use this during eval.
    """
    B = x_audio.size(0)
    mask = torch.ones(B, 3, device=x_audio.device)
    for i in range(B):
        if torch.rand(1).item() < p:
            mask[i, torch.randint(0, 3, (1,)).item()] = 0.0
    n_active = mask.sum(dim=1, keepdim=True).clamp(min=1)
    mask = mask * (3.0 / n_active)
    return (
        x_audio  * mask[:, 0:1],
        x_vision * mask[:, 1:2],
        x_eeg    * mask[:, 2:3],
    )


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def eval_mode(
    model:        nn.Module,
    test_x:       dict[str, torch.Tensor],
    test_y:       torch.Tensor,
    zero_audio:   bool,
    zero_vision:  bool,
    zero_eeg:     bool,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return (accuracy, predictions, labels) under the given modality mask."""
    model.eval()
    with torch.no_grad():
        xa = torch.zeros_like(test_x["audio"])  if zero_audio  else test_x["audio"]
        xv = torch.zeros_like(test_x["vision"]) if zero_vision else test_x["vision"]
        xe = torch.zeros_like(test_x["eeg"])    if zero_eeg    else test_x["eeg"]
        logits = model(xa, xv, xe)["logits"]
        preds  = logits.argmax(dim=-1).cpu().numpy()
    labels = test_y.cpu().numpy()
    return float((preds == labels).mean()), preds, labels


# ---------------------------------------------------------------------------
# Load already-completed subjects from CSV
# ---------------------------------------------------------------------------

def load_completed_subjects() -> set[int]:
    if not RESULTS_CSV.exists():
        return set()
    with open(RESULTS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        return {int(row["subject"]) for row in reader}


# ---------------------------------------------------------------------------
# Per-subject training + multi-mode evaluation
# ---------------------------------------------------------------------------

def train_and_eval(sub: int) -> dict:
    feat_path = FEAT_DIR / f"sub{sub:02d}.npz"
    if not feat_path.exists():
        raise FileNotFoundError(
            f"Features missing for sub{sub:02d}. Run day5_pipeline.py Stage B first."
        )

    z = np.load(feat_path)

    def to_t(key: str, dtype=torch.float32) -> torch.Tensor:
        return torch.from_numpy(z[key]).to(dtype=dtype, device=DEVICE)

    train_x = {k: to_t(f"train_{k}") for k in ("audio", "vision", "eeg")}
    test_x  = {k: to_t(f"test_{k}")  for k in ("audio", "vision", "eeg")}
    train_y = to_t("train_y", dtype=torch.long)
    test_y  = to_t("test_y",  dtype=torch.long)

    model = TrimodalAttentionFusion(
        audio_dim=train_x["audio"].shape[1],
        vision_dim=train_x["vision"].shape[1],
        eeg_dim=train_x["eeg"].shape[1],
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=FUSION_LR, weight_decay=FUSION_WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FUSION_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_acc   = 0.0
    best_state: dict | None        = None
    best_logits: np.ndarray | None = None
    N = len(train_y)
    t0 = time.time()

    for epoch in range(FUSION_EPOCHS):
        # --- Train with softhard dropout ---
        model.train()
        perm       = torch.randperm(N, device=DEVICE)
        epoch_loss = 0.0
        for i in range(0, N, FUSION_BATCH):
            idx = perm[i : i + FUSION_BATCH]
            xa, xv, xe = softhard_dropout(
                train_x["audio"][idx],
                train_x["vision"][idx],
                train_x["eeg"][idx],
                p=DROP_P,
            )
            optimizer.zero_grad()
            out  = model(xa, xv, xe)
            loss = criterion(out["logits"], train_y[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        scheduler.step()

        # --- Evaluate on full modality (no dropout) ---
        model.eval()
        with torch.no_grad():
            out_full   = model(test_x["audio"], test_x["vision"], test_x["eeg"])
            logits_np  = out_full["logits"].cpu().numpy()
            acc        = (out_full["logits"].argmax(dim=-1) == test_y).float().mean().item()

        if acc > best_acc:
            best_acc    = acc
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_logits = logits_np

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == FUSION_EPOCHS - 1:
            print(f"    epoch {epoch+1:3d}:  loss={epoch_loss/N:.4f}  "
                  f"test_acc(full)={acc:.3f}  best={best_acc:.3f}")

    # --- Reload best checkpoint and run all eval modes ---
    model.load_state_dict(best_state)

    result = {"sub": sub}
    full_preds = full_labels = None
    for mode, (za, zv, ze) in EVAL_MODES.items():
        acc, preds, labels = eval_mode(model, test_x, test_y,
                                       zero_audio=za, zero_vision=zv, zero_eeg=ze)
        result[f"acc_{mode}"] = acc
        if mode == "full":
            full_preds, full_labels = preds, labels

    # --- Per-class F1 and confusion matrix on full-modality eval ---
    f1s = f1_score(full_labels, full_preds,
                   labels=list(range(5)), average=None, zero_division=0)
    cm  = confusion_matrix(full_labels, full_preds, labels=list(range(5)))
    for i, emo in enumerate(EMOTIONS):
        result[f"f1_{emo.lower()}"] = float(f1s[i])

    elapsed = time.time() - t0
    print(
        f"  sub{sub:02d}: full={result['acc_full']:.3f}  "
        f"av={result['acc_av']:.3f}  ae={result['acc_ae']:.3f}  "
        f"ve={result['acc_ve']:.3f}  ({elapsed:.0f}s)"
    )
    print("         F1: " +
          "  ".join(f"{e}={result[f'f1_{e.lower()}']:.3f}" for e in EMOTIONS))

    # --- Persist outputs ---
    LOGITS_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        LOGITS_DIR / f"sub{sub:02d}.npz",
        logits=best_logits,
        labels=z["test_y"],
        confusion_matrix=cm,
    )
    torch.save(best_state, STATE_DIR / f"sub{sub:02d}_fusion.pt")
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"Device      : {DEVICE}")
    print(f"DROP_P      : {DROP_P}   (per-sample prob of dropping one modality)")
    print(f"Epochs      : {FUSION_EPOCHS}   LR={FUSION_LR}   WD={FUSION_WD}")
    print(f"Feature dir : {FEAT_DIR}")
    print()

    fieldnames = (
        ["subject"]
        + [f"acc_{m}" for m in EVAL_MODES]
        + [f"f1_{e.lower()}" for e in EMOTIONS]
    )

    completed = load_completed_subjects()
    write_header = not RESULTS_CSV.exists()
    fh = open(RESULTS_CSV, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    summary = []
    for sub in SUBJECTS:
        if sub in completed:
            print(f"  [skip] sub{sub:02d} already in {RESULTS_CSV.name}")
            continue
        print(f"\n========== Subject {sub:02d} ==========")
        result = train_and_eval(sub)
        summary.append(result)
        row = {"subject": sub}
        row.update({k: f"{v:.6f}" for k, v in result.items() if k != "sub"})
        writer.writerow(row)
        fh.flush()

    fh.close()

    if not summary:
        print("\nAll subjects already completed. Reading from CSV for summary.")
        with open(RESULTS_CSV, newline="") as f:
            for row in csv.DictReader(f):
                summary.append({k: float(v) if k != "subject" else int(v)
                                 for k, v in row.items()})

    print("\n========== 42-Subject Summary ==========")
    for mode in EVAL_MODES:
        col  = f"acc_{mode}"
        vals = [r[col] for r in summary]
        print(f"  {col:<12s}: mean={np.mean(vals):.3f}  std={np.std(vals):.3f}")

    print("\n  Per-emotion macro-F1 (full modality):")
    for emo in EMOTIONS:
        col  = f"f1_{emo.lower()}"
        vals = [r[col] for r in summary]
        print(f"    {emo:<12s}: mean={np.mean(vals):.3f}  std={np.std(vals):.3f}")

    full_mean = np.mean([r["acc_full"] for r in summary])
    av_mean   = np.mean([r["acc_av"]   for r in summary])
    print(f"\n  Day 5 (no dropout)        mean acc_full : 0.802")
    print(f"  Day 7 (softhard dropout)  mean acc_full : {full_mean:.3f}")
    print(f"  Demo path (AV, EEG=0)     mean acc_av   : {av_mean:.3f}")
    print(f"\nResults    : {RESULTS_CSV}")
    print(f"State dicts: {STATE_DIR}")
    print(f"Logits     : {LOGITS_DIR}")


if __name__ == "__main__":
    main()
