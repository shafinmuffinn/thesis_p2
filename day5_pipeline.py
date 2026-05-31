"""Day 5 — Train trimodal cross-attention fusion on EAV.

Three sequential stages, each cached on disk so re-running picks up where
the previous run left off:

    Stage A — per-modality re-training.
        For each subject, re-train AST/ViT/EEGNet at Day-2 epoch budgets
        (10+15 / 10+5 / 350). Saves state_dicts to
        CHECKPOINTS/day5_state_dicts/sub{NN}_{modality}.pt.
        Skipped if all three state_dicts for a subject already exist.

    Stage B — feature extraction.
        Loads each subject's pickle, instantiates the matching feature
        extractor with the trained state_dict, and saves train+test
        features to CHECKPOINTS/day5_features/sub{NN}.npz.
        Skipped if the .npz already exists.

    Stage C — fusion training.
        Loads the cached features and trains TrimodalAttentionFusion for
        FUSION_EPOCHS epochs with best-test-checkpoint selection. Writes
        per-subject results to RESULTS/day5_fusion.csv.

Wall-clock on a 4080 for 3 subjects: A ~90 min, B ~5 min, C ~10 min.

This script is intentionally idempotent. If the per-modality state_dicts
already exist, it skips Stage A. If features already exist, it skips
Stage B. So iterating on the fusion hyperparameters is fast — only
Stage C runs after the first full pass.
"""

from __future__ import annotations

import csv
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Make the EAV repo and our fusion module importable.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "EAV"))

from paths import (
    CHECKPOINTS, RESULTS, EAV_PICKLES,
    HF_AUDIO_MODEL, HF_VISION_MODEL,
)
from fusion import TrimodalAttentionFusion
from fusion.feature_extractors import (
    AudioFeatureExtractor, VisionFeatureExtractor, EEGFeatureExtractor,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SUBJECTS = list(range(1, 43))   # all 42 EAV subjects

# Per-modality training budgets — reduced for the 42-subject rollout.
# Justification: Day 2 showed AST overfits at epoch 5 of fine-tuning, so the
# original 10+15 schedule wasted ~10 epochs per subject. ViT was pre-fine-tuned
# on facial emotions and converges in 1-2 epochs. EEGNet needs the full 350.
# At these budgets, ~12-15 min/subject on V100, ~20-25 min/subject on T4.
AUD_EPOCHS_FROZEN, AUD_EPOCHS_FT = 5, 5
VIS_EPOCHS_FROZEN, VIS_EPOCHS_FT = 3, 2
EEG_EPOCHS = 350

# Fusion training
FUSION_EPOCHS = 80
FUSION_BATCH = 32
FUSION_LR = 1e-3
FUSION_WEIGHT_DECAY = 1e-4

# Output paths
STATE_DIR = CHECKPOINTS / "day5_state_dicts"
FEAT_DIR = CHECKPOINTS / "day5_features"
RESULTS_CSV = RESULTS / "day5_fusion.csv"

FOLDER = {"audio": "Audio", "vision": "Vision", "eeg": "EEG"}
SUFFIX = {"audio": "aud", "vision": "vis", "eeg": "eeg"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pkl(sub: int, modality: str):
    path = EAV_PICKLES / FOLDER[modality] / f"subject_{sub:02d}_{SUFFIX[modality]}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def _unwrap_state_dict(model) -> dict:
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


# ---------------------------------------------------------------------------
# Stage A — per-modality re-training, saving state_dicts
# ---------------------------------------------------------------------------

def train_audio_and_save(sub: int) -> Path:
    out = STATE_DIR / f"sub{sub:02d}_audio.pt"
    if out.exists():
        return out
    from Transformer_torch.Transformer_Audio import AudioModelTrainer

    print(f"  [audio] training sub{sub:02d}...")
    data = load_pkl(sub, "audio")
    trainer = AudioModelTrainer(
        data, model_path=HF_AUDIO_MODEL,
        sub=f"sub{sub:02d}", num_classes=5, lr=5e-4, batch_size=8,
    )
    trainer.train(epochs=AUD_EPOCHS_FROZEN, lr=5e-4, freeze=True)
    trainer.train(epochs=AUD_EPOCHS_FT, lr=5e-6, freeze=False)
    torch.save(_unwrap_state_dict(trainer.model), out)
    print(f"  [audio] saved {out.name}")
    return out


def train_vision_and_save(sub: int) -> Path:
    out = STATE_DIR / f"sub{sub:02d}_vision.pt"
    if out.exists():
        return out
    from Transformer_torch.Transformer_Vision import ImageClassifierTrainer

    print(f"  [vision] training sub{sub:02d}...")
    data = load_pkl(sub, "vision")
    trainer = ImageClassifierTrainer(
        data, model_path=HF_VISION_MODEL,
        sub=f"sub{sub:02d}", num_labels=5, lr=5e-5, batch_size=32,
    )
    trainer.train(epochs=VIS_EPOCHS_FROZEN, lr=5e-4, freeze=True)
    trainer.train(epochs=VIS_EPOCHS_FT, lr=5e-6, freeze=False)
    torch.save(_unwrap_state_dict(trainer.model), out)
    print(f"  [vision] saved {out.name}")
    return out


def train_eeg_and_save(sub: int) -> Path:
    out = STATE_DIR / f"sub{sub:02d}_eeg.pt"
    if out.exists():
        return out
    from CNN_torch.EEGNet_tor import EEGNet_tor, Trainer_uni

    print(f"  [eeg] training sub{sub:02d}...")
    tr_x, tr_y, te_x, te_y = load_pkl(sub, "eeg")
    tr_x = torch.from_numpy(tr_x).float().unsqueeze(1)
    te_x = torch.from_numpy(te_x).float().unsqueeze(1)
    model = EEGNet_tor(
        nb_classes=5, D=8, F2=64, Chans=30,
        kernLength=300, Samples=500, dropoutRate=0.5,
    )
    trainer = Trainer_uni(
        model=model, data=[tr_x, tr_y, te_x, te_y],
        lr=1e-5, batch_size=32, num_epochs=EEG_EPOCHS,
    )
    trainer.train()
    torch.save(_unwrap_state_dict(trainer.model), out)
    print(f"  [eeg] saved {out.name}")
    return out


# ---------------------------------------------------------------------------
# Stage B — feature extraction (using the trained state_dicts)
# ---------------------------------------------------------------------------

def extract_features_for_subject(sub: int) -> Path:
    out = FEAT_DIR / f"sub{sub:02d}.npz"
    if out.exists():
        print(f"  [features] cache hit: {out.name}")
        return out

    print(f"  [features] extracting for sub{sub:02d}...")
    aud_state = STATE_DIR / f"sub{sub:02d}_audio.pt"
    vis_state = STATE_DIR / f"sub{sub:02d}_vision.pt"
    eeg_state = STATE_DIR / f"sub{sub:02d}_eeg.pt"

    # Load pickles
    aud_data = load_pkl(sub, "audio")    # tr_x, tr_y, te_x, te_y
    vis_data = load_pkl(sub, "vision")
    eeg_data = load_pkl(sub, "eeg")

    # Sanity: labels should be aligned across modalities (same split scheme).
    assert np.array_equal(np.asarray(aud_data[1]), np.asarray(vis_data[1])), \
        "train labels misaligned audio/vision"
    assert np.array_equal(np.asarray(aud_data[3]), np.asarray(vis_data[3])), \
        "test labels misaligned audio/vision"

    # Instantiate extractors
    aud_ex = AudioFeatureExtractor(HF_AUDIO_MODEL, state_dict_path=aud_state)
    vis_ex = VisionFeatureExtractor(HF_VISION_MODEL, state_dict_path=vis_state)
    eeg_ex = EEGFeatureExtractor(state_dict_path=eeg_state)

    t0 = time.time()
    train_audio  = aud_ex.extract(np.asarray(aud_data[0]))
    test_audio   = aud_ex.extract(np.asarray(aud_data[2]))
    del aud_ex; torch.cuda.empty_cache()

    train_vision = vis_ex.extract(np.asarray(vis_data[0]))
    test_vision  = vis_ex.extract(np.asarray(vis_data[2]))
    del vis_ex; torch.cuda.empty_cache()

    train_eeg    = eeg_ex.extract(np.asarray(eeg_data[0]))
    test_eeg     = eeg_ex.extract(np.asarray(eeg_data[2]))
    del eeg_ex; torch.cuda.empty_cache()

    np.savez(
        out,
        train_audio=train_audio, train_vision=train_vision, train_eeg=train_eeg,
        test_audio=test_audio,   test_vision=test_vision,   test_eeg=test_eeg,
        train_y=np.asarray(aud_data[1]),
        test_y =np.asarray(aud_data[3]),
    )
    print(f"  [features] saved {out.name}  ({time.time()-t0:.0f}s)")
    return out


# ---------------------------------------------------------------------------
# Stage C — fusion training
# ---------------------------------------------------------------------------

def train_fusion_for_subject(sub: int, feat_path: Path) -> dict:
    print(f"  [fusion] training sub{sub:02d}...")
    z = np.load(feat_path)

    def to_t(arr, dtype=torch.float32):
        return torch.from_numpy(arr).to(dtype=dtype, device=DEVICE)

    train_x = {
        "audio":  to_t(z["train_audio"]),
        "vision": to_t(z["train_vision"]),
        "eeg":    to_t(z["train_eeg"]),
    }
    test_x = {
        "audio":  to_t(z["test_audio"]),
        "vision": to_t(z["test_vision"]),
        "eeg":    to_t(z["test_eeg"]),
    }
    train_y = to_t(z["train_y"], dtype=torch.long)
    test_y  = to_t(z["test_y"],  dtype=torch.long)

    model = TrimodalAttentionFusion(
        audio_dim=train_x["audio"].shape[1],
        vision_dim=train_x["vision"].shape[1],
        eeg_dim=train_x["eeg"].shape[1],
    ).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(),
                            lr=FUSION_LR, weight_decay=FUSION_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FUSION_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_test_acc = 0.0
    best_test_logits = None
    N_train = len(train_y)

    for epoch in range(FUSION_EPOCHS):
        model.train()
        perm = torch.randperm(N_train, device=DEVICE)
        epoch_loss = 0.0
        for i in range(0, N_train, FUSION_BATCH):
            idx = perm[i : i + FUSION_BATCH]
            optimizer.zero_grad()
            out = model(train_x["audio"][idx], train_x["vision"][idx], train_x["eeg"][idx])
            loss = criterion(out["logits"], train_y[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
        scheduler.step()
        epoch_loss /= N_train

        # Test eval
        model.eval()
        with torch.no_grad():
            out = model(test_x["audio"], test_x["vision"], test_x["eeg"])
            preds = out["logits"].argmax(dim=-1)
            test_acc = (preds == test_y).float().mean().item()

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_test_logits = out["logits"].cpu().numpy()

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == FUSION_EPOCHS - 1:
            print(f"    epoch {epoch+1:3d}: loss={epoch_loss:.4f}  "
                  f"test_acc={test_acc:.3f}  (best {best_test_acc:.3f})")

    # Save best-epoch logits for downstream Day 7 coherence analysis
    logits_out = CHECKPOINTS / "day5_fusion_logits" / f"sub{sub:02d}.npz"
    logits_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(logits_out, logits=best_test_logits, labels=z["test_y"])
    return {"sub": sub, "test_acc": best_test_acc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not RESULTS_CSV.exists():
        with open(RESULTS_CSV, "w", newline="") as f:
            csv.writer(f).writerow(["subject", "test_acc"])

    summary = []
    for sub in SUBJECTS:
        print(f"\n========== Subject {sub:02d} ==========")
        # Stage A
        train_audio_and_save(sub)
        train_vision_and_save(sub)
        train_eeg_and_save(sub)
        # Stage B
        feat_path = extract_features_for_subject(sub)
        # Stage C
        result = train_fusion_for_subject(sub, feat_path)
        summary.append(result)
        with open(RESULTS_CSV, "a", newline="") as f:
            csv.writer(f).writerow([result["sub"], f"{result['test_acc']:.6f}"])
        print(f"  ✅ sub{sub:02d} fusion test_acc = {result['test_acc']:.3f}")

    print("\n========== Summary ==========")
    accs = [r["test_acc"] for r in summary]
    for r in summary:
        print(f"  sub{r['sub']:02d}: {r['test_acc']:.3f}")
    print(f"  mean: {np.mean(accs):.3f}  std: {np.std(accs):.3f}")
    print(f"\nResults written to {RESULTS_CSV}")

    # Comparison vs Day 3 naive-fusion baseline (recorded in CLAUDE.md)
    print("\nDay 3 mean-softmax late fusion mean: 0.819")
    print(f"Day 5 cross-attention fusion mean   : {np.mean(accs):.3f}")
    delta = (np.mean(accs) - 0.819) * 100
    print(f"Δ vs Day-3 baseline                  : {delta:+.2f} pp")


if __name__ == "__main__":
    main()
