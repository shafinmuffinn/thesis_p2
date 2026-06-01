"""Compute per-modality test logits for all subjects from Day-5 state_dicts.

Day-5's pipeline saved per-modality state_dicts but not per-modality test
logits (it only saved the fused logits at the end). This script reads each
subject's state_dicts, runs inference on the test set, and writes per-
modality logits to:

    CHECKPOINTS / day5_per_modality_logits / sub{NN}_{audio,vision,eeg}.npz

Those files are direct inputs to:
    - day3_late_fusion.py (naïve late-fusion baseline on 42 subjects)
    - suppression_matrix.py (the headline coherence figure)

Both consumer scripts read from a configurable LOGITS_SUBDIR env var (default
'day5_per_modality_logits') so the existing 3-subject day2_logits/ files
remain untouched.

Wall-clock on Colab L4 for all 42 subjects: ~20 min.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "EAV"))

from paths import (
    CHECKPOINTS, EAV_PICKLES,
    HF_AUDIO_MODEL, HF_VISION_MODEL,
)

SUBJECTS = list(range(1, 43))

STATE_DIR = CHECKPOINTS / "day5_state_dicts"
OUT_DIR = CHECKPOINTS / "day5_per_modality_logits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

FOLDER = {"audio": "Audio", "vision": "Vision", "eeg": "EEG"}
SUFFIX = {"audio": "aud", "vision": "vis", "eeg": "eeg"}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


def _strip_dataparallel(sd: dict) -> dict:
    if not any(k.startswith("module.") for k in sd):
        return sd
    return {k.removeprefix("module."): v for k, v in sd.items()}


def load_pkl(sub: int, modality: str):
    path = EAV_PICKLES / FOLDER[modality] / f"subject_{sub:02d}_{SUFFIX[modality]}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)  # tr_x, tr_y, te_x, te_y


# ---------------------------------------------------------------------------
# Per-modality inference (full forward, including classifier head)
# ---------------------------------------------------------------------------

def infer_audio(sub: int) -> tuple[np.ndarray, np.ndarray]:
    from transformers import ASTFeatureExtractor, AutoModelForAudioClassification

    _, _, te_x, te_y = load_pkl(sub, "audio")
    state = torch.load(STATE_DIR / f"sub{sub:02d}_audio.pt", map_location=DEVICE)

    model = AutoModelForAudioClassification.from_pretrained(
        HF_AUDIO_MODEL, num_labels=5, ignore_mismatched_sizes=True,
    ).to(DEVICE)
    model.load_state_dict(_strip_dataparallel(state))
    model.eval()

    processor = ASTFeatureExtractor()
    logits_chunks = []
    BATCH = 8
    with torch.no_grad():
        for i in range(0, len(te_x), BATCH):
            batch = list(np.asarray(te_x[i:i+BATCH]))
            inputs = processor(
                batch, sampling_rate=16000, padding="max_length",
                return_tensors="pt",
            ).input_values.to(DEVICE)
            logits = model(inputs).logits.cpu().numpy()
            logits_chunks.append(logits)
    del model
    torch.cuda.empty_cache()
    return np.concatenate(logits_chunks, axis=0).astype(np.float32), np.asarray(te_y)


def infer_vision(sub: int) -> tuple[np.ndarray, np.ndarray]:
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    _, _, te_x, te_y = load_pkl(sub, "vision")
    state = torch.load(STATE_DIR / f"sub{sub:02d}_vision.pt", map_location=DEVICE)

    model = AutoModelForImageClassification.from_pretrained(
        HF_VISION_MODEL, num_labels=5, ignore_mismatched_sizes=True,
    ).to(DEVICE)
    model.load_state_dict(_strip_dataparallel(state))
    model.eval()

    processor = AutoImageProcessor.from_pretrained(HF_VISION_MODEL)

    te_x = np.asarray(te_x)
    N, F = te_x.shape[0], te_x.shape[1]   # N clips, F=25 frames per clip
    flat_frames = [frame for clip in te_x for frame in clip]
    all_logits = []
    BATCH = 64
    with torch.no_grad():
        for i in range(0, len(flat_frames), BATCH):
            batch = flat_frames[i:i+BATCH]
            inputs = processor(images=batch, return_tensors="pt").pixel_values.to(DEVICE)
            logits = model(inputs).logits.cpu().numpy()   # (B, 5)
            all_logits.append(logits)
    per_frame = np.concatenate(all_logits, axis=0)         # (N*F, 5)
    per_clip = per_frame.reshape(N, F, -1).mean(axis=1)    # (N, 5)
    del model
    torch.cuda.empty_cache()
    return per_clip.astype(np.float32), np.asarray(te_y)


def infer_eeg(sub: int) -> tuple[np.ndarray, np.ndarray]:
    from CNN_torch.EEGNet_tor import EEGNet_tor

    _, _, te_x, te_y = load_pkl(sub, "eeg")
    state = torch.load(STATE_DIR / f"sub{sub:02d}_eeg.pt", map_location=DEVICE)

    model = EEGNet_tor(
        nb_classes=5, D=8, F2=64, Chans=30,
        kernLength=300, Samples=500, dropoutRate=0.5,
    ).to(DEVICE)
    model.load_state_dict(_strip_dataparallel(state))
    model.eval()

    te_x = np.asarray(te_x)
    if te_x.ndim == 3:
        te_x = te_x[:, None]   # (N, 1, 30, 500)
    with torch.no_grad():
        x = torch.from_numpy(te_x).float().to(DEVICE)
        logits = model(x).cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return logits.astype(np.float32), np.asarray(te_y)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

INFER_FNS = {"audio": infer_audio, "vision": infer_vision, "eeg": infer_eeg}


def main():
    print(f"Output dir: {OUT_DIR}")
    print(f"State dicts at: {STATE_DIR}")
    print(f"Will process {len(SUBJECTS)} subjects × 3 modalities = "
          f"{len(SUBJECTS)*3} inference runs\n")

    t0_all = time.time()
    for sub in SUBJECTS:
        print(f"========== Subject {sub:02d} ==========")
        for mod in ("audio", "vision", "eeg"):
            out = OUT_DIR / f"sub{sub:02d}_{mod}.npz"
            if out.exists():
                print(f"  [{mod:6s}] cache hit: {out.name}")
                continue
            state_path = STATE_DIR / f"sub{sub:02d}_{mod}.pt"
            if not state_path.exists():
                print(f"  [{mod:6s}] ⚠ state_dict missing: {state_path.name}")
                continue
            t0 = time.time()
            logits, labels = INFER_FNS[mod](sub)
            np.savez(out, logits=logits, labels=labels)
            print(f"  [{mod:6s}] saved {out.name}  shape={logits.shape}  "
                  f"({time.time()-t0:.1f}s)")

    print(f"\n✅ Done in {(time.time()-t0_all)/60:.1f} min")
    print(f"Files in {OUT_DIR}:")
    for p in sorted(OUT_DIR.glob("*.npz"))[:5]:
        print(f"  {p.name}")
    n_files = len(list(OUT_DIR.glob("*.npz")))
    print(f"  ... ({n_files} files total)")


if __name__ == "__main__":
    main()
