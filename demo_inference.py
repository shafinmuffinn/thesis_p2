"""demo_inference.py — Sliding-window trimodal emotion inference on an MP4 file.

Slides a configurable window (default 5 s) at a configurable stride (default 1 s)
over the input video. For each window:

  Audio  → librosa loads the full waveform at 16 kHz; window slice = 80 000 samples
            → AST feature extractor (base pretrained, no per-subject fine-tuning) → 768-d
  Vision → 25 frames sampled uniformly from the window → ViT feature extractor → 768-d
  EEG    → zeros(960)  (no hardware; softhard-dropout training makes this valid)

Fusion: ensemble of all available Day-7 fusion checkpoints
(CHECKPOINTS/day7_state_dicts/sub{NN}_fusion.pt). Each model produces a softmax
distribution; the per-model softmax vectors are averaged and argmax-decoded.
Using the base (non-subject-specific) feature extractors + an ensemble of all
42 per-subject fusion heads is the most generalisable inference strategy for a
new, unseen speaker.

Outputs:
  <video_stem>_predictions.json — list of window dicts (start_sec, end_sec,
                                   emotion, confidence, probs[5])

Usage (Colab):
  !python demo_inference.py /path/to/demo.mp4
  !python demo_inference.py /path/to/demo.mp4 --stride 1 --window 5 --out preds.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import librosa
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "EAV"))

from paths import CHECKPOINTS, HF_AUDIO_MODEL, HF_VISION_MODEL, IDX_TO_EMOTION
from fusion import TrimodalAttentionFusion
from fusion.feature_extractors import AudioFeatureExtractor, VisionFeatureExtractor

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
EMOTIONS  = [IDX_TO_EMOTION[i] for i in range(5)]
STATE_DIR = CHECKPOINTS / "day7_state_dicts"

FRAME_SIZE = 224   # resize all frames to this; ViT processor handles it


# ---------------------------------------------------------------------------
# Load ensemble of Day-7 fusion models
# ---------------------------------------------------------------------------

def load_fusion_ensemble() -> list[TrimodalAttentionFusion]:
    pts = sorted(STATE_DIR.glob("sub*_fusion.pt"))
    if not pts:
        raise FileNotFoundError(
            f"No Day-7 fusion checkpoints found in {STATE_DIR}.\n"
            "Run day7_modality_dropout.py first."
        )
    models = []
    for pt in pts:
        m = TrimodalAttentionFusion(
            audio_dim=768, vision_dim=768, eeg_dim=960
        ).to(DEVICE)
        m.load_state_dict(torch.load(pt, map_location=DEVICE))
        m.eval()
        models.append(m)
    print(f"  Loaded {len(models)} fusion models from {STATE_DIR}")
    return models


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def load_video_frames(video_path: Path, frame_size: int = FRAME_SIZE) -> tuple[np.ndarray, float]:
    """Read all frames from video, resize to (frame_size, frame_size, 3) RGB.

    Returns (frames, fps) where frames has shape (T, frame_size, frame_size, 3) uint8.
    Memory: ~300 MB for a 60 s video at 25 fps and frame_size=224.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (frame_size, frame_size))
        frames.append(frame)
    cap.release()
    return np.stack(frames).astype(np.uint8), float(fps)


def sample_frames_for_window(
    frames: np.ndarray,
    t_start: float,
    t_end: float,
    fps: float,
    n_frames: int = 25,
) -> np.ndarray:
    """Return n_frames uniformly sampled from the [t_start, t_end) range.

    Pads by repeating the last frame if the window extends past the video end.
    """
    i_start = int(t_start * fps)
    i_end   = min(int(t_end * fps), len(frames))
    if i_start >= len(frames):
        # Past end of video — return last frame repeated
        return np.repeat(frames[-1:], n_frames, axis=0)
    if i_end <= i_start + 1:
        return np.repeat(frames[i_start : i_start + 1], n_frames, axis=0)
    indices = np.linspace(i_start, i_end - 1, n_frames).round().astype(int)
    return frames[indices]   # (n_frames, H, W, 3)


# ---------------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------------

def load_audio_waveform(video_path: Path, sr: int = 16000) -> np.ndarray:
    """Load audio track from video file using librosa (requires ffmpeg).

    Returns (total_samples,) float32 array at `sr` Hz.
    """
    waveform, _ = librosa.load(str(video_path), sr=sr, mono=True)
    return waveform.astype(np.float32)


def slice_audio_window(
    waveform: np.ndarray,
    t_start: float,
    window_sec: float = 5.0,
    sr: int = 16000,
) -> np.ndarray:
    """Slice a fixed-length audio window, zero-padding if near the end."""
    n = int(window_sec * sr)            # 80 000 at 5 s / 16 kHz
    i_start = int(t_start * sr)
    chunk = waveform[i_start : i_start + n]
    if len(chunk) < n:
        chunk = np.pad(chunk, (0, n - len(chunk)))
    return chunk                        # (n,)


# ---------------------------------------------------------------------------
# Ensemble inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_ensemble(
    models:        list[TrimodalAttentionFusion],
    feat_audio:    torch.Tensor,   # (N, 768)
    feat_vision:   torch.Tensor,   # (N, 768)
) -> np.ndarray:
    """Average softmax over all ensemble models; EEG is zero.

    Returns (N, 5) float32 probability array.
    """
    feat_audio  = feat_audio.to(DEVICE)
    feat_vision = feat_vision.to(DEVICE)
    eeg_zeros   = torch.zeros(feat_audio.size(0), 960, device=DEVICE)

    avg_probs = torch.zeros(feat_audio.size(0), 5, device=DEVICE)
    for model in models:
        logits = model(feat_audio, feat_vision, eeg_zeros)["logits"]
        avg_probs += torch.softmax(logits, dim=-1)
    avg_probs /= len(models)
    return avg_probs.cpu().numpy()


# ---------------------------------------------------------------------------
# Main inference pipeline
# ---------------------------------------------------------------------------

def run_inference(
    video_path:  Path,
    window_sec:  float = 5.0,
    stride_sec:  float = 1.0,
    output_path: Path | None = None,
) -> list[dict]:

    print(f"\n{'='*60}")
    print(f"Video   : {video_path}")
    print(f"Window  : {window_sec} s    Stride: {stride_sec} s")
    print(f"Device  : {DEVICE}")
    print(f"{'='*60}\n")

    t0 = time.time()

    # 1. Load ensemble
    print("[1/5] Loading Day-7 fusion ensemble ...")
    models = load_fusion_ensemble()

    # 2. Load base feature extractors (no per-subject fine-tuning — most general)
    print("[2/5] Loading base feature extractors (pretrained, no subject-specific weights) ...")
    aud_ex = AudioFeatureExtractor(HF_AUDIO_MODEL, state_dict_path=None)
    vis_ex = VisionFeatureExtractor(HF_VISION_MODEL, state_dict_path=None)

    # 3. Load video + audio
    print("[3/5] Loading video frames and audio ...")
    frames, fps = load_video_frames(video_path)
    waveform    = load_audio_waveform(video_path)
    duration    = len(frames) / fps
    print(f"  Duration : {duration:.1f} s   FPS : {fps:.1f}   Frames : {len(frames)}")

    # 4. Build windows
    starts = np.arange(0.0, duration - window_sec + stride_sec, stride_sec)
    starts = starts[starts + window_sec <= duration + 0.5]   # allow partial last window
    n_windows = len(starts)
    print(f"  Windows  : {n_windows}  ({stride_sec} s stride)")

    # 5. Prepare batched inputs
    print("[4/5] Preparing audio and vision inputs ...")
    audio_batch  = np.stack([
        slice_audio_window(waveform, t, window_sec) for t in starts
    ])   # (N, 80000)

    vision_batch = np.stack([
        sample_frames_for_window(frames, t, t + window_sec, fps)
        for t in starts
    ])   # (N, 25, FRAME_SIZE, FRAME_SIZE, 3)

    # 6. Extract features
    print("  Extracting audio features ...")
    feat_audio  = torch.from_numpy(aud_ex.extract(audio_batch))   # (N, 768)
    print("  Extracting vision features ...")
    feat_vision = torch.from_numpy(vis_ex.extract(vision_batch))  # (N, 768)

    # 7. Ensemble fusion
    print("[5/5] Running ensemble fusion inference ...")
    probs = run_ensemble(models, feat_audio, feat_vision)   # (N, 5)

    # 8. Format predictions
    predictions = []
    for i, t in enumerate(starts):
        p = probs[i]
        best = int(p.argmax())
        predictions.append({
            "start_sec":  float(round(t, 2)),
            "end_sec":    float(round(t + window_sec, 2)),
            "emotion":    EMOTIONS[best],
            "confidence": float(round(float(p[best]), 4)),
            "probs": {e: float(round(float(p[j]), 4)) for j, e in enumerate(EMOTIONS)},
        })

    # 9. Save
    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_predictions.json"
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=2)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f} s")
    print(f"Predictions saved to: {output_path}")

    # Quick summary
    from collections import Counter
    counts = Counter(p["emotion"] for p in predictions)
    print("\nEmotion distribution across windows:")
    for emo, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        pct = 100 * cnt / n_windows
        print(f"  {emo:<12s} {cnt:3d} windows ({pct:.1f}%)")

    return predictions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Trimodal emotion inference on MP4.")
    parser.add_argument("video", type=Path, help="Input MP4 file.")
    parser.add_argument("--stride", type=float, default=1.0,
                        help="Window stride in seconds (default: 1.0).")
    parser.add_argument("--window", type=float, default=5.0,
                        help="Window size in seconds (default: 5.0).")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output JSON path (default: <video_stem>_predictions.json).")
    args = parser.parse_args()

    if not args.video.exists():
        sys.exit(f"Error: file not found: {args.video}")

    run_inference(args.video, window_sec=args.window,
                  stride_sec=args.stride, output_path=args.out)


if __name__ == "__main__":
    main()
