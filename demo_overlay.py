"""demo_overlay.py — Render per-window emotion labels onto an MP4 file.

Reads the JSON produced by demo_inference.py and renders three overlay elements
onto each frame:

  Top-left    : coloured emotion pill (label + confidence %)
  Bottom      : five-bar confidence chart, one bar per emotion
  Top-right   : "EEG unavailable — audio+video only" disclaimer
  Bottom-right: timestamp (mm:ss.f)

Audio is re-attached from the original video via FFmpeg.

Usage (Colab):
  !python demo_overlay.py demo.mp4 demo_predictions.json
  !python demo_overlay.py demo.mp4 demo_predictions.json --output demo_overlay.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Emotion colour palette (BGR for OpenCV)
# ---------------------------------------------------------------------------

EMOTION_BGR: dict[str, tuple[int, int, int]] = {
    "Neutral":   (160, 160, 160),
    "Sadness":   (200,  80,  30),
    "Anger":     ( 30,  30, 210),
    "Happiness": ( 30, 200,  80),
    "Calmness":  (200, 160,  30),
}
EMOTIONS = ["Neutral", "Sadness", "Anger", "Happiness", "Calmness"]

# Bar chart colours (same palette)
BAR_BGR = {e: EMOTION_BGR[e] for e in EMOTIONS}


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _put_text(
    img:    np.ndarray,
    text:   str,
    origin: tuple[int, int],    # (x, y)
    scale:  float = 0.7,
    color:  tuple[int, int, int] = (255, 255, 255),
    bold:   bool = False,
    shadow: bool = True,
) -> None:
    font      = cv2.FONT_HERSHEY_DUPLEX
    thickness = 2 if bold else 1
    if shadow:
        cv2.putText(img, text, (origin[0] + 1, origin[1] + 1),
                    font, scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, origin, font, scale, color, thickness, cv2.LINE_AA)


def draw_emotion_pill(
    img:       np.ndarray,
    emotion:   str,
    confidence: float,
    x: int = 18,
    y: int = 18,
    padding: int = 10,
) -> None:
    """Draw a filled rounded-rectangle pill with the emotion label."""
    label  = f"{emotion}  {confidence*100:.0f}%"
    font   = cv2.FONT_HERSHEY_DUPLEX
    scale  = 0.85
    thick  = 2
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    rx, ry = x, y
    rw, rh = tw + 2 * padding, th + 2 * padding
    color  = EMOTION_BGR.get(emotion, (120, 120, 120))
    # Filled rectangle
    cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), color, -1)
    # Text
    cv2.putText(img, label, (rx + padding, ry + rh - padding),
                font, scale, (255, 255, 255), thick, cv2.LINE_AA)


def draw_confidence_bars(
    img:      np.ndarray,
    probs:    dict[str, float],
    bar_h:    int = 16,
    bar_gap:  int = 4,
    label_w:  int = 100,
    max_bar_w: int = 200,
    margin:   int = 14,
) -> None:
    """Draw a vertical stack of labelled confidence bars at the bottom of the frame."""
    H, W = img.shape[:2]
    n = len(EMOTIONS)
    block_h = n * (bar_h + bar_gap) + bar_gap
    y0 = H - block_h - margin

    for i, emo in enumerate(EMOTIONS):
        p   = probs.get(emo, 0.0)
        bw  = int(p * max_bar_w)
        y   = y0 + i * (bar_h + bar_gap)
        x0  = margin + label_w
        color = BAR_BGR[emo]

        # Background track
        cv2.rectangle(img, (x0, y), (x0 + max_bar_w, y + bar_h), (50, 50, 50), -1)
        # Filled bar
        if bw > 0:
            cv2.rectangle(img, (x0, y), (x0 + bw, y + bar_h), color, -1)
        # Emotion label (left of bar)
        _put_text(img, emo[:9], (margin, y + bar_h - 3), scale=0.45, shadow=True)
        # Probability value (right of bar)
        _put_text(img, f"{p*100:.0f}%", (x0 + max_bar_w + 4, y + bar_h - 3),
                  scale=0.45, shadow=True)


def draw_disclaimer(img: np.ndarray) -> None:
    """Draw the EEG-unavailable caption in the top-right corner."""
    text  = "EEG unavailable"
    text2 = "audio + video only"
    H, W  = img.shape[:2]
    scale = 0.45
    font  = cv2.FONT_HERSHEY_DUPLEX
    (w1, h1), _ = cv2.getTextSize(text,  font, scale, 1)
    (w2, h2), _ = cv2.getTextSize(text2, font, scale, 1)
    x1 = W - max(w1, w2) - 14
    _put_text(img, text,  (x1, 28), scale=scale, color=(200, 200, 200))
    _put_text(img, text2, (x1, 28 + h1 + 6), scale=scale, color=(200, 200, 200))


def draw_timestamp(img: np.ndarray, t_sec: float) -> None:
    """Draw mm:ss.f timestamp in the bottom-right corner."""
    mins = int(t_sec) // 60
    secs = t_sec - mins * 60
    text = f"{mins:02d}:{secs:05.2f}"
    H, W = img.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.5
    (tw, th), _ = cv2.getTextSize(text, font, scale, 1)
    _put_text(img, text, (W - tw - 14, H - 12), scale=scale, color=(220, 220, 220))


# ---------------------------------------------------------------------------
# Prediction lookup
# ---------------------------------------------------------------------------

def build_prediction_index(predictions: list[dict]) -> dict[float, dict]:
    """Map start_sec → prediction dict for O(1) lookup."""
    return {float(p["start_sec"]): p for p in predictions}


def get_active_prediction(
    t_sec: float,
    predictions: list[dict],
    stride_sec: float,
    window_sec: float,
) -> dict | None:
    """Return the prediction whose window is active at time t_sec.

    With overlapping windows (stride < window) we pick the window whose
    start_sec is the largest value ≤ t_sec.
    """
    best = None
    for p in predictions:
        s = p["start_sec"]
        if s <= t_sec < s + window_sec:
            if best is None or s > best["start_sec"]:
                best = p
    return best


# ---------------------------------------------------------------------------
# Rendering pipeline
# ---------------------------------------------------------------------------

def render_overlay(
    video_path:   Path,
    predictions:  list[dict],
    output_path:  Path,
    window_sec:   float = 5.0,
    stride_sec:   float = 1.0,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps   = cap.get(cv2.CAP_PROP_FPS)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Write silent video to a temp file; audio is re-attached by FFmpeg later.
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(tmp_path), fourcc, fps, (W, H))

    print(f"Rendering {total} frames ({W}×{H} @ {fps:.1f} fps) ...")

    frame_idx = 0
    current_pred = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t_sec = frame_idx / fps
        pred  = get_active_prediction(t_sec, predictions, stride_sec, window_sec)

        if pred is not None:
            draw_emotion_pill(frame, pred["emotion"], pred["confidence"])
            draw_confidence_bars(frame, pred["probs"])
            draw_disclaimer(frame)
        draw_timestamp(frame, t_sec)

        writer.write(frame)
        frame_idx += 1

        if frame_idx % int(fps * 5) == 0:
            print(f"  {t_sec:.1f}s / {total/fps:.1f}s  — "
                  f"{pred['emotion'] if pred else 'no prediction'}")

    cap.release()
    writer.release()
    print(f"Frames written to temp file: {tmp_path}")

    # Re-attach original audio via FFmpeg
    print("Re-attaching audio with FFmpeg ...")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(tmp_path),          # silent rendered video
        "-i", str(video_path),        # original (for audio track)
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    tmp_path.unlink(missing_ok=True)

    if result.returncode != 0:
        print("FFmpeg stderr:", result.stderr[-800:])
        # Fall back: rename tmp video without audio
        import shutil
        mp4_fallback = output_path.with_suffix(".noaudio.avi")
        shutil.copy(str(tmp_path), str(mp4_fallback))
        print(f"FFmpeg failed. Silent video saved to: {mp4_fallback}")
        print("Install FFmpeg or run:  !apt-get install -y ffmpeg")
        return

    print(f"\nOverlay video saved to: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Render emotion overlay on MP4.")
    parser.add_argument("video",       type=Path, help="Original input MP4.")
    parser.add_argument("predictions", type=Path, help="JSON from demo_inference.py.")
    parser.add_argument("--output",    type=Path, default=None,
                        help="Output MP4 path (default: <video_stem>_overlay.mp4).")
    parser.add_argument("--stride",    type=float, default=1.0,
                        help="Stride used in demo_inference.py (default: 1.0 s).")
    parser.add_argument("--window",    type=float, default=5.0,
                        help="Window size used in demo_inference.py (default: 5.0 s).")
    args = parser.parse_args()

    if not args.video.exists():
        sys.exit(f"Error: video not found: {args.video}")
    if not args.predictions.exists():
        sys.exit(f"Error: predictions JSON not found: {args.predictions}")

    with open(args.predictions) as f:
        predictions = json.load(f)
    print(f"Loaded {len(predictions)} predictions from {args.predictions}")

    output_path = args.output or args.video.parent / f"{args.video.stem}_overlay.mp4"
    render_overlay(args.video, predictions, output_path,
                   window_sec=args.window, stride_sec=args.stride)


if __name__ == "__main__":
    main()
