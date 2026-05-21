"""Wrappers that load trained per-modality models and expose pre-classifier
pooled features. Used by Day 5's fusion pipeline.

Contract: each extractor has a .extract(raw_inputs) -> (N, D) ndarray method,
where D is the modality's feature dimension and matches the input dims of
fusion.TrimodalAttentionFusion (768/768/960 for AST/ViT/EEGNet).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Make the EAV repo importable so we can pull in EEGNet_tor.
_EAV_PATH = Path(__file__).resolve().parent.parent / "EAV"
if str(_EAV_PATH) not in sys.path:
    sys.path.insert(0, str(_EAV_PATH))


def _strip_dataparallel(state_dict: dict) -> dict:
    """Strip 'module.' prefix if state_dict was saved from a DataParallel model."""
    if not any(k.startswith("module.") for k in state_dict):
        return state_dict
    return {k.removeprefix("module."): v for k, v in state_dict.items()}


# ---------------------------------------------------------------------------
# Audio (AST → 768-d)
# ---------------------------------------------------------------------------

class AudioFeatureExtractor:
    """Wraps the fine-tuned AST classifier; returns mean-pooled encoder hidden state."""

    def __init__(
        self,
        model_path: str,
        state_dict_path: Optional[Path] = None,
        device: Optional[str] = None,
        num_classes: int = 5,
    ):
        from transformers import ASTFeatureExtractor, AutoModelForAudioClassification

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = ASTFeatureExtractor()
        self.model = AutoModelForAudioClassification.from_pretrained(
            model_path, num_labels=num_classes, ignore_mismatched_sizes=True
        ).to(self.device)

        if state_dict_path is not None:
            sd = torch.load(state_dict_path, map_location=self.device)
            self.model.load_state_dict(_strip_dataparallel(sd))

        self.model.eval()
        self.feat_dim = self.model.config.hidden_size  # 768 for AST-base

    @torch.no_grad()
    def extract(self, raw_audio: np.ndarray, batch_size: int = 8) -> np.ndarray:
        """raw_audio: (N, 80000) at 16 kHz → (N, 768)."""
        feats = []
        for i in range(0, len(raw_audio), batch_size):
            batch = list(raw_audio[i : i + batch_size])
            inputs = self.processor(
                batch, sampling_rate=16000, padding="max_length", return_tensors="pt"
            ).input_values.to(self.device)
            out = self.model.audio_spectrogram_transformer(inputs)
            pooled = out.last_hidden_state.mean(dim=1)   # (B, 768)
            feats.append(pooled.cpu().numpy())
        return np.concatenate(feats, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Vision (ViT → 768-d, mean over 25 frames per clip)
# ---------------------------------------------------------------------------

class VisionFeatureExtractor:
    """Wraps the fine-tuned ViT; returns clip-level mean of per-frame [CLS] embeddings."""

    def __init__(
        self,
        model_path: str,
        state_dict_path: Optional[Path] = None,
        device: Optional[str] = None,
        num_classes: int = 5,
    ):
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        self.model = AutoModelForImageClassification.from_pretrained(
            model_path, num_labels=num_classes, ignore_mismatched_sizes=True
        ).to(self.device)

        if state_dict_path is not None:
            sd = torch.load(state_dict_path, map_location=self.device)
            self.model.load_state_dict(_strip_dataparallel(sd))

        self.model.eval()
        self.feat_dim = self.model.config.hidden_size  # 768 for ViT-base

    @torch.no_grad()
    def extract(self, clips: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """clips: (N, 25, H, W, 3) uint8 → (N, 768). Mean-pools per-frame CLS."""
        N = len(clips)
        F = clips.shape[1]   # 25 frames per clip

        # Flatten clip × frame → one big batch of images.
        flat = [frame for clip in clips for frame in clip]

        all_cls = []
        for i in range(0, len(flat), batch_size):
            batch = flat[i : i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").pixel_values.to(self.device)
            out = self.model.vit(inputs)
            cls = out.last_hidden_state[:, 0]            # (B, 768)
            all_cls.append(cls.cpu().numpy())

        feats_per_frame = np.concatenate(all_cls, axis=0)            # (N*F, 768)
        feats_per_clip = feats_per_frame.reshape(N, F, -1).mean(axis=1)
        return feats_per_clip.astype(np.float32)


# ---------------------------------------------------------------------------
# EEG (EEGNet → 960-d, pre-classifier flatten)
# ---------------------------------------------------------------------------

class EEGFeatureExtractor:
    """Wraps the fine-tuned EEGNet; returns the flattened pre-classifier features."""

    def __init__(
        self,
        state_dict_path: Optional[Path] = None,
        device: Optional[str] = None,
        num_classes: int = 5,
    ):
        from CNN_torch.EEGNet_tor import EEGNet_tor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EEGNet_tor(
            nb_classes=num_classes, D=8, F2=64, Chans=30,
            kernLength=300, Samples=500, dropoutRate=0.5,
        ).to(self.device)

        if state_dict_path is not None:
            sd = torch.load(state_dict_path, map_location=self.device)
            self.model.load_state_dict(_strip_dataparallel(sd))

        self.model.eval()
        # 960 = F2(64) * (Samples(500) // 4 // 8)(15)
        self.feat_dim = 64 * (500 // 4 // 8)

    @torch.no_grad()
    def extract(self, eeg: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """eeg: (N, 30, 500) or (N, 1, 30, 500) → (N, 960)."""
        if eeg.ndim == 3:
            eeg = eeg[:, None, :, :]

        feats = []
        m = self.model
        for i in range(0, len(eeg), batch_size):
            x = torch.from_numpy(eeg[i : i + batch_size]).float().to(self.device)
            x = m.firstConv(x);     x = m.firstBN(x);  x = m.elu(x)
            x = m.depthwiseConv(x); x = m.depthwiseBN(x); x = m.elu(x)
            x = m.depthwisePool(x)
            x = m.separableConv(x); x = m.separableBN(x); x = m.elu(x)
            x = m.separablePool(x)
            x = m.flatten(x)        # stop before dense
            feats.append(x.cpu().numpy())
        return np.concatenate(feats, axis=0).astype(np.float32)


if __name__ == "__main__":
    # Smoke test: instantiate each extractor against tiny synthetic input.
    # Loads pretrained-only (state_dict_path=None) just to verify shapes.
    from paths import HF_AUDIO_MODEL, HF_VISION_MODEL

    print("Audio extractor:")
    aud = AudioFeatureExtractor(HF_AUDIO_MODEL)
    fake_audio = np.random.randn(2, 80000).astype(np.float32)
    feats_a = aud.extract(fake_audio)
    print(f"  in : (2, 80000)    out: {feats_a.shape}    dtype: {feats_a.dtype}")
    assert feats_a.shape == (2, 768)

    print("Vision extractor:")
    vis = VisionFeatureExtractor(HF_VISION_MODEL)
    fake_clips = (np.random.rand(2, 25, 56, 56, 3) * 255).astype(np.uint8)
    feats_v = vis.extract(fake_clips, batch_size=16)
    print(f"  in : (2, 25, 56, 56, 3)  out: {feats_v.shape}    dtype: {feats_v.dtype}")
    assert feats_v.shape == (2, 768)

    print("EEG extractor:")
    eeg = EEGFeatureExtractor(state_dict_path=None)
    fake_eeg = np.random.randn(2, 30, 500).astype(np.float32)
    feats_e = eeg.extract(fake_eeg)
    print(f"  in : (2, 30, 500)        out: {feats_e.shape}    dtype: {feats_e.dtype}")
    assert feats_e.shape == (2, 960)

    print("\n✅ All three feature extractors produce the expected dims.")
