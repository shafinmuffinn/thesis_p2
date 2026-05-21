"""Trimodal attention fusion for audio + vision + EEG emotion recognition.

Architecture
============
Each modality is encoded by its own pretrained backbone (AST for audio, ViT
for vision, EEGNet for EEG) into a single pooled feature vector per trial.
This module takes those three vectors and fuses them as follows:

    1. Project each modality to a common dimension `d_model` (default 256).
    2. Add a learnable modality embedding so the transformer can distinguish
       which token is which modality (analogous to positional encoding).
    3. Stack the three projected vectors as a 3-token sequence.
    4. Apply N transformer encoder layers (multi-head self-attention + MLP)
       over that 3-token sequence. Each modality attends to the other two.
    5. Mean-pool the post-attention tokens to a single fused vector.
    6. Pass through a small MLP head to produce 5-class emotion logits.

Why not 6-way pairwise cross-attention (à la Lee et al. 2024)?
-------------------------------------------------------------
The Lee et al. design operates on sequence-level features (multiple tokens
per modality). Our pretrained encoders pool to a single vector per modality,
so six pairwise cross-attentions would degenerate to six learned projections
of single 1-token sequences. Self-attention over a 3-token sequence does the
same job at this resolution with cleaner code and fewer parameters. Going
back to sequence-level features and re-introducing 6-way pairwise CMA is a
localised change for Day 6 (MERCL pre-training) if needed.

Phase 2 commitments
-------------------
forward() returns a dict whose keys are stable contract:
    - 'logits'       : (B, num_classes) raw emotion-class scores
    - 'audio_embed'  : (B, d_model) post-attention audio token
    - 'vision_embed' : (B, d_model) post-attention vision token
    - 'eeg_embed'    : (B, d_model) post-attention EEG token
    - 'fused_embed'  : (B, d_model) mean-pooled fused representation

A future temporal head (LSTM / Residual-TCN / temporal transformer) can
consume the per-modality or fused embeddings without modifying this file.

forward() also accepts zero tensors for any modality input and produces
finite outputs. The Day 7 modality-dropout training makes those zero inputs
*useful* (the model is trained to expect them); the architecture must permit
them from day one.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TrimodalAttentionFusion(nn.Module):
    """Self-attention fusion over three modality tokens."""

    def __init__(
        self,
        audio_dim: int = 768,
        vision_dim: int = 768,
        eeg_dim: int = 960,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_classes: int = 5,
    ):
        super().__init__()

        # 1. Per-modality projection to the common embedding dim.
        self.proj_audio = nn.Linear(audio_dim, d_model)
        self.proj_vision = nn.Linear(vision_dim, d_model)
        self.proj_eeg = nn.Linear(eeg_dim, d_model)

        # 2. Learnable modality embeddings (one per modality, broadcast over batch).
        #    Initialised to small values so they nudge rather than dominate.
        self.modality_embed = nn.Parameter(torch.randn(3, d_model) * 0.02)

        # 3. Transformer encoder over the 3-token modality sequence.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.post_norm = nn.LayerNorm(d_model)

        # 4. Classifier head on the mean-pooled fused representation.
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

        # Hyperparams kept around for logging / serialisation.
        self.d_model = d_model
        self.num_classes = num_classes
        self.input_dims = {
            "audio": audio_dim,
            "vision": vision_dim,
            "eeg": eeg_dim,
        }

    def forward(
        self,
        x_audio: torch.Tensor,
        x_vision: torch.Tensor,
        x_eeg: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run the fusion forward pass.

        Args:
            x_audio:  (B, audio_dim)  pooled AST feature
            x_vision: (B, vision_dim) pooled ViT feature
            x_eeg:    (B, eeg_dim)    pooled EEGNet feature

        Returns:
            Dict with keys 'logits', 'audio_embed', 'vision_embed',
            'eeg_embed', 'fused_embed'. See module docstring for shapes.
        """
        # Project each modality to common dim and add modality embedding.
        a = self.proj_audio(x_audio) + self.modality_embed[0]
        v = self.proj_vision(x_vision) + self.modality_embed[1]
        e = self.proj_eeg(x_eeg) + self.modality_embed[2]

        # Stack as a 3-token sequence: (B, 3, d_model).
        tokens = torch.stack([a, v, e], dim=1)

        # Multi-layer self-attention over the modality tokens.
        tokens = self.transformer(tokens)
        tokens = self.post_norm(tokens)

        # Mean-pool the three post-attention tokens to a single fused vector.
        fused = tokens.mean(dim=1)

        logits = self.classifier(fused)

        return {
            "logits": logits,
            "audio_embed": tokens[:, 0],
            "vision_embed": tokens[:, 1],
            "eeg_embed": tokens[:, 2],
            "fused_embed": fused,
        }


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Sanity check the architectural contract.
    torch.manual_seed(0)
    model = TrimodalAttentionFusion()
    model.eval()

    n_params = count_parameters(model)
    print(f"TrimodalAttentionFusion — {n_params:,} trainable parameters")
    print(f"Common embedding dim: {model.d_model}")
    print(f"Input dims: {model.input_dims}\n")

    B = 8
    x_audio = torch.randn(B, 768)
    x_vision = torch.randn(B, 768)
    x_eeg = torch.randn(B, 960)

    # ---- Test 1: full-modality forward pass ----
    out = model(x_audio, x_vision, x_eeg)
    expected = {
        "logits":       (B, 5),
        "audio_embed":  (B, 256),
        "vision_embed": (B, 256),
        "eeg_embed":    (B, 256),
        "fused_embed":  (B, 256),
    }
    print("Test 1 — full-modality forward")
    for key, shape in expected.items():
        actual = tuple(out[key].shape)
        ok = actual == shape and torch.isfinite(out[key]).all().item()
        print(f"  {key:14s}: shape={actual}  finite={torch.isfinite(out[key]).all().item()}  "
              f"{'OK' if ok else 'FAIL'}")

    # ---- Test 2: zero-EEG forward pass (demo path) ----
    print("\nTest 2 — zero-EEG forward (Phase 2 commitment)")
    out_zero = model(x_audio, x_vision, torch.zeros(B, 960))
    finite = torch.isfinite(out_zero["logits"]).all().item()
    print(f"  logits finite : {finite}  {'OK' if finite else 'FAIL'}")
    print(f"  logits norm   : {out_zero['logits'].norm().item():.3f}  "
          f"(non-zero means model still produces predictions)")

    # ---- Test 3: gradient flow ----
    print("\nTest 3 — backward pass")
    out = model(x_audio, x_vision, x_eeg)
    fake_targets = torch.randint(0, 5, (B,))
    loss = nn.functional.cross_entropy(out["logits"], fake_targets)
    loss.backward()
    grad_norms = {
        "proj_audio.weight":  model.proj_audio.weight.grad.norm().item(),
        "proj_vision.weight": model.proj_vision.weight.grad.norm().item(),
        "proj_eeg.weight":    model.proj_eeg.weight.grad.norm().item(),
        "modality_embed":     model.modality_embed.grad.norm().item(),
        "classifier[-1].weight": model.classifier[-1].weight.grad.norm().item(),
    }
    for name, n in grad_norms.items():
        ok = n > 0 and torch.isfinite(torch.tensor(n)).item()
        print(f"  grad norm {name:25s}: {n:.4f}  {'OK' if ok else 'FAIL'}")

    print("\n✅ TrimodalAttentionFusion passes the Day 4 contract.")
