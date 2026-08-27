"""Smoke tests for the cross-subject CV pipeline. Runs on CPU, no data needed.

Stages A and B need a GPU and the EAV pickles, so they are not covered here.
Stage C -- where the five fusion variants are trained and scored, and where a
subject-alignment bug would silently corrupt every reported number -- is
covered end to end against fabricated caches.

    python test_cv_pipeline.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

import cv_pipeline as cv
from cv_split import make_folds, test_subjects, train_subjects, validate

N_TRIALS = 40          # per subject; the real run has 400
FOLD = 0
VARIANTS = {"naive_late", "cross_attn", "concat_mlp", "dropout_full", "dropout_av"}


def build_fake_caches(feat_dir: Path, seed: int = 0) -> None:
    """Write feature caches with the exact schema Stage B produces.

    Class centres are GLOBAL across subjects, so a real cross-subject mapping
    exists for the heads to learn; each subject gets a small idiosyncratic
    offset standing in for between-subject variability. This is what makes the
    test diagnostic -- if the pipeline ever misaligned features with labels or
    shuffled subjects, the learned heads would collapse to chance (0.20).
    """
    rng = np.random.default_rng(seed)
    feat_dir.mkdir(parents=True, exist_ok=True)
    centres = {m: rng.normal(0, 3, (cv.N_CLASSES, cv.FEAT_DIMS[m])) for m in cv.MODALITIES}

    for sub in cv.SUBJECTS:
        y = rng.integers(0, cv.N_CLASSES, N_TRIALS)
        offset = {m: rng.normal(0, 0.5, (1, cv.FEAT_DIMS[m])) for m in cv.MODALITIES}

        def feats(m):
            return (centres[m][y] + offset[m]
                    + rng.normal(0, 1.0, (N_TRIALS, cv.FEAT_DIMS[m]))).astype(np.float32)

        def logits():
            z = rng.normal(0, 1, (N_TRIALS, cv.N_CLASSES))
            z[np.arange(N_TRIALS), y] += 2.0      # informative but imperfect
            return z.astype(np.float32)

        np.savez(feat_dir / f"fold{FOLD}_sub{sub:02d}.npz",
                 audio=feats("audio"), vision=feats("vision"), eeg=feats("eeg"),
                 logits_audio=logits(), logits_vision=logits(), logits_eeg=logits(),
                 y=y)


def test_folds_partition() -> None:
    validate()
    folds = make_folds()
    assert sum(len(f) for f in folds) == len(cv.SUBJECTS)
    assert sorted(s for f in folds for s in f) == cv.SUBJECTS
    print("OK: folds are a disjoint, exhaustive partition; each subject tested once")


def test_stage_c() -> None:
    tmp = Path(tempfile.mkdtemp())
    cv.FEAT_DIR = tmp / "feat"
    cv.LOGITS_DIR = tmp / "logits"
    cv.RESULTS_CSV = tmp / "cv5_fusion.csv"
    cv.FUSION_EPOCHS = 15
    build_fake_caches(cv.FEAT_DIR)

    rows: list[dict] = []
    cv.stage_c(FOLD, standardize=True, rows=rows)

    te = test_subjects(FOLD)
    assert {r["variant"] for r in rows} == VARIANTS, "variant set mismatch"
    assert len(rows) == len(VARIANTS) * len(te)

    for v in VARIANTS:
        subs = sorted(r["subject"] for r in rows if r["variant"] == v)
        assert subs == sorted(te), f"{v}: scored {subs}, expected fold {te}"

    for r in rows:
        assert 0.0 <= r["test_acc"] <= 1.0
        assert r["n_trials"] == N_TRIALS

    assert not (set(train_subjects(FOLD)) & set(te)), "train/test subject overlap"

    with np.load(cv.LOGITS_DIR / f"fold{FOLD}.npz") as z:
        assert sorted(set(z["subject"].tolist())) == sorted(te)
        assert z["cross_attn"].shape == (len(te) * N_TRIALS, cv.N_CLASSES)

    cv.append_rows(rows)
    assert cv.RESULTS_CSV.exists()

    learned = [r["test_acc"] for r in rows
               if r["variant"] in {"cross_attn", "concat_mlp", "dropout_full"}]
    mean_learned = float(np.mean(learned))
    assert mean_learned > 0.50, \
        f"learned heads at {mean_learned:.3f} -- cross-subject signal not learned"

    print(f"OK: {len(rows)} rows = {len(VARIANTS)} variants x {len(te)} held-out subjects")
    print("OK: train/test subjects disjoint; saved logits cover exactly the test fold")
    print(f"OK: learned heads mean acc {mean_learned:.3f} (chance=0.20) -- "
          "features and labels stay aligned across the subject boundary")


if __name__ == "__main__":
    test_folds_partition()
    test_stage_c()
    print("\nAll CV pipeline smoke tests PASSED")
    sys.exit(0)
