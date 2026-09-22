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
TRAINED = {"cross_attn", "concat_mlp", "dropout_full"}
VARIANTS = ({"audio_only", "vision_only", "eeg_only", "naive_late", "dropout_av"}
            | TRAINED)


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


def test_vision_pooling() -> None:
    """Vision pooling must hand the trainer clips, not pre-flattened frames.

    Regression test for the Stage A crash: ImageClassifierTrainer flattens
    clips into frames itself (`for clip in image_list for img in clip`) and
    derives frame_per_sample from tr_x.shape[1]. Pre-flattening made it iterate
    rows of a frame, and giving fit/val different frame counts desynchronised
    the validation labels.
    """
    H = W = 56
    CLIPS, FRAMES = 400, 25

    def fake_load(sub, modality):
        rng = np.random.default_rng(sub)
        x = rng.integers(0, 255, (CLIPS, FRAMES, H, W, 3), dtype=np.uint8)
        return x, rng.integers(0, cv.N_CLASSES, CLIPS)

    original, cv.load_all_trials = cv.load_all_trials, fake_load
    try:
        fit, val = [1, 2, 3, 4], [5, 6]
        fpc, cps = cv.plan_vision_sampling(len(fit), budget_gb=1.0)

        budget_frames = int(1.0 * 1024 ** 3 / cv.PREPROC_FRAME_BYTES)
        assert 1 <= fpc <= FRAMES and 1 <= cps <= CLIPS
        assert len(fit) * cps * fpc <= budget_frames, "plan exceeds its RAM budget"

        tr_x, tr_y = cv.pool_subjects(fit, "vision", frames_per_clip=fpc,
                                      clips_per_subject=cps)
        va_x, va_y = cv.pool_subjects(val, "vision", frames_per_clip=fpc,
                                      clips_per_subject=cps)

        # Clip structure preserved, labels per clip (not per frame).
        assert tr_x.shape == (len(fit) * cps, fpc, H, W, 3), f"bad shape {tr_x.shape}"
        assert tr_y.shape == (len(fit) * cps,), f"bad label shape {tr_y.shape}"

        # The trainer reads frame_per_sample off tr_x and applies it to BOTH
        # splits, so val must carry the same frames per clip.
        assert tr_x.shape[1] == va_x.shape[1], "fit/val frames-per-clip mismatch"

        # Simulate the trainer exactly.
        frame_per_sample = np.shape(tr_x)[1]
        flat = [img for clip in tr_x for img in clip]
        assert len(flat) == len(tr_x) * frame_per_sample
        assert all(im.shape == (H, W, 3) for im in flat), \
            "flattening did not yield (H, W, 3) images"
        assert len(np.repeat(tr_y, frame_per_sample)) == len(flat), \
            "labels and frames desynchronised"
        assert len(np.repeat(va_y, frame_per_sample)) == len(va_x) * frame_per_sample, \
            "validation labels desynchronised"

        print(f"OK: vision pooling -> {tr_x.shape} clips, {fpc} frames/clip, "
              f"{cps} clips/subject; trainer flattening and labels align")
    finally:
        cv.load_all_trials = original


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

    learned = [r["test_acc"] for r in rows if r["variant"] in TRAINED]
    mean_learned = float(np.mean(learned))
    assert mean_learned > 0.50, \
        f"learned heads at {mean_learned:.3f} -- cross-subject signal not learned"

    # Inner-validation selection must not have touched the held-out fold: the
    # fit/val subjects both have to come from the training fold.
    fit_subs, val_subs = cv._inner_split(train_subjects(FOLD))
    assert not (set(val_subs) & set(te)), "inner-val subjects leak into test fold"
    assert not (set(fit_subs) & set(te)), "fit subjects leak into test fold"
    assert not (set(fit_subs) & set(val_subs)), "fit and inner-val overlap"

    print(f"OK: {len(rows)} rows = {len(VARIANTS)} variants x {len(te)} held-out subjects")
    print("OK: train/test subjects disjoint; saved logits cover exactly the test fold")
    print(f"OK: learned heads mean acc {mean_learned:.3f} (chance=0.20) -- "
          "features and labels stay aligned across the subject boundary")


def test_subject_normalisation() -> None:
    """Per-participant standardisation must remove per-participant offsets, and
    the two protocols must not overwrite each other's artefacts."""
    rng = np.random.default_rng(0)
    subj = np.repeat([1, 2, 3], 50)
    d = {m: rng.normal(loc=subj[:, None] * 10.0, scale=2.0, size=(150, 4))
         for m in cv.MODALITIES}
    d["subject"] = subj
    out = cv.subject_standardize(d)
    for s in (1, 2, 3):
        mask = subj == s
        assert abs(out["audio"][mask].mean()) < 1e-4
        assert abs(out["audio"][mask].std() - 1.0) < 1e-2

    tmp = Path(tempfile.mkdtemp())
    cv.FEAT_DIR, cv.LOGITS_DIR = tmp / "feat", tmp / "logits"
    cv.FUSION_STATE_DIR, cv.FUSION_EPOCHS = tmp / "heads", 2
    build_fake_caches(cv.FEAT_DIR)
    rows_a: list[dict] = []
    rows_b: list[dict] = []
    cv.stage_c(FOLD, standardize=True, rows=rows_a, subject_norm=False)
    cv.stage_c(FOLD, standardize=True, rows=rows_b, subject_norm=True)

    assert {r["standardized"] for r in rows_a} == {1}
    assert {r["standardized"] for r in rows_b} == {2}
    names = {p.name for p in cv.LOGITS_DIR.glob("*.npz")}
    assert {f"fold{FOLD}.npz", f"fold{FOLD}_subjnorm.npz"} <= names, names
    print("OK: per-participant standardisation removes subject offsets; "
          "the two protocols write separate artefacts and cannot collide")


if __name__ == "__main__":
    test_folds_partition()
    test_vision_pooling()
    test_stage_c()
    test_subject_normalisation()
    print("\nAll CV pipeline smoke tests PASSED")
    sys.exit(0)
