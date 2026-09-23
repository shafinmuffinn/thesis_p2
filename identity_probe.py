"""Is subject identity in the features -- and does subject calibration remove it?

Two linear probes per modality, each under two feature treatments:

    global    one mean/variance for everyone (the default Stage C protocol)
    subject   each participant z-scored by their own statistics (--subject-norm)

Probe A -- identity. A 42-way linear classifier predicts WHICH participant a
trial came from, trained on half of every participant's trials and scored on
the other half. Chance is 1/42 = 2.4%. High accuracy under `global` means the
features carry identity; the drop under `subject` measures how much of that
identity the calibration removes.

Probe B -- emotion, cross-subject. A 5-way linear classifier trained on the
fold's training participants and scored on its held-out participants, one
accuracy per participant. This shows which modality's emotion signal is
recovered once identity is removed. Chapter 7 attributes vision's 33-point
cross-subject collapse to identity; if that is right, vision should gain most.

Linear probes are deliberate: they measure what is linearly available in the
representation, not what a flexible model could extract. Each is trained for a
fixed number of full-batch steps with no epoch selection, so no test signal
enters training.

    python identity_probe.py                 # all folds, GPU if available
    python identity_probe.py --folds 0       # quick check
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np
import torch
import torch.nn as nn

import cv_pipeline as cv
from cv_split import N_FOLDS, SUBJECTS, test_subjects, train_subjects

OUT_CSV = cv.RESULTS / "identity_probe.csv"
FIELDS = ["fold", "probe", "modality", "norm", "subject", "acc"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STEPS, LR, WD = 300, 1e-2, 1e-4


def load_all(fold: int) -> dict[str, np.ndarray]:
    """Every participant's features under fold `fold`'s encoders."""
    acc = {m: [] for m in cv.MODALITIES}
    acc.update({"y": [], "subject": []})
    for sub in SUBJECTS:
        with np.load(cv.FEAT_DIR / f"fold{fold}_sub{sub:02d}.npz") as z:
            for m in cv.MODALITIES:
                acc[m].append(z[m])
            acc["y"].append(z["y"])
            acc["subject"].append(np.full(len(z["y"]), sub))
    return {k: np.concatenate(v, axis=0) for k, v in acc.items()}


def global_standardize(d: dict, fit_mask: np.ndarray) -> dict[str, np.ndarray]:
    """One mean/variance, fitted on the rows in `fit_mask`, applied to all rows."""
    out = {}
    for m in cv.MODALITIES:
        mu = d[m][fit_mask].mean(axis=0, keepdims=True)
        sd = d[m][fit_mask].std(axis=0, keepdims=True) + 1e-6
        out[m] = ((d[m] - mu) / sd).astype(np.float32)
    return out


def linear_probe(Xtr, ytr, Xte, n_classes: int, seed: int = 0) -> np.ndarray:
    """Full-batch multinomial logistic regression; returns test predictions."""
    torch.manual_seed(seed)
    Xtr_t = torch.from_numpy(Xtr).to(DEVICE)
    ytr_t = torch.from_numpy(ytr.astype(np.int64)).to(DEVICE)
    model = nn.Linear(Xtr.shape[1], n_classes).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(STEPS):
        opt.zero_grad()
        loss_fn(model(Xtr_t), ytr_t).backward()
        opt.step()
    with torch.no_grad():
        return model(torch.from_numpy(Xte).to(DEVICE)).argmax(-1).cpu().numpy()


def identity_probe(fold: int, d: dict, feats: dict[str, dict], rows: list) -> None:
    """42-way participant decoding from a 50/50 within-participant trial split."""
    rng = np.random.default_rng(fold)
    train_rows = np.zeros(len(d["y"]), dtype=bool)
    for sub in SUBJECTS:
        idx = np.flatnonzero(d["subject"] == sub)
        train_rows[rng.choice(idx, size=len(idx) // 2, replace=False)] = True
    sid = np.searchsorted(SUBJECTS, d["subject"])            # 0..41

    # Subject normalisation for THIS probe must take its statistics from the
    # probe's training half only. Using all of a participant's trials makes the
    # two halves' means mirror images once the overall mean is subtracted, so a
    # probe learns anti-correlated means and scores systematically BELOW chance
    # -- which would overstate how much identity the calibration removes.
    subj_from_train = {}
    for m in cv.MODALITIES:
        X = d[m].astype(np.float32, copy=True)
        for sub in SUBJECTS:
            rows_s = d["subject"] == sub
            ref = rows_s & train_rows
            mu = X[ref].mean(axis=0, keepdims=True)
            sd = X[ref].std(axis=0, keepdims=True) + 1e-6
            X[rows_s] = (X[rows_s] - mu) / sd
        subj_from_train[m] = X
    feats = {"global": feats["global"], "subject": subj_from_train}

    for norm, F in feats.items():
        for m in cv.MODALITIES:
            pred = linear_probe(F[m][train_rows], sid[train_rows],
                                F[m][~train_rows], len(SUBJECTS))
            acc = float((pred == sid[~train_rows]).mean())
            rows.append(dict(fold=fold, probe="identity", modality=m, norm=norm,
                             subject="all", acc=acc))
            print(f"    identity  {m:7s} {norm:8s} {acc:.4f}")


def emotion_probe(fold: int, d: dict, feats: dict[str, dict], rows: list) -> None:
    """5-way emotion decoding, trained on training participants, per held-out one."""
    tr = np.isin(d["subject"], train_subjects(fold))
    for norm, F in feats.items():
        for m in cv.MODALITIES:
            pred = linear_probe(F[m][tr], d["y"][tr], F[m][~tr], cv.N_CLASSES)
            y_te, s_te = d["y"][~tr], d["subject"][~tr]
            for sub in test_subjects(fold):
                sel = s_te == sub
                rows.append(dict(fold=fold, probe="emotion", modality=m, norm=norm,
                                 subject=sub, acc=float((pred[sel] == y_te[sel]).mean())))
            print(f"    emotion   {m:7s} {norm:8s} {float((pred == y_te).mean()):.4f}")


def summarise(rows: list[dict]) -> None:
    from scipy.stats import wilcoxon

    print("\n" + "=" * 60)
    print(f"A. IDENTITY decoding, 42-way (chance {1 / len(SUBJECTS):.3f}), mean over folds")
    print("=" * 60)
    print(f"{'modality':10s}{'global':>10s}{'subject':>10s}{'removed':>10s}")
    for m in cv.MODALITIES:
        g = np.mean([float(r["acc"]) for r in rows if r["probe"] == "identity"
                     and r["modality"] == m and r["norm"] == "global"])
        s = np.mean([float(r["acc"]) for r in rows if r["probe"] == "identity"
                     and r["modality"] == m and r["norm"] == "subject"])
        print(f"{m:10s}{g:10.3f}{s:10.3f}{(g - s) * 100:+9.1f}pp")

    print("\n" + "=" * 60)
    print("B. EMOTION decoding cross-subject, linear probe, 5-way (chance 0.200)")
    print("=" * 60)
    print(f"{'modality':10s}{'global':>10s}{'subject':>10s}{'gain':>9s}{'p':>9s}  better")
    for m in cv.MODALITIES:
        per = {}
        for norm in ("global", "subject"):
            per[norm] = {int(r["subject"]): float(r["acc"]) for r in rows
                         if r["probe"] == "emotion" and r["modality"] == m
                         and r["norm"] == norm}
        subs = sorted(set(per["global"]) & set(per["subject"]))
        g = np.array([per["global"][s] for s in subs])
        s_ = np.array([per["subject"][s] for s in subs])
        p = wilcoxon(s_, g).pvalue if not np.allclose(s_, g) else 1.0
        print(f"{m:10s}{g.mean():10.3f}{s_.mean():10.3f}{(s_.mean() - g.mean()) * 100:+8.1f}"
              f"pp{p:9.4f}  {int((s_ > g).sum())}/{len(subs)}")
    print("\nThree p-values in B; with Holm the smallest is multiplied by 3.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", default=",".join(str(k) for k in range(N_FOLDS)))
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    if args.summary_only:
        with open(OUT_CSV) as f:
            summarise(list(csv.DictReader(f)))
        return 0

    rows: list[dict] = []
    for k in [int(x) for x in args.folds.split(",") if x.strip() != ""]:
        print(f"fold {k} (device={DEVICE})")
        d = load_all(k)
        tr = np.isin(d["subject"], train_subjects(k))
        feats = {"global": global_standardize(d, tr),
                 "subject": cv.subject_standardize(d)}
        identity_probe(k, d, feats, rows)
        emotion_probe(k, d, feats, rows)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {OUT_CSV}")
    summarise(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
