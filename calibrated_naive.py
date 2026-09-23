"""Does calibrated learned fusion still beat averaging when averaging is calibrated too?

Context
-------
Chapter 8 standardises each participant's FUSION FEATURES with their own
statistics and finds every learned head 11-14 pp above naive late fusion. Naive
late fusion averages encoder softmaxes, never reads those features, and so was
never calibrated: the comparison is calibrated heads vs UNCALIBRATED averaging
(Limitation 3). This script gives averaging the analogous label-free,
per-participant correction and re-runs the comparison on the same trials.

Pre-specified design (fixed before any result was seen)
-------------------------------------------------------
Calibration of a participant's encoder logits z_m (one modality m, 5 classes)
from their calibration clips C:

    naive_centred   PRIMARY.   z_m - mean_C(z_m), then mean softmax over modalities.
                    Removes the participant's per-class bias in each encoder --
                    the logit-space counterpart of subtracting their feature mean.
    naive_zscored   SECONDARY. (z_m - mean_C(z_m)) / std_C(z_m), then mean softmax.
    audio_centred   best single modality cross-subject (audio), centred the same
                    way -- the "calibrated best unimodal" reference.

Settings (same draws as calibration_study.py, so rows are paired with it):
    transductive    C = all 400 trials, scored on all 400
    random n=20     C = 20 random trials, scored on the other 380 (5 seeds)
    random n=50     C = 50 random trials, scored on the other 350 (5 seeds)

The calibrated heads are re-scored on the same evaluation trials; their means
must reproduce calibration_study.csv (sanity check printed at the end).

Tests (paired Wilcoxon over 42 participants, Holm over all 21 in one family):
per setting, each of the 4 heads vs naive_centred; naive_centred vs naive_late;
naive_zscored vs naive_late; naive_centred vs audio_centred.

    python calibrated_naive.py
    python calibrated_naive.py --selftest        # CPU check, no data needed
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

OUT_NAME = "calibrated_naive.csv"
FIELDS = ["fold", "subject", "setting", "n_calib", "seed", "variant", "acc"]
HEADS = ["concat_mlp", "dropout_full", "cross_attn", "dropout_av"]
SETTINGS = [("transductive", 400), ("random", 20), ("random", 50)]
SEEDS = 5


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def calibrate_logits(z: np.ndarray, calib: np.ndarray, evaluate: np.ndarray,
                     mode: str) -> np.ndarray:
    """Per-participant logit correction from the calibration rows only."""
    mu = z[calib].mean(axis=0, keepdims=True)
    out = z[evaluate] - mu
    if mode == "zscored":
        out = out / (z[calib].std(axis=0, keepdims=True) + 1e-6)
    elif mode != "centred":
        raise ValueError(mode)
    return out


def naive_scores(logits: dict[str, np.ndarray], y: np.ndarray, calib: np.ndarray,
                 evaluate: np.ndarray) -> dict[str, float]:
    ye = y[evaluate]
    acc = lambda p: float((p.argmax(-1) == ye).mean())
    res = {}
    for mode in ("centred", "zscored"):
        probs = [_softmax(calibrate_logits(logits[m], calib, evaluate, mode))
                 for m in ("audio", "vision", "eeg")]
        res[f"naive_{mode}"] = acc(np.mean(probs, axis=0))
    res["audio_centred"] = acc(calibrate_logits(logits["audio"], calib, evaluate, "centred"))
    return res


def run(folds: list[int]) -> list[dict]:
    import cv_pipeline as cv
    from calibration_study import load_heads, normalise_from, sample_calibration, score
    from cv_split import test_subjects

    rows = []
    for k in folds:
        heads = load_heads(k)
        print(f"fold {k}: {len(test_subjects(k))} held-out participants", flush=True)
        for sub in test_subjects(k):
            with np.load(cv.FEAT_DIR / f"fold{k}_sub{sub:02d}.npz") as z:
                feats = {m: z[m] for m in cv.MODALITIES}
                logits = {m: z[f"logits_{m}"] for m in cv.MODALITIES}
                y = z["y"]
            everything = np.arange(len(y))
            for setting, n in SETTINGS:
                seeds = [0] if setting == "transductive" else range(SEEDS)
                for seed in seeds:
                    if setting == "transductive":
                        calib = evaluate = everything
                    else:
                        # Identical draw to calibration_study.run -> paired rows.
                        rng = np.random.default_rng(1000 * sub + 10 * seed + n)
                        calib = sample_calibration(y, n, "random", rng)
                        evaluate = np.setdiff1d(everything, calib)
                    x = normalise_from(feats, calib, evaluate)
                    lg = {m: logits[m][evaluate] for m in cv.MODALITIES}
                    res = score(heads, x, y[evaluate], lg)          # heads + naive_late
                    res.update(naive_scores(logits, y, calib, evaluate))
                    for v, a in res.items():
                        rows.append(dict(fold=k, subject=sub, setting=setting,
                                         n_calib=len(calib), seed=seed, variant=v, acc=a))
    return rows


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p); m = len(p); adj = [0.0] * m; run_ = 0.0
    for rank, i in enumerate(order):
        run_ = max(run_, min(1.0, (m - rank) * p[i])); adj[i] = run_
    return adj


def summarise(rows: list[dict]) -> None:
    from scipy.stats import wilcoxon

    cells: dict[tuple, dict[int, list[float]]] = {}
    for r in rows:
        key = (r["setting"], int(r["n_calib"]), r["variant"])
        cells.setdefault(key, {}).setdefault(int(r["subject"]), []).append(float(r["acc"]))
    per = {k: {s: float(np.mean(v)) for s, v in d.items()} for k, d in cells.items()}

    variants = ["naive_late", "naive_centred", "naive_zscored", "audio_centred"] + HEADS
    settings = [(s, n) for s, n in SETTINGS]
    print(f"\n{'setting':14s}{'n':>5s}" + "".join(f"{v:>15s}" for v in variants))
    for s, n in settings:
        vals = [np.mean(list(per[(s, n, v)].values())) if (s, n, v) in per else np.nan
                for v in variants]
        print(f"{s:14s}{n:5d}" + "".join(f"{x:15.4f}" for x in vals))

    tests, labels = [], []
    for s, n in settings:
        pairs = [(h, "naive_centred") for h in HEADS] + [
            ("naive_centred", "naive_late"), ("naive_zscored", "naive_late"),
            ("naive_centred", "audio_centred")]
        for a, b in pairs:
            A, B = per[(s, n, a)], per[(s, n, b)]
            subs = sorted(set(A) & set(B))
            x, yv = np.array([A[i] for i in subs]), np.array([B[i] for i in subs])
            p = 1.0 if np.allclose(x, yv) else float(wilcoxon(x, yv).pvalue)
            tests.append(p)
            labels.append((s, n, a, b, (x - yv).mean() * 100, int((x > yv).sum()), len(subs)))
    adj = holm(tests)
    print(f"\nPaired Wilcoxon, Holm over {len(tests)} tests")
    for (s, n, a, b, d, better, ns), p0, p in zip(labels, tests, adj):
        print(f"  {s:12s}{n:4d}  {a:14s} vs {b:14s} {d:+7.2f}pp  p={p0:.3g}  "
              f"p_holm={p:.3g}{' *' if p < 0.05 else ''}  ({better}/{ns})")

    print("\nSanity: heads/naive_late must match calibration_study.csv "
          "(transductive concat_mlp 0.7593, naive_late 0.6215; random n=20 concat_mlp 0.7384).")


def selftest() -> int:
    """Centring must undo a participant-specific constant logit bias exactly."""
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(5), 80)
    clean = rng.normal(size=(400, 5)); clean[np.arange(400), y] += 3.0
    clean -= clean.mean(axis=0, keepdims=True)              # zero-mean, like a balanced participant
    biased = clean + np.array([4.0, 0, 0, 0, 0])             # this person looks "Neutral"
    allr = np.arange(400)
    before = float((biased.argmax(-1) == y).mean())
    after = float((calibrate_logits(biased, allr, allr, "centred").argmax(-1) == y).mean())
    target = float((clean.argmax(-1) == y).mean())
    assert abs(after - target) < 1e-9, (after, target)
    assert before < after, (before, after)
    # Disjoint rows: statistics must come from calib rows only.
    calib, ev = allr[:50], allr[50:]
    out = calibrate_logits(biased, calib, ev, "zscored")
    assert out.shape == (350, 5) and np.isfinite(out).all()
    print(f"selftest OK: biased {before:.3f} -> centred {after:.3f} (clean {target:.3f})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    from paths import RESULTS
    out = RESULTS / OUT_NAME
    if args.summary_only:
        with open(out) as f:
            summarise(list(csv.DictReader(f)))
        return 0
    rows = run([int(x) for x in args.folds.split(",") if x.strip()])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {len(rows):,} rows -> {out}")
    summarise(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
