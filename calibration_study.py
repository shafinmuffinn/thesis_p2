"""How much unlabeled calibration data does subject normalisation need?

Context
-------
Per-participant feature standardisation (`cv_pipeline --subject-norm`) lifts the
learned fusion heads from ~58% to ~75% cross-subject. As run, that protocol has
two properties a sceptical reader will object to:

1. It is transductive on the SCORED trials: each held-out participant is
   normalised with statistics computed from all 400 of their trials, which are
   the same trials the accuracy is then measured on.
2. It leans on a hidden class-balance prior: every EAV participant has exactly
   80 trials per class, so their mean is a perfectly class-balanced average. A
   real calibration session need not be balanced.

This study answers both. The fusion heads trained under --subject-norm are
loaded unchanged; only the normalisation of each HELD-OUT participant varies.
Their statistics are computed from `n` calibration trials, and accuracy is
measured on the remaining, disjoint trials. The calibration trials are drawn
three ways:

    random       uniformly at random -- uses no labels at all
    neutral      Neutral-cued trials only -- models an instructed
                 "sit neutrally for a minute" calibration session
    skewed       class proportions drawn from Dirichlet(0.3) -- a stress
                 test of the class-balance prior

Naive late fusion is scored on exactly the same evaluation trials in every
cell, so each comparison is paired.

Inference only; CPU is fine.

    python calibration_study.py
    python calibration_study.py --folds 0 --seeds 2      # quick check
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

import cv_pipeline as cv
from cv_split import N_FOLDS, test_subjects

NEUTRAL = 0                       # paths.EMOTION_TO_IDX["Neutral"]
N_GRID = [5, 10, 20, 50, 100, 200]
CONDITIONS = ["random", "neutral", "skewed"]
HEADS = {"cross_attn": "cross_attn", "concat_mlp": "concat_mlp", "dropout": "dropout"}
OUT_CSV = cv.RESULTS / "calibration_study.csv"
FIELDS = ["fold", "subject", "condition", "n_calib", "seed", "variant",
          "acc", "n_eval"]


def load_heads(fold: int) -> dict[str, torch.nn.Module]:
    """The heads Stage C trained under --subject-norm for this fold."""
    dims = {"audio_dim": cv.FEAT_DIMS["audio"], "vision_dim": cv.FEAT_DIMS["vision"],
            "eeg_dim": cv.FEAT_DIMS["eeg"]}
    builders = {
        "cross_attn": lambda: cv.TrimodalAttentionFusion(**dims),
        "concat_mlp": lambda: cv.ConcatMLPWrapped(**dims),
        "dropout": lambda: cv.TrimodalAttentionFusion(**dims),
    }
    heads = {}
    for name, build in builders.items():
        path = cv.FUSION_STATE_DIR / f"fold{fold}_subjnorm_{name}.pt"
        if not path.exists():
            raise FileNotFoundError(
                f"{path.name} missing -- run `cv_pipeline.py --stages C "
                f"--subject-norm` for fold {fold} first")
        model = build()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        heads[name] = model
    return heads


def sample_calibration(y: np.ndarray, n: int, condition: str,
                       rng: np.random.Generator) -> np.ndarray | None:
    """Indices of the calibration trials, or None if the draw is impossible."""
    idx = np.arange(len(y))
    if condition == "random":
        return rng.choice(idx, size=n, replace=False) if n < len(y) else None
    if condition == "neutral":
        pool = idx[y == NEUTRAL]
        return rng.choice(pool, size=n, replace=False) if n <= len(pool) else None
    if condition == "skewed":
        props = rng.dirichlet(np.full(cv.N_CLASSES, 0.3))
        counts = rng.multinomial(n, props)
        picks = []
        for c, k in enumerate(counts):
            pool = idx[y == c]
            if k > len(pool):
                return None
            picks.append(rng.choice(pool, size=k, replace=False))
        return np.concatenate(picks)
    raise ValueError(condition)


def normalise_from(feats: dict[str, np.ndarray], calib: np.ndarray,
                   evaluate: np.ndarray) -> dict[str, torch.Tensor]:
    """Standardise `evaluate` rows with statistics from `calib` rows only."""
    out = {}
    for m in cv.MODALITIES:
        X = feats[m]
        mu = X[calib].mean(axis=0, keepdims=True)
        sd = X[calib].std(axis=0, keepdims=True) + 1e-6
        out[m] = torch.from_numpy(((X[evaluate] - mu) / sd).astype(np.float32))
    return out


@torch.no_grad()
def score(heads, x: dict[str, torch.Tensor], y: np.ndarray,
          logits: dict[str, np.ndarray]) -> dict[str, float]:
    res = {}
    for name, model in heads.items():
        out = model(x["audio"], x["vision"], x["eeg"])["logits"].argmax(-1).numpy()
        res["dropout_full" if name == "dropout" else name] = float((out == y).mean())
    zero_eeg = torch.zeros_like(x["eeg"])
    out = heads["dropout"](x["audio"], x["vision"], zero_eeg)["logits"].argmax(-1).numpy()
    res["dropout_av"] = float((out == y).mean())
    probs = np.stack([cv.softmax_np(logits[m]) for m in cv.MODALITIES])
    res["naive_late"] = float((probs.mean(axis=0).argmax(-1) == y).mean())
    return res


def run(folds: list[int], seeds: int) -> list[dict]:
    rows = []
    for k in folds:
        heads = load_heads(k)
        print(f"fold {k}: {len(test_subjects(k))} held-out participants")
        for sub in test_subjects(k):
            with np.load(cv.FEAT_DIR / f"fold{k}_sub{sub:02d}.npz") as z:
                feats = {m: z[m] for m in cv.MODALITIES}
                logits = {m: z[f"logits_{m}"] for m in cv.MODALITIES}
                y = z["y"]
            everything = np.arange(len(y))

            # Reference: the transductive protocol as Stage C ran it.
            x = normalise_from(feats, everything, everything)
            for v, a in score(heads, x, y, logits).items():
                rows.append(dict(fold=k, subject=sub, condition="transductive",
                                 n_calib=len(y), seed=0, variant=v, acc=a,
                                 n_eval=len(y)))

            for cond in CONDITIONS:
                for n in N_GRID:
                    for seed in range(seeds):
                        rng = np.random.default_rng(1000 * sub + 10 * seed + n)
                        calib = sample_calibration(y, n, cond, rng)
                        if calib is None:
                            continue
                        evaluate = np.setdiff1d(everything, calib)
                        x = normalise_from(feats, calib, evaluate)
                        lg = {m: logits[m][evaluate] for m in cv.MODALITIES}
                        for v, a in score(heads, x, y[evaluate], lg).items():
                            rows.append(dict(fold=k, subject=sub, condition=cond,
                                             n_calib=n, seed=seed, variant=v, acc=a,
                                             n_eval=len(evaluate)))
    return rows


def holm(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    order = np.argsort(pvals)
    m, adj, running = len(pvals), [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def summarise(rows: list[dict]) -> None:
    from scipy.stats import wilcoxon

    # One value per (condition, n, variant, subject): the mean over seeds.
    cells: dict[tuple, dict[int, list[float]]] = {}
    for r in rows:
        key = (r["condition"], int(r["n_calib"]), r["variant"])
        cells.setdefault(key, {}).setdefault(int(r["subject"]), []).append(float(r["acc"]))
    per_sub = {k: {s: float(np.mean(v)) for s, v in d.items()} for k, d in cells.items()}

    variants = ["naive_late", "concat_mlp", "dropout_full", "cross_attn", "dropout_av"]
    settings = sorted({(c, n) for c, n, _ in per_sub},
                      key=lambda t: (["transductive"] + CONDITIONS).index(t[0]) * 10_000 + t[1])

    print("\nMean accuracy over participants (each averaged over seeds)")
    print(f"{'condition':13s}{'n':>5s}" + "".join(f"{v:>14s}" for v in variants))
    for cond, n in settings:
        vals = []
        for v in variants:
            d = per_sub.get((cond, n, v), {})
            vals.append(f"{np.mean(list(d.values())):14.4f}" if d else f"{'-':>14s}")
        print(f"{cond:13s}{n:5d}" + "".join(vals))

    # Paired tests: each learned head vs naive, same participants, same eval trials.
    tests, labels = [], []
    for cond, n in settings:
        base = per_sub.get((cond, n, "naive_late"), {})
        for v in variants[1:]:
            d = per_sub.get((cond, n, v), {})
            subs = sorted(set(base) & set(d))
            if len(subs) < 6:
                continue
            a, b = np.array([d[s] for s in subs]), np.array([base[s] for s in subs])
            if np.allclose(a, b):
                continue
            tests.append(float(wilcoxon(a, b).pvalue))
            labels.append((cond, n, v, float((a - b).mean() * 100), len(subs)))
    if not tests:
        return
    adj = holm(tests)
    print(f"\nHead vs naive_late, paired Wilcoxon, Holm-adjusted over {len(tests)} tests")
    print(f"{'condition':13s}{'n':>5s}{'variant':>14s}{'delta':>9s}{'p_holm':>10s}")
    for (cond, n, v, delta, ns), p in zip(labels, adj):
        star = " *" if p < 0.05 else ""
        print(f"{cond:13s}{n:5d}{v:>14s}{delta:+7.2f}pp{p:10.4f}{star}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", default=",".join(str(k) for k in range(N_FOLDS)))
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--summary-only", action="store_true",
                    help="re-summarise an existing calibration_study.csv")
    args = ap.parse_args()

    if args.summary_only:
        with open(OUT_CSV) as f:
            summarise(list(csv.DictReader(f)))
        return 0

    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    rows = run(folds, args.seeds)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows):,} rows -> {OUT_CSV}")
    summarise(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
