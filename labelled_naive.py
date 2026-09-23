"""Can averaging use a few labels as well as the learned heads do?

Context
-------
fewshot_calibration.py shows that fine-tuning the calibrated fusion heads on n
LABELLED clips from a new participant adds 5.6-7.4 pp at n = 50. Its comparison
point was UNLABELLED averaging, so it cannot say whether learned fusion uses
labels better than averaging could. This gives averaging the same labelled
clips and the simplest ways to use them.

Pre-specified design (fixed before any result was seen)
-------------------------------------------------------
Same participants, same k (1, 2, 4, 10 per class), same seeds and the SAME
stratified draw as fewshot_calibration.py, so every row is paired with it.
Each modality's logits are first centred with the clips' mean (the label-free
calibration of calibrated_naive.py). Then, on the labelled clips:

    naive_centred    no labels used (reference; = calibrated averaging)
    naive_bias       PRIMARY. mean-softmax averaging plus 5 class-bias terms b
                     added to log(mean softmax); b fitted by cross-entropy.
    naive_weighted   SECONDARY. softmax(w)-weighted average of the three
                     modalities' softmaxes (3 weights) plus the 5 class biases.

Fitting: parameters start at zero (= plain calibrated averaging), full-batch
Adam, lr 0.05, 200 steps, L2 penalty lambda on all fitted parameters. lambda in
{0.001, 0.01, 0.1} is chosen per fold and per variant on the fold's three
inner-validation participants by running this same protocol on them -- the same
selection rule the learned heads received. Held-out participants never
influence the choice.

Tests (paired Wilcoxon over 42 participants, seed-averaged; one Holm family of
24): per k, each labelled head (from fewshot_calibration.csv) vs naive_bias
(16); naive_bias vs naive_centred (4); naive_weighted vs naive_centred (4).
Built-in check: a 0-step fit must equal naive_centred exactly.

    python labelled_naive.py                 # needs fewshot_calibration.csv
    python labelled_naive.py --selftest
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np
import torch

from fewshot_calibration import K_GRID, draw_stratified

LAMBDAS = [1e-3, 1e-2, 1e-1]
LR, STEPS = 0.05, 200
FITTED = ["naive_bias", "naive_weighted"]
HEADS = ["concat_mlp", "dropout_full", "cross_attn", "dropout_av"]
OUT_NAME = "labelled_naive.csv"
FIELDS = ["fold", "subject", "k", "seed", "variant", "acc"]
MODS = ["audio", "vision", "eeg"]


def _probs(logits: dict[str, np.ndarray], mu: dict[str, np.ndarray], idx: np.ndarray) -> torch.Tensor:
    """(3, len(idx), C) per-modality softmax of centred logits."""
    z = np.stack([logits[m][idx] - mu[m] for m in MODS])
    return torch.softmax(torch.from_numpy(z).float(), dim=-1)


def predict(P: torch.Tensor, w: torch.Tensor | None, b: torch.Tensor) -> torch.Tensor:
    avg = P.mean(0) if w is None else torch.einsum("m,mnc->nc", torch.softmax(w, 0), P)
    return torch.log(avg + 1e-12) + b


def fit(P: torch.Tensor, y: torch.Tensor, variant: str, lam: float, steps: int = STEPS):
    C = P.shape[-1]
    b = torch.zeros(C, requires_grad=True)
    w = torch.zeros(3, requires_grad=True) if variant == "naive_weighted" else None
    params = [b] + ([w] if w is not None else [])
    opt = torch.optim.Adam(params, lr=LR)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(predict(P, w, b), y)
        loss = loss + lam * sum((p ** 2).sum() for p in params)
        loss.backward()
        opt.step()
    return (w.detach() if w is not None else None), b.detach()


def one_draw(logits, y, k, seed, sub, lams: dict[str, float] | None):
    """{variant or 'variant@lam': acc} for one draw; lams=None -> grid mode."""
    rng = np.random.default_rng(7000 * sub + 10 * seed + k)      # = fewshot_calibration
    calib = draw_stratified(y, k, rng)
    evaluate = np.setdiff1d(np.arange(len(y)), calib)
    mu = {m: logits[m][calib].mean(axis=0) for m in MODS}
    Pc, Pe = _probs(logits, mu, calib), _probs(logits, mu, evaluate)
    yc = torch.from_numpy(y[calib]).long()
    acc = lambda s: float((s.argmax(-1).numpy() == y[evaluate]).mean())
    out = {"naive_centred": acc(predict(Pe, None, torch.zeros(Pe.shape[-1])))}
    for v in FITTED:
        for lam in (LAMBDAS if lams is None else [lams[v]]):
            w, b = fit(Pc, yc, v, lam)
            out[v if lams is not None else f"{v}@{lam}"] = acc(predict(Pe, w, b))
    return out


def load(fold: int, sub: int):
    import cv_pipeline as cv
    with np.load(cv.FEAT_DIR / f"fold{fold}_sub{sub:02d}.npz") as z:
        return {m: z[f"logits_{m}"] for m in MODS}, z["y"]


def select(fold: int) -> dict[str, float]:
    import cv_pipeline as cv
    from cv_split import train_subjects
    _, val_subs = cv._inner_split(train_subjects(fold))
    scores: dict[str, list[float]] = {}
    for sub in val_subs:
        logits, y = load(fold, sub)
        for k in K_GRID:
            for seed in range(2):
                for key, a in one_draw(logits, y, k, seed, sub, None).items():
                    scores.setdefault(key, []).append(a)
    chosen = {v: max(LAMBDAS, key=lambda l: np.mean(scores[f"{v}@{l}"])) for v in FITTED}
    print(f"  fold {fold} inner-val lambda: " + "  ".join(
        f"{v}={chosen[v]:g} (" + ", ".join(f"{l:g}:{np.mean(scores[f'{v}@{l}']):.4f}"
                                          for l in LAMBDAS) + ")" for v in FITTED), flush=True)
    return chosen


def run(folds: list[int], seeds: int) -> list[dict]:
    from cv_split import test_subjects
    torch.manual_seed(0)
    rows = []
    for k_fold in folds:
        lams = select(k_fold)
        for sub in test_subjects(k_fold):
            logits, y = load(k_fold, sub)
            for k in K_GRID:
                for seed in range(seeds):
                    for v, a in one_draw(logits, y, k, seed, sub, lams).items():
                        rows.append(dict(fold=k_fold, subject=sub, k=k, seed=seed,
                                         variant=v, acc=a))
    return rows


def summarise(rows: list[dict], fewshot: list[dict]) -> None:
    from scipy.stats import wilcoxon

    def per(src, key_fn):
        cells: dict[tuple, dict[int, list[float]]] = {}
        for r in src:
            key = key_fn(r)
            if key is not None:
                cells.setdefault(key, {}).setdefault(int(r["subject"]), []).append(float(r["acc"]))
        return {k: {s: float(np.mean(v)) for s, v in d.items()} for k, d in cells.items()}

    mine = per(rows, lambda r: (int(r["k"]), r["variant"]))
    heads = per(fewshot, lambda r: (int(r["k"]), r["variant"]) if r["mode"] == "labelled" else None)
    table = {**mine, **{(k, f"{v} (labelled)"): d for (k, v), d in heads.items()}}
    cols = ["naive_centred", "naive_bias", "naive_weighted"] + [f"{h} (labelled)" for h in HEADS]
    print(f"\n{'k':>3s}{'n':>4s}" + "".join(f"{c:>24s}" for c in cols))
    for k in K_GRID:
        print(f"{k:3d}{5 * k:4d}" + "".join(
            f"{np.mean(list(table[(k, c)].values())):24.4f}" if (k, c) in table else f"{'-':>24s}"
            for c in cols))

    tests, labels = [], []
    for k in K_GRID:
        pairs = [(f"{h} (labelled)", "naive_bias") for h in HEADS] + [
            ("naive_bias", "naive_centred"), ("naive_weighted", "naive_centred")]
        for a, b in pairs:
            A, B = table[(k, a)], table[(k, b)]
            subs = sorted(set(A) & set(B))
            x, z = np.array([A[s] for s in subs]), np.array([B[s] for s in subs])
            tests.append(1.0 if np.allclose(x, z) else float(wilcoxon(x, z).pvalue))
            labels.append((k, a, b, (x - z).mean() * 100, int((x > z).sum()), len(subs)))
    order = np.argsort(tests); m = len(tests); adj = [0.0] * m; r = 0.0
    for rank, i in enumerate(order):
        r = max(r, min(1.0, (m - rank) * tests[i])); adj[i] = r
    print(f"\nPaired Wilcoxon, Holm over {m} tests")
    for (k, a, b, d, better, n), p0, p in zip(labels, tests, adj):
        print(f"  k={k:2d} (n={5 * k:2d}) {a:24s} vs {b:14s} {d:+7.2f}pp  p={p0:.3g}  "
              f"p_holm={p:.3g}{' *' if p < 0.05 else ''}  ({better}/{n})")


def selftest() -> int:
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(5), 80)
    logits = {}
    for i, m in enumerate(MODS):
        z = rng.normal(size=(400, 5)); z[np.arange(400), y] += 1.0 + i
        z[:, 0] += 2.0                          # participant bias toward class 0
        logits[m] = z
    res = one_draw(logits, y, 4, 0, 1, {"naive_bias": 0.01, "naive_weighted": 0.01})
    # 0-step fit must reproduce centred averaging exactly.
    rng2 = np.random.default_rng(7000 * 1 + 0 + 4)
    calib = draw_stratified(y, 4, rng2); ev = np.setdiff1d(np.arange(400), calib)
    mu = {m: logits[m][calib].mean(0) for m in MODS}
    Pc, Pe = _probs(logits, mu, calib), _probs(logits, mu, ev)
    w, b = fit(Pc, torch.from_numpy(y[calib]).long(), "naive_weighted", 0.01, steps=0)
    same = float((predict(Pe, w, b).argmax(-1).numpy() == y[ev]).mean())
    assert abs(same - res["naive_centred"]) < 1e-12, (same, res["naive_centred"])
    print(f"selftest OK: centred {res['naive_centred']:.3f}  bias {res['naive_bias']:.3f}  "
          f"weighted {res['naive_weighted']:.3f} (0-step fit == centred)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    from paths import RESULTS
    few = RESULTS / "fewshot_calibration.csv"
    if not few.exists():
        sys.exit(f"{few} missing -- run fewshot_calibration.py first")
    with open(few) as f:
        fewshot = list(csv.DictReader(f))
    out = RESULTS / OUT_NAME
    if args.summary_only:
        with open(out) as f:
            summarise(list(csv.DictReader(f)), fewshot)
        return 0
    rows = run([int(x) for x in args.folds.split(",") if x.strip()], args.seeds)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {len(rows):,} rows -> {out}")
    summarise(rows, fewshot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
