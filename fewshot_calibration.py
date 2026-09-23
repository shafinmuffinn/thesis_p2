"""Labelled versus unlabelled calibration: what do labels add?

Context
-------
Chapter 8 shows that normalising a new participant's fusion features with
statistics from n UNLABELLED clips lifts every learned head well above
averaging. This asks whether LABELLING those same clips, and fine-tuning the
head on them, adds anything. Naive averaging has no parameters and cannot use
labels this way; a learned head can.

Pre-specified design (fixed before any result was seen)
-------------------------------------------------------
Heads: the three heads Stage C trained under --subject-norm (cross_attn,
concat_mlp, dropout; dropout is also scored with EEG zeroed).

For each held-out participant, k labelled clips per class are drawn (k in
1, 2, 4, 10, i.e. n = 5k = 5, 10, 20, 50 clips -- the calibration-study grid),
5 seeds. On exactly those clips:
    unlabelled   statistics from the clips normalise the rest; head unchanged
    labelled     same normalisation, then the head is fine-tuned on the clips
                 (all parameters, AdamW, weight decay 1e-4, full batch, train mode)
Both are scored on the participant's remaining 400 - n clips, alongside naive
late fusion on the same clips.

Fine-tuning learning rate and step count come from a 4-point grid
(lr 1e-4 or 1e-3; 10 or 50 steps), chosen per fold and per head on the fold's
three INNER-VALIDATION participants by running this same protocol on them (ties
go to the gentlest setting, lr 1e-4 / 10 steps). No held-out participant
influences the choice.

Tests: labelled vs unlabelled, per head and k, paired Wilcoxon over the 42
participants (seed-averaged), Holm over the 16 tests (4 variants x 4 k).
Built-in check: labelled with 0 steps must equal unlabelled exactly.

    python fewshot_calibration.py
    python fewshot_calibration.py --folds 0 --seeds 2     # quick check
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys

import numpy as np
import torch

K_GRID = [1, 2, 4, 10]
LR_GRID = [1e-4, 1e-3]
STEP_GRID = [10, 50]
WD = 1e-4
VARIANTS = ["concat_mlp", "dropout_full", "cross_attn", "dropout_av"]
OUT_NAME = "fewshot_calibration.csv"
SEL_NAME = "fewshot_selected_hparams.csv"
FIELDS = ["fold", "subject", "k", "seed", "mode", "variant", "acc"]


def draw_stratified(y: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    return np.concatenate([rng.choice(np.flatnonzero(y == c), size=k, replace=False)
                           for c in np.unique(y)])


def finetune(model: torch.nn.Module, x: dict[str, torch.Tensor], y: torch.Tensor,
             lr: float, steps: int) -> torch.nn.Module:
    m = copy.deepcopy(model)
    if steps == 0:
        return m.eval()
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=WD)
    lossf = torch.nn.CrossEntropyLoss()
    m.train()
    for _ in range(steps):
        opt.zero_grad()
        lossf(m(x["audio"], x["vision"], x["eeg"])["logits"], y).backward()
        opt.step()
    return m.eval()


@torch.no_grad()
def head_acc(model, x: dict, y: np.ndarray, zero_eeg: bool = False) -> float:
    e = torch.zeros_like(x["eeg"]) if zero_eeg else x["eeg"]
    return float((model(x["audio"], x["vision"], e)["logits"].argmax(-1).numpy() == y).mean())


def one_participant(heads: dict, feats: dict, logits: dict, y: np.ndarray, k: int,
                    seed: int, sub: int, hp: dict[str, tuple[float, int]] | None,
                    check_zero: bool = False) -> list[tuple[str, str, float]]:
    """(mode, variant, acc) rows for one draw. hp=None -> grid search mode."""
    import cv_pipeline as cv
    from calibration_study import normalise_from

    rng = np.random.default_rng(7000 * sub + 10 * seed + k)
    calib = draw_stratified(y, k, rng)
    evaluate = np.setdiff1d(np.arange(len(y)), calib)
    x_cal = normalise_from(feats, calib, calib)
    x_ev = normalise_from(feats, calib, evaluate)
    y_cal = torch.from_numpy(y[calib]).long()
    y_ev = y[evaluate]
    out = []
    probs = np.stack([cv.softmax_np(logits[m][evaluate]) for m in cv.MODALITIES])
    out.append(("reference", "naive_late", float((probs.mean(0).argmax(-1) == y_ev).mean())))
    for name, model in heads.items():
        variant = "dropout_full" if name == "dropout" else name
        out.append(("unlabelled", variant, head_acc(model, x_ev, y_ev)))
        if name == "dropout":
            out.append(("unlabelled", "dropout_av", head_acc(model, x_ev, y_ev, True)))
        configs = [(lr, st) for lr in LR_GRID for st in STEP_GRID] if hp is None else [hp[name]]
        for lr, st in configs:
            tuned = finetune(model, x_cal, y_cal, lr, st)
            mode = "labelled" if hp is not None else f"grid:{lr}:{st}"
            out.append((mode, variant, head_acc(tuned, x_ev, y_ev)))
            if name == "dropout":
                out.append((mode, "dropout_av", head_acc(tuned, x_ev, y_ev, True)))
        if check_zero:
            same = head_acc(finetune(model, x_cal, y_cal, 1e-3, 0), x_ev, y_ev)
            assert same == head_acc(model, x_ev, y_ev), "0-step fine-tune changed the head"
    return out


def load_participant(fold: int, sub: int):
    import cv_pipeline as cv
    with np.load(cv.FEAT_DIR / f"fold{fold}_sub{sub:02d}.npz") as z:
        return ({m: z[m] for m in cv.MODALITIES},
                {m: z[f"logits_{m}"] for m in cv.MODALITIES}, z["y"])


def select_hparams(fold: int, heads: dict, seeds: int) -> dict[str, tuple[float, int]]:
    """Best (lr, steps) per head, by mean accuracy on the inner-val participants."""
    import cv_pipeline as cv
    from cv_split import train_subjects
    _, val_subs = cv._inner_split(train_subjects(fold))
    scores: dict[tuple, list[float]] = {}
    for sub in val_subs:
        feats, logits, y = load_participant(fold, sub)
        for k in K_GRID:
            for seed in range(seeds):
                for mode, variant, acc in one_participant(heads, feats, logits, y, k, seed,
                                                          sub, None):
                    if mode.startswith("grid:") and variant != "dropout_av":
                        _, lr, st = mode.split(":")
                        scores.setdefault((variant, float(lr), int(st)), []).append(acc)
    chosen = {}
    for name in heads:
        variant = "dropout_full" if name == "dropout" else name
        best = max(((lr, st) for lr in LR_GRID for st in STEP_GRID),
                   key=lambda c: np.mean(scores[(variant, c[0], c[1])]))
        chosen[name] = best
        print(f"    fold {fold} {name:10s} inner-val choice lr={best[0]:g} steps={best[1]}  "
              + "  ".join(f"({lr:g},{st}):{np.mean(scores[(variant, lr, st)]):.4f}"
                          for lr in LR_GRID for st in STEP_GRID), flush=True)
    return chosen


def run(folds: list[int], seeds: int) -> tuple[list[dict], list[dict]]:
    from calibration_study import load_heads
    from cv_split import test_subjects
    torch.manual_seed(0)
    rows, chosen_rows = [], []
    first = True
    for k_fold in folds:
        heads = load_heads(k_fold)
        print(f"fold {k_fold}: selecting fine-tune settings on inner-val participants", flush=True)
        hp = select_hparams(k_fold, heads, seeds=2)
        for name, (lr, st) in hp.items():
            chosen_rows.append(dict(fold=k_fold, head=name, lr=lr, steps=st))
        for sub in test_subjects(k_fold):
            feats, logits, y = load_participant(k_fold, sub)
            for k in K_GRID:
                for seed in range(seeds):
                    for mode, variant, acc in one_participant(heads, feats, logits, y, k,
                                                              seed, sub, hp, check_zero=first):
                        rows.append(dict(fold=k_fold, subject=sub, k=k, seed=seed,
                                         mode=mode, variant=variant, acc=acc))
                    first = False
            print(f"  fold {k_fold} sub{sub:02d} done", flush=True)
    return rows, chosen_rows


def summarise(rows: list[dict]) -> None:
    from scipy.stats import wilcoxon
    cells: dict[tuple, dict[int, list[float]]] = {}
    for r in rows:
        cells.setdefault((int(r["k"]), r["mode"], r["variant"]), {}) \
             .setdefault(int(r["subject"]), []).append(float(r["acc"]))
    per = {key: {s: float(np.mean(v)) for s, v in d.items()} for key, d in cells.items()}
    mean = lambda key: np.mean(list(per[key].values()))

    print(f"\n{'k':>3s}{'n':>4s}{'naive':>8s}" + "".join(f"{v + ' U':>16s}{v + ' L':>16s}"
                                                        for v in VARIANTS))
    for k in K_GRID:
        line = f"{k:3d}{5 * k:4d}{mean((k, 'reference', 'naive_late')):8.4f}"
        for v in VARIANTS:
            line += f"{mean((k, 'unlabelled', v)):16.4f}{mean((k, 'labelled', v)):16.4f}"
        print(line)

    tests, labels = [], []
    for k in K_GRID:
        for v in VARIANTS:
            L, U = per[(k, "labelled", v)], per[(k, "unlabelled", v)]
            subs = sorted(set(L) & set(U))
            a, b = np.array([L[s] for s in subs]), np.array([U[s] for s in subs])
            tests.append(1.0 if np.allclose(a, b) else float(wilcoxon(a, b).pvalue))
            labels.append((k, v, (a - b).mean() * 100, int((a > b).sum()), len(subs)))
    order = np.argsort(tests); m = len(tests); adj = [0.0] * m; r = 0.0
    for rank, i in enumerate(order):
        r = max(r, min(1.0, (m - rank) * tests[i])); adj[i] = r
    print(f"\nLabelled minus unlabelled, paired Wilcoxon, Holm over {m} tests")
    for (k, v, d, better, n), p0, p in zip(labels, tests, adj):
        print(f"  k={k:2d} (n={5 * k:2d}) {v:13s} {d:+7.2f}pp  p={p0:.3g}  p_holm={p:.3g}"
              f"{' *' if p < 0.05 else ''}  ({better}/{n})")

    # EXPLORATORY (added after the results were seen; not part of the
    # pre-specified family): with labels, does the attention model trained with
    # modality dropout overtake concat-MLP? Holm over the 4 k values.
    tests, labels = [], []
    for k in K_GRID:
        A, B = per[(k, "labelled", "dropout_full")], per[(k, "labelled", "concat_mlp")]
        subs = sorted(set(A) & set(B))
        a, b = np.array([A[s] for s in subs]), np.array([B[s] for s in subs])
        tests.append(float(wilcoxon(a, b).pvalue))
        labels.append((k, (a - b).mean() * 100, int((a > b).sum()), len(subs)))
    order = np.argsort(tests); m = len(tests); adj = [0.0] * m; r = 0.0
    for rank, i in enumerate(order):
        r = max(r, min(1.0, (m - rank) * tests[i])); adj[i] = r
    print("\nEXPLORATORY: labelled dropout_full minus labelled concat_mlp (Holm over 4)")
    for (k, d, better, n), p0, p in zip(labels, tests, adj):
        print(f"  k={k:2d} (n={5 * k:2d}) {d:+7.2f}pp  p={p0:.3g}  p_holm={p:.3g}"
              f"{' *' if p < 0.05 else ''}  ({better}/{n})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    from paths import RESULTS
    out = RESULTS / OUT_NAME
    if args.summary_only:
        with open(out) as f:
            summarise(list(csv.DictReader(f)))
        return 0
    rows, chosen = run([int(x) for x in args.folds.split(",") if x.strip()], args.seeds)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    with open(RESULTS / SEL_NAME, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fold", "head", "lr", "steps"])
        w.writeheader(); w.writerows(chosen)
    print(f"\nwrote {len(rows):,} rows -> {out}")
    summarise(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
