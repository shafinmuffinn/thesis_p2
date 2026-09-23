"""Leak-free re-evaluation of the P2 within-subject fusion results.

The problem
-----------
All three P2 fusion scripts (day5_pipeline, concat_mlp_baseline,
day7_modality_dropout) evaluated on the TEST trials after every epoch and kept
the best epoch. Every reported learned-fusion number -- 80.2 / 81.7 / 84.7 /
81.5% -- is therefore the maximum of 80 test measurements. Naive late fusion
(77.5%) has no epochs to select, so the P2 comparison of learned fusion against
averaging is biased in favour of learned fusion.

What this does
--------------
Reproduces the P2 recipe exactly (raw features, AdamW 1e-3, weight decay 1e-4,
cosine schedule, batch 32, 80 epochs; softhard dropout p=0.5 for the dropout
head) and changes only how the epoch is chosen. Each head is trained twice per
participant:

  Run A -- fit on 80% of the 280 training trials (stratified), holding out the
           other 20% as validation. ONE training run, read out three ways:
             test_selected   best test accuracy over epochs   (the P2 protocol)
             val_selected    test accuracy at the best-VALIDATION epoch
             final_fit       test accuracy at the final epoch
           Because all three readouts come from the same weights trajectory,
           test_selected minus val_selected is the selection inflation itself,
           not a difference between two separate runs.

  Run B -- fit on all 280 training trials, no selection at all:
             final_full      test accuracy at the final epoch

val_selected and final_full are both leak-free. Naive late fusion is recomputed
from the cached per-modality logits for paired comparison.

    python p2_leakfree.py
    python p2_leakfree.py --subjects 1,2,3      # quick check
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from concat_mlp_baseline import ConcatMLP
from day7_modality_dropout import softhard_dropout
from fusion.trimodal_attention import TrimodalAttentionFusion
from paths import CHECKPOINTS, RESULTS

FEAT_DIR = CHECKPOINTS / "day5_features"
LOGIT_DIR = CHECKPOINTS / "day5_per_modality_logits"
OUT_CSV = RESULTS / "p2_leakfree.csv"
FIELDS = ["subject", "variant", "readout", "acc"]
MODALITIES = ["audio", "vision", "eeg"]
N_CLASSES = 5

# P2 recipe, unchanged.
EPOCHS, BATCH, LR, WD, DROP_P = 80, 32, 1e-3, 1e-4, 0.5
VAL_FRACTION = 0.2
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ConcatDict(nn.Module):
    """ConcatMLP returns a tensor; give it the dict contract of the others."""

    def __init__(self, **dims):
        super().__init__()
        self.net = ConcatMLP(**dims)

    def forward(self, a, v, e):
        return {"logits": self.net(a, v, e)}


def build(name: str, dims: dict) -> nn.Module:
    if name == "concat_mlp":
        return ConcatDict(**dims)
    return TrimodalAttentionFusion(**dims)          # cross_attn and dropout


def stratified_val(y: np.ndarray, frac: float, seed: int) -> np.ndarray:
    """Boolean mask of validation rows, stratified by class."""
    rng = np.random.default_rng(seed)
    val = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        val[rng.choice(idx, size=max(1, int(round(frac * len(idx)))), replace=False)] = True
    return val


@torch.no_grad()
def accuracy(model, x: dict, y: torch.Tensor, zero_eeg: bool = False) -> float:
    model.eval()
    e = torch.zeros_like(x["eeg"]) if zero_eeg else x["eeg"]
    pred = model(x["audio"], x["vision"], e)["logits"].argmax(-1)
    return float((pred == y).float().mean())


def train(name: str, dims: dict, fit_x: dict, fit_y: torch.Tensor,
          monitors: dict[str, tuple[dict, torch.Tensor]]) -> dict[str, list[float]]:
    """Train with the P2 recipe; record per-epoch accuracy on each monitor set.

    Monitors are only observed, never used to steer training.
    """
    torch.manual_seed(SEED)
    model = build(name, dims).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss()
    n = fit_y.numel()
    curves: dict[str, list[float]] = {k: [] for k in monitors}
    if name == "dropout":
        curves["test_av"] = []

    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            a, v, e = fit_x["audio"][idx], fit_x["vision"][idx], fit_x["eeg"][idx]
            if name == "dropout":
                a, v, e = softhard_dropout(a, v, e, p=DROP_P)
            opt.zero_grad()
            crit(model(a, v, e)["logits"], fit_y[idx]).backward()
            opt.step()
        sched.step()
        for key, (x, y) in monitors.items():
            curves[key].append(accuracy(model, x, y))
        if name == "dropout" and "test" in monitors:
            curves["test_av"].append(accuracy(model, *monitors["test"], zero_eeg=True))
    return curves


def naive_accuracy(sub: int, test_y: np.ndarray) -> float:
    probs = []
    for m in MODALITIES:
        with np.load(LOGIT_DIR / f"sub{sub:02d}_{m}.npz") as z:
            if not np.array_equal(np.asarray(z["labels"]).ravel(), test_y):
                raise ValueError(f"sub{sub:02d} {m}: logit labels do not match "
                                 f"feature test labels -- trial order differs")
            lg = z["logits"]
        e = np.exp(lg - lg.max(axis=-1, keepdims=True))
        probs.append(e / e.sum(axis=-1, keepdims=True))
    return float((np.mean(probs, axis=0).argmax(-1) == test_y).mean())


def run_subject(sub: int) -> list[dict]:
    with np.load(FEAT_DIR / f"sub{sub:02d}.npz") as z:
        tr = {m: z[f"train_{m}"].astype(np.float32) for m in MODALITIES}
        te = {m: z[f"test_{m}"].astype(np.float32) for m in MODALITIES}
        tr_y, te_y = z["train_y"].astype(np.int64), z["test_y"].astype(np.int64)

    t = lambda a: torch.from_numpy(a).to(DEVICE)
    dims = {"audio_dim": tr["audio"].shape[1], "vision_dim": tr["vision"].shape[1],
            "eeg_dim": tr["eeg"].shape[1]}
    val = stratified_val(tr_y, VAL_FRACTION, SEED + sub)
    fit_x = {m: t(tr[m][~val]) for m in MODALITIES}
    val_x = {m: t(tr[m][val]) for m in MODALITIES}
    all_x = {m: t(tr[m]) for m in MODALITIES}
    test_x = {m: t(te[m]) for m in MODALITIES}
    fit_y, val_y, all_y, test_y = t(tr_y[~val]), t(tr_y[val]), t(tr_y), t(te_y)

    rows = [dict(subject=sub, variant="naive_late", readout="none",
                 acc=naive_accuracy(sub, te_y))]
    for name in ("cross_attn", "concat_mlp", "dropout"):
        # Run A: one trajectory, three readouts.
        c = train(name, dims, fit_x, fit_y, {"val": (val_x, val_y), "test": (test_x, test_y)})
        best_val_ep = int(np.argmax(c["val"]))      # first epoch reaching the max
        variant = "dropout_full" if name == "dropout" else name
        rows += [
            dict(subject=sub, variant=variant, readout="test_selected", acc=max(c["test"])),
            dict(subject=sub, variant=variant, readout="val_selected",
                 acc=c["test"][best_val_ep]),
            dict(subject=sub, variant=variant, readout="final_fit", acc=c["test"][-1]),
        ]
        if name == "dropout":
            ts_ep = int(np.argmax(c["test"]))
            rows += [
                dict(subject=sub, variant="dropout_av", readout="test_selected",
                     acc=c["test_av"][ts_ep]),
                dict(subject=sub, variant="dropout_av", readout="val_selected",
                     acc=c["test_av"][best_val_ep]),
                dict(subject=sub, variant="dropout_av", readout="final_fit",
                     acc=c["test_av"][-1]),
            ]
        # Run B: all 280 training trials, no selection.
        cb = train(name, dims, all_x, all_y, {"test": (test_x, test_y)})
        rows.append(dict(subject=sub, variant=variant, readout="final_full", acc=cb["test"][-1]))
        if name == "dropout":
            rows.append(dict(subject=sub, variant="dropout_av", readout="final_full",
                             acc=cb["test_av"][-1]))
    return rows


def summarise(rows: list[dict]) -> None:
    from scipy.stats import wilcoxon

    def vec(variant, readout):
        d = {int(r["subject"]): float(r["acc"]) for r in rows
             if r["variant"] == variant and r["readout"] == readout}
        return d

    naive = vec("naive_late", "none")
    readouts = ["test_selected", "val_selected", "final_fit", "final_full"]
    print(f"\nWithin-subject fusion, {len(naive)} participants "
          f"(naive_late = {np.mean(list(naive.values())):.4f})")
    print(f"{'variant':14s}" + "".join(f"{r:>15s}" for r in readouts) + f"{'inflation':>11s}")
    for v in ("cross_attn", "concat_mlp", "dropout_full", "dropout_av"):
        means = []
        for r in readouts:
            d = vec(v, r)
            means.append(np.mean(list(d.values())) if d else np.nan)
        infl = (means[0] - means[1]) * 100
        print(f"{v:14s}" + "".join(f"{m:15.4f}" for m in means) + f"{infl:+10.2f}pp")

    print("\nLeak-free heads vs naive_late (paired Wilcoxon; Holm over the rows below)")
    tests, labels = [], []
    for v in ("cross_attn", "concat_mlp", "dropout_full", "dropout_av"):
        for r in ("val_selected", "final_full"):
            d = vec(v, r)
            subs = sorted(set(d) & set(naive))
            a = np.array([d[s] for s in subs]); b = np.array([naive[s] for s in subs])
            if len(subs) < 6 or np.allclose(a, b):
                continue
            tests.append(float(wilcoxon(a, b).pvalue))
            labels.append((v, r, (a.mean() - b.mean()) * 100, int((a > b).sum()), len(subs)))
    order = np.argsort(tests); m = len(tests); adj = [0.0] * m; run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * tests[i])); adj[i] = run
    for (v, r, dlt, better, n), p in zip(labels, adj):
        print(f"  {v:13s} {r:13s} {dlt:+7.2f}pp  p_holm={p:.4f}{' *' if p < 0.05 else ''}"
              f"  (better on {better}/{n})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subjects", default=",".join(str(s) for s in range(1, 43)))
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()

    if args.summary_only:
        with open(OUT_CSV) as f:
            summarise(list(csv.DictReader(f)))
        return 0

    rows: list[dict] = []
    subs = [int(s) for s in args.subjects.split(",") if s.strip()]
    for i, sub in enumerate(subs, 1):
        rows += run_subject(sub)
        print(f"  [{i}/{len(subs)}] sub{sub:02d} done", flush=True)
        # Written every subject so a disconnect loses at most one participant.
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {OUT_CSV}  (device={DEVICE})")
    summarise(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
