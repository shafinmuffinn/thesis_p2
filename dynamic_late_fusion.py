"""Parameter-free dynamic late fusion, evaluated cross-subject.

Motivation
----------
Under subject-independent evaluation the learned fusion heads lose to naive
softmax averaging (Chapter 7). The proposed mechanism is that a learned head
fits modality reliability to the TRAINING participants -- on EAV that means
learning to trust vision, which is precisely the modality that does not
transfer. Naive averaging wins by having no fitted quantity to be wrong.

That diagnosis suggests a fusion rule that is dynamic per trial but fits
nothing: weight each modality by its own confidence on THIS trial, rather than
by a preference learned from the training population. Such a rule adapts like
attention, but is subject-agnostic by construction.

Every scheme here has zero trainable parameters and runs on the per-modality
logits cached by cv_pipeline Stage C, so the whole study is minutes on CPU.

    python dynamic_late_fusion.py
    python dynamic_late_fusion.py --folds 0,1
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from cv_pipeline import FEAT_DIR, LOGITS_DIR, MODALITIES, N_CLASSES, _inner_split
from cv_split import N_FOLDS, SUBJECTS, test_subjects, train_subjects

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None


# ---------------------------------------------------------------------------
# Confidence measures -- each maps a (N, K) probability matrix to (N, 1) weights
# ---------------------------------------------------------------------------

def softmax(z: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = z / T
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def w_uniform(p: np.ndarray) -> np.ndarray:
    """Naive late fusion: every modality counts equally, always."""
    return np.ones((len(p), 1))


def w_maxprob(p: np.ndarray) -> np.ndarray:
    """Top-1 probability. Simplest notion of 'how sure is this modality'."""
    return p.max(axis=-1, keepdims=True)


def w_neg_entropy(p: np.ndarray) -> np.ndarray:
    """1 - normalised Shannon entropy; 0 when uniform, 1 when one-hot.

    Uses the whole distribution rather than only its peak, so a modality that
    is torn between two classes is penalised more than max-prob alone implies.
    """
    eps = 1e-12
    H = -(p * np.log(p + eps)).sum(axis=-1, keepdims=True)
    return 1.0 - H / np.log(N_CLASSES)


def w_margin(p: np.ndarray) -> np.ndarray:
    """Gap between the top two classes -- decisiveness rather than peak height."""
    s = np.sort(p, axis=-1)
    return (s[:, -1] - s[:, -2])[:, None]


WEIGHTS = {
    "naive_late": w_uniform,
    "dyn_maxprob": w_maxprob,
    "dyn_negentropy": w_neg_entropy,
    "dyn_margin": w_margin,
}


def fuse(probs: dict[str, np.ndarray], weight_fn, power: float = 1.0) -> np.ndarray:
    """Confidence-weighted average of per-modality probability vectors.

    `power` sharpens or softens the weighting: 0 collapses every scheme to
    naive averaging, larger values concentrate mass on the confident modality.
    """
    ws = np.concatenate([weight_fn(probs[m]) ** power for m in MODALITIES], axis=1)
    ws = ws / np.clip(ws.sum(axis=1, keepdims=True), 1e-12, None)
    stacked = np.stack([probs[m] for m in MODALITIES], axis=0)      # (3, N, K)
    fused = (stacked * ws.T[:, :, None]).sum(axis=0)                # (N, K)
    return fused.argmax(axis=-1)


def fuse_oracle_free_best(probs: dict[str, np.ndarray]) -> np.ndarray:
    """Hand the trial to whichever modality is most confident. No averaging.

    Included as the degenerate extreme of confidence weighting: it shows how
    much of any gain comes from adaptivity versus from still averaging.
    """
    conf = np.concatenate([w_maxprob(probs[m]) for m in MODALITIES], axis=1)
    pick = conf.argmax(axis=1)
    stacked = np.stack([probs[m] for m in MODALITIES], axis=0)
    return stacked[pick, np.arange(len(pick))].argmax(axis=-1)


# ---------------------------------------------------------------------------

def load_folds(folds: list[int]) -> dict:
    acc = {f"logits_{m}": [] for m in MODALITIES}
    acc.update({"y": [], "subject": []})
    missing = []
    for k in folds:
        path = LOGITS_DIR / f"fold{k}.npz"
        if not path.exists():
            missing.append(k)
            continue
        with np.load(path) as z:
            for key in acc:
                acc[key].append(z[key])
    if missing:
        print(f"WARNING: no logits for fold(s) {missing} -- run --stages C first.")
    if not acc["y"]:
        raise SystemExit("No fold logits found; nothing to do.")
    return {k: np.concatenate(v, axis=0) for k, v in acc.items()}


def calibration_report(probs: dict[str, np.ndarray], y: np.ndarray,
                       n_bins: int = 5) -> dict[str, float]:
    """Does each modality's confidence track its correctness?

    This is the precondition for the whole study. Weighting by confidence can
    only help if a confident modality is more often right than an unconfident
    one; an encoder that is confidently wrong will be up-weighted precisely
    when it should be ignored. Reports accuracy by confidence quintile and the
    accuracy gap between the top and bottom quintile.
    """
    print(f"{'modality':12s}" + "".join(f"{'Q'+str(i+1):>9s}" for i in range(n_bins))
          + f"{'spread':>10s}")
    gaps = {}
    for m in MODALITIES:
        conf = probs[m].max(axis=-1)
        correct = probs[m].argmax(axis=-1) == y
        edges = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
        edges[-1] += 1e-9
        accs = []
        for i in range(n_bins):
            sel = (conf >= edges[i]) & (conf < edges[i + 1])
            accs.append(float(correct[sel].mean()) if sel.any() else float("nan"))
        gaps[m] = accs[-1] - accs[0]
        print(f"{m:12s}" + "".join(f"{a:9.3f}" for a in accs) + f"{gaps[m]:+10.3f}")
    print("\nSpread = accuracy in the most-confident fifth minus the least-confident "
          "fifth.\nA clearly positive spread means confidence is informative and "
          "weighting by it can help;\nnear zero or negative means the modality is "
          "confidently wrong and should not be up-weighted.")
    return gaps


def load_inner_val(fold: int) -> dict[str, np.ndarray]:
    """Per-modality logits and labels for a fold's inner-validation subjects.

    These are the three participants withheld from that fold's encoder training,
    so their logits are produced under the same conditions as the test fold's --
    which is what makes them a legitimate set on which to fit a calibration
    parameter. The test fold is never touched.
    """
    subs, out = _inner_split(train_subjects(fold))[1], {}
    acc = {f"logits_{m}": [] for m in MODALITIES}
    acc["y"] = []
    for sub in subs:
        with np.load(FEAT_DIR / f"fold{fold}_sub{sub:02d}.npz") as z:
            for m in MODALITIES:
                acc[f"logits_{m}"].append(z[f"logits_{m}"])
            acc["y"].append(z["y"])
    return {k: np.concatenate(v, axis=0) for k, v in acc.items()}


def fit_temperature(logits: np.ndarray, y: np.ndarray,
                    grid: np.ndarray | None = None) -> float:
    """Temperature minimising NLL on held-out data (Guo et al., 2017).

    One scalar per modality. T > 1 softens an overconfident encoder; T < 1
    sharpens an underconfident one. Fitted by grid search on negative
    log-likelihood, which is convex enough in T that a grid is sufficient and
    avoids an optimiser dependency.

    This is the minimal learned correction available: three parameters for the
    whole system, fitted on subjects the encoders never saw. A full fusion head
    has millions and fits them to the training population's modality
    reliability -- the quantity that does not transfer.
    """
    if grid is None:
        grid = np.concatenate([np.linspace(0.25, 1.0, 16), np.linspace(1.05, 6.0, 100)])
    best_T, best_nll = 1.0, np.inf
    for T in grid:
        p = softmax(logits, T=float(T))
        nll = -np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None)).mean()
        if nll < best_nll:
            best_T, best_nll = float(T), float(nll)
    return best_T


def per_subject(preds: np.ndarray, y: np.ndarray, subj: np.ndarray) -> np.ndarray:
    """One accuracy per participant, ordered by participant id."""
    return np.array([float((preds[subj == s] == y[subj == s]).mean())
                     for s in sorted(set(subj.tolist()))])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", default=",".join(str(k) for k in range(N_FOLDS)))
    ap.add_argument("--powers", default="1,2,4",
                    help="sharpening exponents to sweep for each scheme")
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    powers = [float(x) for x in args.powers.split(",") if x.strip() != ""]

    d = load_folds(folds)
    y, subj = d["y"], d["subject"]
    probs = {m: softmax(d[f"logits_{m}"]) for m in MODALITIES}
    subs = sorted(set(subj.tolist()))
    print(f"{len(subs)} participants, {len(y):,} trials, folds {folds}\n")

    print("=" * 52)
    print("CALIBRATION: is confidence informative?")
    print("=" * 52)
    calibration_report(probs, y)
    print()

    results: dict[str, np.ndarray] = {}

    for m in MODALITIES:
        results[f"{m}_only"] = per_subject(probs[m].argmax(axis=-1), y, subj)

    for name, fn in WEIGHTS.items():
        if name == "naive_late":
            results[name] = per_subject(fuse(probs, fn), y, subj)
            continue
        for pw in powers:
            label = f"{name}^{pw:g}"
            results[label] = per_subject(fuse(probs, fn, power=pw), y, subj)

    results["dyn_winner_takes_all"] = per_subject(fuse_oracle_free_best(probs), y, subj)

    # --- per-modality temperature scaling, fitted on inner-validation subjects ---
    #
    # Mean-softmax averaging is already implicitly confidence-weighted: a peaked
    # distribution dominates the mean, a flat one cancels. What it cannot do is
    # correct a modality whose confidence is systematically wrong. If vision is
    # overconfident on unseen participants, averaging over-trusts it on exactly
    # the trials where it should be discounted. Temperature scaling fixes the
    # scale of each modality's confidence using three scalars, fitted on
    # participants the encoders never saw.
    try:
        cal = {m: np.empty_like(probs[m]) for m in MODALITIES}
        temps: dict[int, dict[str, float]] = {}
        for k in folds:
            iv = load_inner_val(k)
            temps[k] = {m: fit_temperature(iv[f"logits_{m}"], iv["y"])
                        for m in MODALITIES}
            mask = np.isin(subj, test_subjects(k))
            for m in MODALITIES:
                cal[m][mask] = softmax(d[f"logits_{m}"][mask], T=temps[k][m])

        print("--- fitted temperatures (per fold) ---")
        for k in folds:
            print(f"  fold {k}: " + "  ".join(
                f"{m}={temps[k][m]:.2f}" for m in MODALITIES))
        mean_T = {m: float(np.mean([temps[k][m] for k in folds])) for m in MODALITIES}
        print("  mean : " + "  ".join(f"{m}={mean_T[m]:.2f}" for m in MODALITIES))
        print("  T>1 means the encoder was overconfident on unseen participants;")
        print("  the larger the temperature, the more averaging was over-trusting it.\n")

        results["calibrated_late"] = per_subject(fuse(cal, w_uniform), y, subj)
        for pw in powers:
            results[f"calibrated_negentropy^{pw:g}"] = per_subject(
                fuse(cal, w_neg_entropy, power=pw), y, subj)
    except FileNotFoundError as e:
        print(f"\nNOTE: temperature scaling skipped (needs Stage B caches) -- {e}")

    print(f"{'scheme':24s}{'mean':>8s}{'std':>8s}")
    for name, a in sorted(results.items(), key=lambda kv: -kv[1].mean()):
        print(f"{name:24s}{a.mean():8.4f}{a.std(ddof=1):8.4f}")

    base = results["naive_late"]
    print(f"\n--- paired against naive_late (n={len(base)}) ---")
    for name, a in sorted(results.items(), key=lambda kv: -kv[1].mean()):
        if name == "naive_late" or name.endswith("_only"):
            continue
        diff = (a.mean() - base.mean()) * 100
        if wilcoxon is not None and not np.allclose(a, base):
            p = wilcoxon(a, base).pvalue
            star = " *" if p < 0.05 else ""
            print(f"  {name:24s}{diff:+7.2f}pp  p={p:.4f}{star}  "
                  f"(better on {int((a > base).sum())}/{len(base)})")
        else:
            print(f"  {name:24s}{diff:+7.2f}pp  (identical to baseline)")

    print("\nA scheme that beats naive_late here is dynamic without being fitted to "
          "the training participants -- the property the learned heads lack.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
