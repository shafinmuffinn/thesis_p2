"""Is the suppression matrix anything more than EEG's confusion structure?

The P2 suppression matrix counts trials where audio and vision agree on an
emotion (external consensus), EEG predicts a different one, and EEG is
confident (top-1 probability >= 0.5). It is read as a map of emotion
suppression. A defence panel's natural objection: in a cue-based corpus the
AV consensus is usually the cued label, so a "suppression event" is mostly
"EEG misclassified a trial that AV got right", and the matrix is just EEG's
confusion matrix seen through a filter.

Three tests, on the same per-modality logits the P2 matrix was built from:

(a) Consensus correctness. What fraction of suppression events have an AV
    consensus equal to the cued label? Near 100% means the rows of the matrix
    are the TRUE classes, and every event is an EEG error on that class.

(b) Similarity to EEG's own error structure. Correlate the matrix's
    off-diagonal cells with EEG's confident-error confusion matrix computed on
    (i) all trials and (ii) trials where AV is correct. High correlation means
    the matrix reproduces what EEG gets wrong regardless of what AV does.

(c) Permutation null -- the decisive test. Within each participant and each
    cued class, shuffle EEG's predictions across trials while holding AV fixed.
    This keeps every classifier's per-class accuracy and confusion pattern
    exactly, and destroys only the trial-level pairing between AV and EEG. If
    the observed matrix lies inside the null distribution, "suppression events"
    carry no trial-level coupling beyond independent per-class errors -- the
    matrix is fully explained by each classifier's error rates.

    python suppression_audit.py                 # within-subject (P2 matrix)
    python suppression_audit.py --source cross  # cross-subject cv5 logits
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from paths import CHECKPOINTS, IDX_TO_EMOTION

MODALITIES = ["audio", "vision", "eeg"]
K = 5
EEG_CONF = 0.5
NAMES = [IDX_TO_EMOTION[i] for i in range(K)]


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def load_within() -> list[dict]:
    """Per-participant test logits from the P2 within-subject rollout."""
    d = CHECKPOINTS / "day5_per_modality_logits"
    out = []
    for sub in range(1, 43):
        rec = {"subject": sub}
        for m in MODALITIES:
            with np.load(d / f"sub{sub:02d}_{m}.npz") as z:
                rec[m] = z["logits"]
                y = np.asarray(z["labels"]).ravel().astype(int)
                if "y" in rec and not np.array_equal(rec["y"], y):
                    raise ValueError(f"sub{sub:02d}: label order differs across modalities")
                rec["y"] = y
        out.append(rec)
    return out


def load_cross() -> list[dict]:
    """Held-out participants' logits from the cross-subject folds."""
    from cv_pipeline import FEAT_DIR
    from cv_split import N_FOLDS, test_subjects
    out = []
    for k in range(N_FOLDS):
        for sub in test_subjects(k):
            with np.load(FEAT_DIR / f"fold{k}_sub{sub:02d}.npz") as z:
                out.append({"subject": sub, "y": z["y"].astype(int),
                            **{m: z[f"logits_{m}"] for m in MODALITIES}})
    return out


def events(recs: list[dict], eeg_pred: list[np.ndarray] | None = None,
           eeg_conf: list[np.ndarray] | None = None):
    """Suppression matrix (rows AV consensus, cols EEG), event count, AV-correct count.

    `eeg_pred` / `eeg_conf` override EEG's own prediction and confidence per
    participant -- used by the permutation null, where they are shuffled.
    """
    M = np.zeros((K, K), dtype=int)
    av_correct = 0
    total = 0
    for i, r in enumerate(recs):
        a = r["audio"].argmax(-1)
        v = r["vision"].argmax(-1)
        if eeg_pred is None:
            pe = softmax(r["eeg"])
            e, conf = pe.argmax(-1), pe.max(-1)
        else:
            e, conf = eeg_pred[i], eeg_conf[i]
        sel = (a == v) & (e != a) & (conf >= EEG_CONF)
        for row, col in zip(a[sel], e[sel]):
            M[row, col] += 1
        av_correct += int((a[sel] == r["y"][sel]).sum())
        total += int(sel.sum())
    return M, total, av_correct


def eeg_confusion(recs: list[dict], only_av_correct: bool) -> np.ndarray:
    """EEG confident-error counts, rows = cued class, cols = EEG prediction."""
    C = np.zeros((K, K), dtype=int)
    for r in recs:
        pe = softmax(r["eeg"])
        e, conf, y = pe.argmax(-1), pe.max(-1), r["y"]
        keep = (e != y) & (conf >= EEG_CONF)
        if only_av_correct:
            a, v = r["audio"].argmax(-1), r["vision"].argmax(-1)
            keep &= (a == v) & (a == y)
        for row, col in zip(y[keep], e[keep]):
            C[row, col] += 1
    return C


def offdiag(M: np.ndarray) -> np.ndarray:
    return M[~np.eye(K, dtype=bool)].astype(float)


def permutation_null(recs: list[dict], n_perm: int, seed: int) -> np.ndarray:
    """Null matrices from shuffling EEG (prediction, confidence) within class."""
    rng = np.random.default_rng(seed)
    base = []
    for r in recs:
        pe = softmax(r["eeg"])
        base.append((pe.argmax(-1), pe.max(-1)))
    null = np.zeros((n_perm, K, K), dtype=int)
    for p in range(n_perm):
        preds, confs = [], []
        for r, (e, c) in zip(recs, base):
            e2, c2 = e.copy(), c.copy()
            for cls in range(K):
                idx = np.flatnonzero(r["y"] == cls)
                perm = rng.permutation(idx)
                e2[idx], c2[idx] = e[perm], c[perm]
            preds.append(e2); confs.append(c2)
        null[p] = events(recs, eeg_pred=preds, eeg_conf=confs)[0]
    return null


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i])); adj[i] = run
    return adj


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["within", "cross"], default="within")
    ap.add_argument("--perms", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    recs = load_within() if args.source == "within" else load_cross()
    n_trials = sum(len(r["y"]) for r in recs)
    M, total, av_ok = events(recs)
    print(f"source={args.source}: {len(recs)} participants, {n_trials:,} trials")
    print(f"suppression events: {total}  ({100 * total / n_trials:.1f}% base rate)")

    print("\nObserved suppression matrix (rows = AV consensus, cols = EEG)")
    print(" " * 11 + "".join(f"{n[:9]:>10s}" for n in NAMES))
    for i in range(K):
        print(f"{NAMES[i][:10]:>10s} " + "".join(f"{M[i, j]:10d}" for j in range(K)))

    # (a)
    print(f"\n(a) AV consensus equals the cued label on {av_ok}/{total} events "
          f"({100 * av_ok / max(total, 1):.1f}%).")

    # (b)
    for label, only in (("all trials", False), ("AV-correct trials", True)):
        C = eeg_confusion(recs, only_av_correct=only)
        r = np.corrcoef(offdiag(M), offdiag(C))[0, 1]
        print(f"(b) correlation with EEG confident-error matrix on {label}: r = {r:.3f}")

    # (c)
    print(f"\n(c) permutation null: {args.perms} within-class shuffles of EEG ...")
    null = permutation_null(recs, args.perms, args.seed)
    tot_null = null.sum(axis=(1, 2))
    p_tot = (np.sum(np.abs(tot_null - tot_null.mean()) >= abs(total - tot_null.mean())) + 1) \
        / (args.perms + 1)
    print(f"    total events: observed {total}, null {tot_null.mean():.1f} "
          f"+/- {tot_null.std():.1f}, two-sided p = {p_tot:.4f}")

    cells, ps, zs = [], [], []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            nd = null[:, i, j]
            mu, sd = nd.mean(), nd.std() + 1e-9
            p = (np.sum(np.abs(nd - mu) >= abs(M[i, j] - mu)) + 1) / (args.perms + 1)
            cells.append((i, j, M[i, j], mu)); ps.append(p); zs.append((M[i, j] - mu) / sd)
    adj = holm(np.array(ps))
    print("\n    cells differing from the null (Holm over 20 cells):")
    shown = 0
    for (i, j, obs, mu), z, p in sorted(zip(cells, zs, adj), key=lambda t: t[2]):
        if p < 0.05:
            print(f"      {NAMES[i]:>9s} -> {NAMES[j]:<9s} observed {obs:4d}  "
                  f"null {mu:6.1f}  z = {z:+5.2f}  p_holm = {p:.4f}")
            shown += 1
    if not shown:
        print("      none -- every cell is consistent with independent per-class errors")

    print("\nReading: (a) near 100% and (b) near 1 mean the rows are the true classes "
          "and the matrix mirrors EEG's own errors. (c) is decisive: a matrix inside "
          "the null carries no trial-level AV-EEG coupling, so 'suppression' is not "
          "supported beyond the classifiers' independent error rates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
