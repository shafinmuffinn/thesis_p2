"""Day 3 — Naive late fusion of per-modality logits saved on Day 2.

Loads the .npz archives produced by Day 2's driver, sanity-checks that the
three modalities' test labels are trial-aligned, and computes four fusion
schemes:

    1. Per-modality accuracy (reconstructed from logits — must match Day 2 CSV)
    2. Mean of softmax probabilities (the canonical "naive late fusion")
    3. Sum of log-softmax (= product of probabilities; independent-evidence rule)
    4. Hard majority vote (each modality casts one vote for its top class)

Results are written to results/day3_late_fusion.csv and printed as a table.

Runs locally on the Mac (no GPU needed). If the .npz files live on Google
Drive, sync them down to data/EAV/checkpoints/day2_logits/ first.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from paths import CHECKPOINTS, RESULTS

LOGIT_DIR = CHECKPOINTS / "day2_logits"
OUT_CSV = RESULTS / "day3_late_fusion.csv"
PER_TRIAL_CSV = RESULTS / "day3_per_trial_coherence.csv"

MODALITIES = ("audio", "vision", "eeg")


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along `axis`."""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    return x - np.log(np.sum(np.exp(x), axis=axis, keepdims=True))


def accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean())


def symmetric_kl(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Per-row symmetric KL divergence between two probability matrices of
    shape (N, 5). Returns an (N,) array of nonneg scalars."""
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return 0.5 * (np.sum(p * np.log(p / q), axis=-1)
                  + np.sum(q * np.log(q / p), axis=-1))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def discover_subjects() -> list[int]:
    """Return subject IDs that have all three modalities saved."""
    if not LOGIT_DIR.exists():
        sys.exit(f"❌ Logit directory not found: {LOGIT_DIR}\n"
                 f"   Sync `MyDrive/Thesis_EAV/checkpoints/day2_logits/` "
                 f"down to this path, then re-run.")
    found: dict[int, set[str]] = {}
    for p in LOGIT_DIR.glob("sub*_*.npz"):
        try:
            stem = p.stem                               # e.g. "sub01_audio"
            sub = int(stem[3:5])
            mod = stem.split("_", 1)[1]
        except (ValueError, IndexError):
            continue
        found.setdefault(sub, set()).add(mod)
    complete = sorted(s for s, mods in found.items() if set(MODALITIES) <= mods)
    if not complete:
        sys.exit(f"❌ No subjects with all three modalities under {LOGIT_DIR}")
    return complete


def load_subject(sub: int) -> dict[str, dict[str, np.ndarray]]:
    """Load logits + labels for one subject across three modalities."""
    out: dict[str, dict[str, np.ndarray]] = {}
    for mod in MODALITIES:
        npz = np.load(LOGIT_DIR / f"sub{sub:02d}_{mod}.npz")
        out[mod] = {"logits": npz["logits"], "labels": npz["labels"]}
    return out


# ---------------------------------------------------------------------------
# Fusion schemes
# ---------------------------------------------------------------------------

def fuse_mean_softmax(logits: dict[str, np.ndarray]) -> np.ndarray:
    """Naive late fusion: average of softmax probabilities."""
    probs = np.stack([softmax(logits[m]) for m in MODALITIES], axis=0)
    return probs.mean(axis=0).argmax(axis=-1)


def fuse_log_sum(logits: dict[str, np.ndarray]) -> np.ndarray:
    """Independent-evidence rule: sum of log-softmaxes = product of probabilities."""
    log_probs = np.stack([log_softmax(logits[m]) for m in MODALITIES], axis=0)
    return log_probs.sum(axis=0).argmax(axis=-1)


def fuse_majority_vote(logits: dict[str, np.ndarray]) -> np.ndarray:
    """Hard majority vote across the three modalities.

    Ties (three-way disagreement) are broken by the modality with the highest
    top-class confidence on that trial.
    """
    per_mod_probs = {m: softmax(logits[m]) for m in MODALITIES}
    per_mod_pred = {m: per_mod_probs[m].argmax(axis=-1) for m in MODALITIES}
    per_mod_top_conf = {m: per_mod_probs[m].max(axis=-1) for m in MODALITIES}

    N = next(iter(per_mod_pred.values())).shape[0]
    out = np.empty(N, dtype=np.int64)
    for i in range(N):
        votes = [per_mod_pred[m][i] for m in MODALITIES]
        confs = [per_mod_top_conf[m][i] for m in MODALITIES]
        # Count votes; if a class gets ≥2 votes, it wins.
        unique, counts = np.unique(votes, return_counts=True)
        top = unique[counts == counts.max()]
        if len(top) == 1:
            out[i] = top[0]
        else:
            # Three-way disagreement — pick the most-confident modality's vote.
            best_mod = int(np.argmax(confs))
            out[i] = votes[best_mod]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    subjects = discover_subjects()
    print(f"Found logits for subjects: {subjects}\n")

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        csv.writer(f).writerow(
            ["subject", "scheme", "test_acc", "note"]
        )
    # Per-trial coherence file: feeds Day 7's Listen-vs-Speak analysis.
    with open(PER_TRIAL_CSV, "w", newline="") as f:
        csv.writer(f).writerow([
            "subject", "trial", "true_label",
            "audio_pred", "vision_pred", "eeg_pred",
            "kl_audio_vision", "kl_audio_eeg", "kl_vision_eeg",
            "mean_disagreement", "all_three_agree",
            "fused_pred_mean_softmax", "fused_correct",
        ])

    def log_row(sub, scheme, acc, note=""):
        with open(OUT_CSV, "a", newline="") as f:
            csv.writer(f).writerow([sub, scheme, f"{acc:.6f}", note])

    def log_trials(rows):
        with open(PER_TRIAL_CSV, "a", newline="") as f:
            csv.writer(f).writerows(rows)

    # Header
    cols = ("Sub", "audio", "vision", "eeg",
            "mean-softmax", "log-sum", "majority-vote")
    print(" ".join(f"{c:>14}" for c in cols))
    print(" ".join(f"{'-'*14}" for _ in cols))

    per_scheme_sums: dict[str, list[float]] = {
        "audio": [], "vision": [], "eeg": [],
        "mean-softmax": [], "log-sum": [], "majority-vote": [],
    }

    for sub in subjects:
        data = load_subject(sub)

        # ---- alignment check ----
        label_seqs = {m: data[m]["labels"] for m in MODALITIES}
        ref = label_seqs["audio"]
        aligned = all(np.array_equal(label_seqs[m], ref) for m in MODALITIES)
        if not aligned:
            print(f"\n⚠ Subject {sub:02d}: label sequences NOT aligned across modalities.")
            for m in MODALITIES:
                print(f"   {m:>6} labels[:20] = {label_seqs[m][:20].tolist()}")
            log_row(sub, "alignment", 0.0, "labels misaligned across modalities")
            continue

        labels = ref
        logits = {m: data[m]["logits"] for m in MODALITIES}

        # ---- per-modality accuracy (sanity vs Day 2 CSV) ----
        per_mod_acc = {
            m: accuracy(softmax(logits[m]).argmax(axis=-1), labels)
            for m in MODALITIES
        }
        for m in MODALITIES:
            per_scheme_sums[m].append(per_mod_acc[m])
            log_row(sub, m, per_mod_acc[m], "reconstructed from logits")

        # ---- fusion schemes ----
        acc_mean = accuracy(fuse_mean_softmax(logits), labels)
        acc_log = accuracy(fuse_log_sum(logits), labels)
        acc_mv = accuracy(fuse_majority_vote(logits), labels)

        per_scheme_sums["mean-softmax"].append(acc_mean)
        per_scheme_sums["log-sum"].append(acc_log)
        per_scheme_sums["majority-vote"].append(acc_mv)

        log_row(sub, "mean-softmax",    acc_mean)
        log_row(sub, "log-sum",         acc_log)
        log_row(sub, "majority-vote",   acc_mv)

        # ---- per-trial coherence (for Day 7) ----
        p_a = softmax(logits["audio"])
        p_v = softmax(logits["vision"])
        p_e = softmax(logits["eeg"])
        top1 = {m: p.argmax(axis=-1) for m, p in
                zip(MODALITIES, (p_a, p_v, p_e))}
        kl_av = symmetric_kl(p_a, p_v)
        kl_ae = symmetric_kl(p_a, p_e)
        kl_ve = symmetric_kl(p_v, p_e)
        mean_kl = (kl_av + kl_ae + kl_ve) / 3.0
        all_agree = (top1["audio"] == top1["vision"]) & \
                    (top1["vision"] == top1["eeg"])
        fused_pred = ((p_a + p_v + p_e) / 3.0).argmax(axis=-1)

        rows = [
            [sub, i, int(labels[i]),
             int(top1["audio"][i]), int(top1["vision"][i]), int(top1["eeg"][i]),
             f"{kl_av[i]:.6f}", f"{kl_ae[i]:.6f}", f"{kl_ve[i]:.6f}",
             f"{mean_kl[i]:.6f}", bool(all_agree[i]),
             int(fused_pred[i]), bool(fused_pred[i] == labels[i])]
            for i in range(len(labels))
        ]
        log_trials(rows)

        row = (f"{sub:02d}",
               f"{per_mod_acc['audio']:.3f}",
               f"{per_mod_acc['vision']:.3f}",
               f"{per_mod_acc['eeg']:.3f}",
               f"{acc_mean:.3f}",
               f"{acc_log:.3f}",
               f"{acc_mv:.3f}")
        print(" ".join(f"{c:>14}" for c in row))

    # ---- mean row ----
    print(" ".join(f"{'-'*14}" for _ in cols))
    means = ("Mean",) + tuple(
        f"{np.mean(per_scheme_sums[k]):.3f}" if per_scheme_sums[k] else "—"
        for k in ("audio", "vision", "eeg",
                  "mean-softmax", "log-sum", "majority-vote")
    )
    print(" ".join(f"{c:>14}" for c in means))

    # ---- summary commentary ----
    best_single = max(np.mean(per_scheme_sums[m]) for m in MODALITIES)
    best_fusion = max(np.mean(per_scheme_sums[s])
                      for s in ("mean-softmax", "log-sum", "majority-vote"))
    gain = (best_fusion - best_single) * 100
    print()
    print(f"Best single-modality mean : {best_single*100:.2f}%")
    print(f"Best fusion mean          : {best_fusion*100:.2f}%")
    print(f"Fusion gain over best single: {gain:+.2f} pp")
    print()
    print(f"✅ wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
