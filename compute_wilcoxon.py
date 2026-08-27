"""Compute paired Wilcoxon signed-rank tests for the four method
comparisons reported in chapter 6, Section "Statistical Analysis".

All per-subject accuracies are pasted in from the day3 / day5 / day7
CSVs on Drive (verified against published summary statistics).

Outputs a clean table of:
    statistic W, two-sided p-value,
    median paired difference + IQR,
    rank-biserial effect size r,
    95% bootstrap CI on the paired mean difference.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import wilcoxon

RNG = np.random.default_rng(42)

# -------------------------------------------------------------------------
# Data (per-subject accuracies, subject ID 1..42)
# -------------------------------------------------------------------------

# Cross-attention fusion (day5_fusion.csv)
CROSS_ATTN = np.array([
    0.725000, 0.850000, 0.900000, 0.916667, 0.783333, 0.891667, 0.866667,
    0.683333, 0.800000, 0.791667, 0.750000, 0.600000, 0.866667, 0.666667,
    0.816667, 0.608333, 0.916667, 0.783333, 0.783333, 0.958333, 0.833333,
    0.825000, 0.733333, 0.825000, 0.808333, 0.850000, 0.883333, 0.825000,
    0.775000, 0.766667, 0.783333, 0.708333, 0.883333, 0.708333, 0.808333,
    0.833333, 0.733333, 0.841667, 0.816667, 0.675000, 0.958333, 0.841667,
])

# Naive late fusion, mean-softmax variant (day3_late_fusion.csv)
LATE_FUSION = np.array([
    0.658333, 0.900000, 0.808333, 0.891667, 0.750000, 0.883333, 0.800000,
    0.733333, 0.841667, 0.675000, 0.675000, 0.725000, 0.841667, 0.691667,
    0.766667, 0.691667, 0.941667, 0.741667, 0.758333, 0.925000, 0.833333,
    0.816667, 0.783333, 0.850000, 0.691667, 0.783333, 0.866667, 0.833333,
    0.591667, 0.733333, 0.816667, 0.683333, 0.866667, 0.650000, 0.591667,
    0.733333, 0.650000, 0.850000, 0.833333, 0.658333, 0.833333, 0.891667,
])

# Per-modality vision accuracy (best single modality)
VISION = np.array([
    0.625000, 0.733333, 0.833333, 0.875000, 0.658333, 0.916667, 0.725000,
    0.708333, 0.716667, 0.725000, 0.566667, 0.558333, 0.875000, 0.650000,
    0.833333, 0.566667, 0.925000, 0.741667, 0.658333, 0.916667, 0.783333,
    0.775000, 0.650000, 0.825000, 0.700000, 0.758333, 0.916667, 0.800000,
    0.725000, 0.725000, 0.733333, 0.633333, 0.841667, 0.741667, 0.675000,
    0.666667, 0.708333, 0.758333, 0.725000, 0.600000, 0.950000, 0.841667,
])

# Modality-dropout-trained fusion, full-modality (acc_full)
DROPOUT_FULL = np.array([
    0.750000, 0.866667, 0.875000, 0.958333, 0.775000, 0.950000, 0.916667,
    0.775000, 0.891667, 0.816667, 0.775000, 0.708333, 0.925000, 0.783333,
    0.866667, 0.666667, 0.983333, 0.800000, 0.825000, 0.975000, 0.875000,
    0.875000, 0.791667, 0.841667, 0.858333, 0.841667, 0.966667, 0.925000,
    0.850000, 0.783333, 0.866667, 0.700000, 0.933333, 0.733333, 0.841667,
    0.941667, 0.758333, 0.883333, 0.858333, 0.750000, 0.933333, 0.883333,
])

# Modality-dropout-trained fusion, audio-visual only (acc_av, demo path)
DROPOUT_AV = np.array([
    0.700000, 0.808333, 0.883333, 0.908333, 0.775000, 0.933333, 0.875000,
    0.716667, 0.833333, 0.791667, 0.733333, 0.683333, 0.916667, 0.758333,
    0.841667, 0.541667, 0.983333, 0.791667, 0.766667, 0.950000, 0.850000,
    0.800000, 0.758333, 0.816667, 0.825000, 0.833333, 0.975000, 0.866667,
    0.800000, 0.775000, 0.816667, 0.658333, 0.908333, 0.775000, 0.841667,
    0.858333, 0.700000, 0.850000, 0.816667, 0.750000, 0.950000, 0.816667,
])

# Concat-MLP fusion baseline (concat_mlp_baseline.py)
CONCAT_MLP = np.array([
    0.766667, 0.908333, 0.833333, 0.933333, 0.741667, 0.916667, 0.875000,
    0.658333, 0.883333, 0.800000, 0.741667, 0.758333, 0.883333, 0.708333,
    0.800000, 0.641667, 0.941667, 0.816667, 0.783333, 0.966667, 0.883333,
    0.875000, 0.791667, 0.808333, 0.800000, 0.800000, 0.841667, 0.908333,
    0.766667, 0.783333, 0.858333, 0.808333, 0.925000, 0.716667, 0.741667,
    0.850000, 0.725000, 0.858333, 0.816667, 0.725000, 0.825000, 0.850000,
])


# -------------------------------------------------------------------------
# Statistical helpers
# -------------------------------------------------------------------------

def bootstrap_paired_diff_ci(a: np.ndarray, b: np.ndarray, n: int = 10000,
                              alpha: float = 0.05) -> tuple[float, float, float]:
    """Bootstrap 95% CI for the mean paired difference a - b."""
    d = a - b
    obs_mean = float(d.mean())
    boot = np.empty(n)
    for i in range(n):
        idx = RNG.integers(0, len(d), len(d))
        boot[i] = d[idx].mean()
    lo, hi = np.quantile(boot, [alpha / 2, 1 - alpha / 2])
    return obs_mean, float(lo), float(hi)


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Effect-size estimate from Wilcoxon W: r = 1 - (2W / [n(n+1)]).
    Reports the directed signed effect; |r| in [0, 1]."""
    d = a - b
    nonzero = d[d != 0]
    n = len(nonzero)
    abs_ranks = np.argsort(np.argsort(np.abs(nonzero))) + 1
    pos_sum = float(abs_ranks[nonzero > 0].sum())
    neg_sum = float(abs_ranks[nonzero < 0].sum())
    total = n * (n + 1) / 2
    return (pos_sum - neg_sum) / total


def run_test(name: str, a: np.ndarray, b: np.ndarray,
             label_a: str, label_b: str) -> None:
    print(f"\n{'='*72}")
    print(f"  {name}")
    print(f"  {label_a}  vs  {label_b}")
    print(f"{'='*72}")

    res = wilcoxon(a, b, alternative="two-sided",
                   zero_method="wilcox", method="approx")
    d = a - b
    n_pos = int((d > 0).sum())
    n_neg = int((d < 0).sum())
    n_tie = int((d == 0).sum())
    n_eff = n_pos + n_neg

    mean_diff, ci_lo, ci_hi = bootstrap_paired_diff_ci(a, b)
    r_rb = rank_biserial(a, b)

    print(f"  n (subjects)                : 42")
    print(f"  n_effective (non-zero pairs): {n_eff}")
    print(f"  positive pairs ({label_a} > {label_b}): {n_pos}")
    print(f"  negative pairs ({label_a} < {label_b}): {n_neg}")
    print(f"  tied pairs                  : {n_tie}")
    print()
    print(f"  Mean(a)     : {a.mean():.4f}")
    print(f"  Mean(b)     : {b.mean():.4f}")
    print(f"  Mean(a - b) : {mean_diff:+.4f}")
    print(f"  Median(a-b) : {np.median(d):+.4f}")
    print(f"  IQR(a - b)  : [{np.quantile(d, 0.25):+.4f}, "
          f"{np.quantile(d, 0.75):+.4f}]")
    print()
    print(f"  Wilcoxon statistic W : {res.statistic:.2f}")
    print(f"  Two-sided p-value    : {res.pvalue:.4g}")
    print(f"  Effect size (rank-biserial r) : {r_rb:+.3f}")
    print(f"  95% bootstrap CI on mean diff : "
          f"[{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print()
    decision = "reject" if res.pvalue < 0.05 else "fail to reject"
    print(f"  Decision at alpha = 0.05 : {decision} the null")


def main() -> None:
    # Sanity check the data
    print(f"  CROSS_ATTN   n={len(CROSS_ATTN):2d}  "
          f"mean={CROSS_ATTN.mean():.4f}  std={CROSS_ATTN.std(ddof=0):.4f}")
    print(f"  LATE_FUSION  n={len(LATE_FUSION):2d}  "
          f"mean={LATE_FUSION.mean():.4f}  std={LATE_FUSION.std(ddof=0):.4f}")
    print(f"  VISION       n={len(VISION):2d}  "
          f"mean={VISION.mean():.4f}  std={VISION.std(ddof=0):.4f}")
    print(f"  DROPOUT_FULL n={len(DROPOUT_FULL):2d}  "
          f"mean={DROPOUT_FULL.mean():.4f}  std={DROPOUT_FULL.std(ddof=0):.4f}")
    print(f"  DROPOUT_AV   n={len(DROPOUT_AV):2d}  "
          f"mean={DROPOUT_AV.mean():.4f}  std={DROPOUT_AV.std(ddof=0):.4f}")
    print(f"  CONCAT_MLP   n={len(CONCAT_MLP):2d}  "
          f"mean={CONCAT_MLP.mean():.4f}  std={CONCAT_MLP.std(ddof=0):.4f}")

    # RQ3: cross-attention vs naive late fusion
    run_test("RQ3 -- architectural contribution",
             CROSS_ATTN, LATE_FUSION,
             "Cross-attention", "Naive late fusion")

    # Dropout vs cross-attention (training discipline)
    run_test("Modality dropout vs cross-attention (training gain)",
             DROPOUT_FULL, CROSS_ATTN,
             "Cross-attn + dropout", "Cross-attn")

    # RQ2: best single modality (vision) vs naive late fusion
    run_test("RQ2 -- fusion gain",
             LATE_FUSION, VISION,
             "Naive late fusion", "Vision-only")

    # Bonus: zero-EEG demo path vs cross-attention (does the demo cost much?)
    run_test("Demo path -- AV-only dropout vs full-modality cross-attn",
             DROPOUT_AV, CROSS_ATTN,
             "Dropout + AV-only", "Cross-attn (full)")

    # Concat-MLP comparisons (the new baseline)
    run_test("Concat-MLP vs cross-attention fusion",
             CONCAT_MLP, CROSS_ATTN,
             "Concat-MLP", "Cross-attn")

    run_test("Concat-MLP vs naive late fusion",
             CONCAT_MLP, LATE_FUSION,
             "Concat-MLP", "Naive late fusion")

    run_test("Modality-dropout cross-attn vs Concat-MLP",
             DROPOUT_FULL, CONCAT_MLP,
             "Cross-attn + dropout", "Concat-MLP")


if __name__ == "__main__":
    main()
