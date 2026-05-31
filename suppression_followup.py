"""Follow-up analyses on suppression_per_trial.csv.

Reads the per-trial CSV produced by suppression_matrix.py and produces two
diagnostic analyses that the headline matrix alone cannot answer:

    1. Per-subject suppression matrices.
       The headline matrix sums events across all subjects. If a single
       outlier subject is driving the visible patterns, the result is
       fragile. This analysis prints a separate 5x5 matrix per subject so
       we can see whether the patterns (Calmness->Neutral, Anger->Happiness,
       Happiness->Sadness, ...) appear in each subject independently or
       come from one subject only.

    2. EEG marginal distribution comparison.
       The headline matrix showed Neutral as the most common 'internal'
       emotion when AV disagrees with EEG (column total 13). This could
       be real internal state OR a model-uncertainty artefact (EEG model
       defaulting to Neutral when confused). To disambiguate, we compare
       the marginal distribution of EEG top-1 predictions across three
       subsets: (a) all trials, (b) coherent trials where EEG agrees with
       AV, (c) suppression-event trials where EEG confidently disagrees
       with AV. If Neutral is dramatically inflated in (c) relative to (a),
       it's evidence of the model bias.

Outputs:
    results/suppression_per_subject.csv  - long-format per-subject matrix data
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from paths import RESULTS

IN_CSV = RESULTS / "suppression_per_trial.csv"
OUT_CSV = RESULTS / "suppression_per_subject.csv"

EMOTIONS = ["Neutral", "Sadness", "Anger", "Happiness", "Calmness"]


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def load_rows() -> list[dict]:
    rows = []
    with open(IN_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "subject":              int(r["subject"]),
                "trial":                int(r["trial"]),
                "audio_pred":           int(r["audio_pred"]),
                "vision_pred":          int(r["vision_pred"]),
                "eeg_pred":             int(r["eeg_pred"]),
                "eeg_conf":             float(r["eeg_conf"]),
                "av_agree":             r["av_agree"] == "True",
                "av_consensus":         int(r["av_consensus"]),
                "is_suppression_event": r["is_suppression_event"] == "True",
            })
    return rows


# ---------------------------------------------------------------------------
# Analysis 1 — per-subject matrices
# ---------------------------------------------------------------------------

def per_subject_matrices(rows: list[dict]) -> dict[int, np.ndarray]:
    subjects = sorted({r["subject"] for r in rows})
    mats: dict[int, np.ndarray] = {}
    for sub in subjects:
        m = np.zeros((5, 5), dtype=int)
        for r in rows:
            if r["subject"] != sub or not r["is_suppression_event"]:
                continue
            m[r["av_consensus"], r["eeg_pred"]] += 1
        mats[sub] = m
    return mats


def print_per_subject(mats: dict[int, np.ndarray]) -> None:
    print("=" * 78)
    print("Analysis 1 — Per-subject suppression matrices")
    print("=" * 78)
    print()
    for sub, m in mats.items():
        total = int(m.sum())
        print(f"Subject {sub:02d} — {total} suppression events")
        if total == 0:
            print("  (no suppression events)\n")
            continue
        header = "    " + " " * 11 + "".join(f"{e[:8]:>9s}" for e in EMOTIONS)
        print(header)
        print("    " + "-" * (11 + 9 * 5))
        for i, ext in enumerate(EMOTIONS):
            cells = "".join(f"{m[i, j]:>9d}" for j in range(5))
            print(f"    {ext:>10s} {cells}")
        print()

    # Consistency summary: which cells appear in multiple subjects?
    print("Cells appearing in ≥2 subjects (cell counts side-by-side):")
    subjects = sorted(mats.keys())
    any_consistent = False
    for i, ext in enumerate(EMOTIONS):
        for j, intl in enumerate(EMOTIONS):
            counts = [int(mats[s][i, j]) for s in subjects]
            nonzero = sum(1 for c in counts if c > 0)
            if nonzero >= 2:
                pretty = " / ".join(f"sub{s:02d}={c}" for s, c in zip(subjects, counts))
                print(f"  {ext:>10s} → {intl:<10s}  total {sum(counts):>2d}   ({pretty})")
                any_consistent = True
    if not any_consistent:
        print("  (no cell has events in ≥2 subjects — patterns are subject-specific)")
    print()


# ---------------------------------------------------------------------------
# Analysis 2 — EEG marginal comparison
# ---------------------------------------------------------------------------

def eeg_marginal(rows: list[dict], pred) -> Counter:
    c = Counter()
    for r in rows:
        if pred(r):
            c[r["eeg_pred"]] += 1
    return c


def print_marginals(rows: list[dict]) -> None:
    print("=" * 78)
    print("Analysis 2 — EEG prediction distribution comparison")
    print("=" * 78)
    print()

    all_c = eeg_marginal(rows, lambda r: True)
    coh_c = eeg_marginal(
        rows,
        lambda r: r["av_agree"] and r["eeg_pred"] == r["av_consensus"],
    )
    sup_c = eeg_marginal(rows, lambda r: r["is_suppression_event"])

    t_all = sum(all_c.values())
    t_coh = sum(coh_c.values())
    t_sup = sum(sup_c.values())

    print(f"  {'EEG predicts':>13s} | {f'All trials (n={t_all})':>22s} | "
          f"{f'Coherent (n={t_coh})':>20s} | {f'Suppression (n={t_sup})':>22s}")
    print("  " + "-" * 85)
    for i, name in enumerate(EMOTIONS):
        a = all_c.get(i, 0); ap = a / t_all * 100 if t_all else 0
        c = coh_c.get(i, 0); cp = c / t_coh * 100 if t_coh else 0
        s = sup_c.get(i, 0); sp = s / t_sup * 100 if t_sup else 0
        print(f"  {name:>13s} | {a:>4d}  ({ap:>5.1f}%)        | "
              f"{c:>4d}  ({cp:>5.1f}%)      | "
              f"{s:>4d}  ({sp:>5.1f}%)")

    print()
    # Headline diagnostic: is Neutral disproportionately predicted on
    # suppression events?
    a_neu = all_c.get(0, 0) / t_all * 100 if t_all else 0
    s_neu = sup_c.get(0, 0) / t_sup * 100 if t_sup else 0
    delta = s_neu - a_neu
    print(f"Neutral inflation on suppression events: "
          f"{s_neu:.1f}% (suppression) vs {a_neu:.1f}% (all trials) = "
          f"{delta:+.1f} pp")
    if delta > 15:
        print("  → Large positive inflation. Suggests EEG model defaults to")
        print("    Neutral when confused. Treat the Neutral column of the")
        print("    suppression matrix with caution — interpret as 'EEG was")
        print("    uncertain' rather than 'subject was internally Neutral'.")
    elif delta > 5:
        print("  → Moderate inflation. Some EEG-model bias is plausible but")
        print("    the Neutral column likely also reflects real internal state.")
    else:
        print("  → No meaningful inflation. The Neutral column of the matrix")
        print("    reflects genuine EEG signal, not model uncertainty.")
    print()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_per_subject_csv(mats: dict[int, np.ndarray]) -> None:
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "external_emotion", "internal_emotion", "count"])
        for sub in sorted(mats.keys()):
            m = mats[sub]
            for i, ext in enumerate(EMOTIONS):
                for j, intl in enumerate(EMOTIONS):
                    if m[i, j] > 0:
                        w.writerow([sub, ext, intl, int(m[i, j])])
    print(f"✅ Per-subject matrix data saved to: {OUT_CSV}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not IN_CSV.exists():
        sys.exit(
            f"❌ {IN_CSV} not found. Run suppression_matrix.py first to "
            f"generate it."
        )

    rows = load_rows()
    print(f"Loaded {len(rows)} per-trial rows from {IN_CSV.name}\n")

    mats = per_subject_matrices(rows)
    print_per_subject(mats)
    print_marginals(rows)
    save_per_subject_csv(mats)


if __name__ == "__main__":
    main()
