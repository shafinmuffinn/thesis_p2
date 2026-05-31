"""Compute the cross-modal suppression matrix from per-modality test logits.

The suppression matrix is a 5x5 contingency table that captures which emotion
each subject *externally displayed* (via audio + vision agreement) versus
which emotion the *internally measured* EEG predicted, on trials where the
two diverged. Rows are external emotion (AV consensus), columns are internal
emotion (EEG top-1), cells count the number of subject-trial events.

Diagonal cells are zero by construction — we only count divergence events.
High off-diagonal cells (e.g. cell [Happy, Sad]) mean the system often
detected outward Happiness on trials where EEG suggested internal Sadness,
which is the classical "masking" pattern the thesis sets out to detect.

A trial is counted as a suppression event iff:
    1. Audio and Vision agree on their top-1 emotion (consensus exists)
    2. EEG top-1 differs from the AV consensus
    3. EEG's top-1 confidence >= EEG_CONF_THRESHOLD (default 0.5)
       (otherwise EEG is essentially guessing and shouldn't be counted)

Inputs:  CHECKPOINTS/day2_logits/sub{NN}_{audio,vision,eeg}.npz
         Each contains 'logits' (N, 5) and 'labels' (N,).

Outputs:
    results/suppression_matrix.csv      — the 5x5 matrix as a labelled CSV
    results/suppression_per_trial.csv   — one row per (subject, trial) with
                                          all the per-modality info needed
                                          for downstream analyses
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from paths import CHECKPOINTS, RESULTS

LOGITS_DIR = CHECKPOINTS / "day2_logits"
OUT_MATRIX_CSV = RESULTS / "suppression_matrix.csv"
OUT_TRIAL_CSV = RESULTS / "suppression_per_trial.csv"

# EAV emotion mapping. Order matches Dataload_audio.py's emotion_to_index:
# {Neutral:0, Sadness:1, Anger:2, Happiness:3, Calmness:4}
EMOTIONS = ["Neutral", "Sadness", "Anger", "Happiness", "Calmness"]
MODALITIES = ("audio", "vision", "eeg")
EEG_CONF_THRESHOLD = 0.5   # require EEG to be reasonably confident


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def discover_subjects() -> list[int]:
    if not LOGITS_DIR.exists():
        sys.exit(
            f"❌ No logit directory at {LOGITS_DIR}\n"
            f"   Sync the day2_logits/ folder from Drive to this location, "
            f"or set the CHECKPOINTS env var to point at the Drive folder."
        )
    found: dict[int, set[str]] = {}
    for p in LOGITS_DIR.glob("sub*_*.npz"):
        stem = p.stem  # e.g. 'sub01_audio'
        try:
            sub = int(stem[3:5])
            mod = stem.split("_", 1)[1]
        except (ValueError, IndexError):
            continue
        found.setdefault(sub, set()).add(mod)
    return sorted(s for s, mods in found.items() if set(MODALITIES) <= mods)


def load_subject(sub: int) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for mod in MODALITIES:
        z = np.load(LOGITS_DIR / f"sub{sub:02d}_{mod}.npz")
        out[mod] = {"logits": z["logits"], "labels": z["labels"]}
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    subjects = discover_subjects()
    if not subjects:
        sys.exit("No subjects with all 3 modalities found in day2_logits/.")
    print(f"Found subjects with all three modalities: {subjects}")

    RESULTS.mkdir(parents=True, exist_ok=True)

    matrix = np.zeros((5, 5), dtype=int)
    per_trial_rows = []

    # Running counters for the summary
    n_total = 0
    n_av_agree = 0
    n_av_agree_eeg_disagree = 0
    n_suppression_events = 0

    for sub in subjects:
        data = load_subject(sub)
        labels = np.asarray(data["audio"]["labels"])
        n = len(labels)
        n_total += n

        p_a = softmax(data["audio"]["logits"])
        p_v = softmax(data["vision"]["logits"])
        p_e = softmax(data["eeg"]["logits"])

        a_pred, a_conf = p_a.argmax(-1), p_a.max(-1)
        v_pred, v_conf = p_v.argmax(-1), p_v.max(-1)
        e_pred, e_conf = p_e.argmax(-1), p_e.max(-1)

        av_agree = a_pred == v_pred
        n_av_agree += int(av_agree.sum())

        eeg_disagrees = av_agree & (e_pred != a_pred)
        n_av_agree_eeg_disagree += int(eeg_disagrees.sum())

        suppression = eeg_disagrees & (e_conf >= EEG_CONF_THRESHOLD)
        n_suppression_events += int(suppression.sum())

        # Update the 5x5 matrix from this subject's suppression events.
        for ext, intl in zip(a_pred[suppression], e_pred[suppression]):
            matrix[ext, intl] += 1

        # Per-trial rows for downstream analyses.
        for i in range(n):
            per_trial_rows.append({
                "subject":              sub,
                "trial":                i,
                "true_label":           int(labels[i]),
                "audio_pred":           int(a_pred[i]),
                "audio_conf":           round(float(a_conf[i]), 4),
                "vision_pred":          int(v_pred[i]),
                "vision_conf":          round(float(v_conf[i]), 4),
                "eeg_pred":             int(e_pred[i]),
                "eeg_conf":             round(float(e_conf[i]), 4),
                "av_agree":             bool(av_agree[i]),
                "av_consensus":         int(a_pred[i]) if av_agree[i] else -1,
                "is_suppression_event": bool(suppression[i]),
            })

    # ---- Write the matrix CSV ----
    with open(OUT_MATRIX_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["External \\ Internal"] + EMOTIONS)
        for i, name in enumerate(EMOTIONS):
            w.writerow([name] + matrix[i].tolist())

    # ---- Write the per-trial CSV ----
    fields = list(per_trial_rows[0].keys())
    with open(OUT_TRIAL_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(per_trial_rows)

    # ---- Pretty-print the summary and the matrix ----
    pct = lambda n: f"({n/n_total*100:.1f}%)"
    print(f"\n=== Trial statistics across {len(subjects)} subjects ===")
    print(f"  Total trials                            : {n_total}")
    print(f"  AV agree on top-1                       : {n_av_agree}  {pct(n_av_agree)}")
    print(f"  AV agree, EEG disagrees                 : {n_av_agree_eeg_disagree}  "
          f"{pct(n_av_agree_eeg_disagree)}")
    print(f"  Suppression events (EEG conf ≥ {EEG_CONF_THRESHOLD})       : "
          f"{n_suppression_events}  {pct(n_suppression_events)}")

    print(f"\n=== Suppression matrix ===")
    print(f"  Rows = external emotion (AV consensus when they agree)")
    print(f"  Cols = internal emotion (EEG top-1 prediction)")
    print(f"  Cell = count of suppression events across all subjects")
    print(f"  Diagonal is zero by construction.\n")

    header = "  " + " " * 12 + "".join(f"{e[:8]:>10s}" for e in EMOTIONS)
    print(header)
    print("  " + "-" * (12 + 10 * 5))
    for i, ext in enumerate(EMOTIONS):
        cells = "".join(f"{matrix[i, j]:>10d}" for j in range(5))
        print(f"  {ext:>10s}: {cells}")

    # Row & column totals for quick interpretation
    print()
    print(f"  Row totals (× count of trials where AV displayed this emotion "
          f"but EEG disagreed):")
    for i, name in enumerate(EMOTIONS):
        print(f"    {name:>10s}: {matrix[i].sum():4d}")
    print(f"\n  Column totals (× count of trials where EEG suggested this "
          f"emotion but AV displayed something else):")
    for j, name in enumerate(EMOTIONS):
        print(f"    {name:>10s}: {matrix[:, j].sum():4d}")

    print(f"\n✅ Matrix written to: {OUT_MATRIX_CSV}")
    print(f"✅ Per-trial data written to: {OUT_TRIAL_CSV}")


if __name__ == "__main__":
    main()
