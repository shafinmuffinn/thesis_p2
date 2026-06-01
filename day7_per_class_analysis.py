"""Day 7 — Per-class analysis on Day-7 fusion logits.

Reads per-subject .npz files produced by day7_modality_dropout.py and
computes:

  1. Aggregate confusion matrix across all 42 subjects (5040 trials).
  2. Per-class precision, recall, F1 — both per-subject mean and aggregate.
  3. Degraded-modality accuracy table: full / av / ae / ve, from the CSV.
  4. Five-number summary per emotion (min/Q1/median/Q3/max test accuracy).

Outputs (on Drive):
  RESULTS/day7_confusion_matrix.csv       — 5×5 count matrix
  RESULTS/day7_per_class_metrics.csv      — precision/recall/F1 per emotion
  RESULTS/day7_degraded_modality.csv      — per-subject per-mode accuracy
  Console: formatted tables ready to copy into the thesis

Run after day7_modality_dropout.py has finished all 42 subjects.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from paths import CHECKPOINTS, RESULTS, IDX_TO_EMOTION

LOGITS_DIR  = CHECKPOINTS / "day7_fusion_logits"
RESULTS_CSV = RESULTS / "day7_fusion_dropout.csv"

EMOTIONS = [IDX_TO_EMOTION[i] for i in range(5)]
SUBJECTS = list(range(1, 43))

EVAL_MODES = ["full", "av", "ae", "ve"]
MODE_LABELS = {
    "full": "Full (A+V+E)",
    "av":   "AV only (demo path)",
    "ae":   "AE only",
    "ve":   "VE only",
}


# ---------------------------------------------------------------------------
# Load per-subject logits + confusion matrices
# ---------------------------------------------------------------------------

def load_subject(sub: int) -> dict | None:
    path = LOGITS_DIR / f"sub{sub:02d}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    preds  = z["logits"].argmax(axis=1)
    labels = z["labels"].astype(int)
    cm     = z["confusion_matrix"].astype(int)
    return {"sub": sub, "preds": preds, "labels": labels, "cm": cm}


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _divider(width=70):
    print("-" * width)


def print_confusion_matrix(cm: np.ndarray, title: str = "Confusion matrix"):
    print(f"\n{title}")
    header = f"{'':>12s}" + "".join(f"{e:>12s}" for e in EMOTIONS)
    print(header)
    _divider(12 + 12 * 5)
    for i, emo in enumerate(EMOTIONS):
        row_str = f"{emo:>12s}" + "".join(f"{cm[i, j]:>12d}" for j in range(5))
        print(row_str)


def print_per_class_table(metrics: dict[str, dict[str, float]]):
    """metrics[emotion] = {precision, recall, f1, support}"""
    print(f"\n{'Emotion':<14s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}")
    _divider()
    for emo in EMOTIONS:
        m = metrics[emo]
        print(f"{emo:<14s} {m['precision']:>10.3f} {m['recall']:>10.3f} "
              f"{m['f1']:>10.3f} {m['support']:>10.0f}")


def print_degraded_modality_table(rows: list[dict]):
    """rows: list of dicts with acc_full, acc_av, acc_ae, acc_ve per subject."""
    print(f"\n{'Mode':<22s} {'Mean':>8s} {'Std':>8s} {'Min':>8s} {'Max':>8s}")
    _divider()
    for mode in EVAL_MODES:
        col  = f"acc_{mode}"
        vals = np.array([r[col] for r in rows])
        print(f"{MODE_LABELS[mode]:<22s} {vals.mean():>8.3f} {vals.std():>8.3f} "
              f"{vals.min():>8.3f} {vals.max():>8.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Load per-subject data
    subjects_data = []
    missing = []
    for sub in SUBJECTS:
        d = load_subject(sub)
        if d is None:
            missing.append(sub)
        else:
            subjects_data.append(d)

    if missing:
        print(f"WARNING: missing logits for subjects: {missing}")
        print("Run day7_modality_dropout.py to completion before running this script.")
    if not subjects_data:
        raise SystemExit("No subject data found. Aborting.")

    print(f"Loaded logits for {len(subjects_data)} subjects.")

    # 2. Aggregate confusion matrix across all subjects
    agg_cm = sum(d["cm"] for d in subjects_data)
    all_preds  = np.concatenate([d["preds"]  for d in subjects_data])
    all_labels = np.concatenate([d["labels"] for d in subjects_data])

    print_confusion_matrix(agg_cm, title="Aggregate confusion matrix (42 subjects, 5040 trials)")

    # Row-normalise for readability
    cm_norm = agg_cm.astype(float) / agg_cm.sum(axis=1, keepdims=True).clip(min=1)
    print_confusion_matrix((cm_norm * 100).round(1).astype(int),
                            title="Confusion matrix — row-normalised (%, predicted class per true class)")

    # 3. Per-class precision / recall / F1
    prec, rec, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(5)), zero_division=0
    )
    metrics = {
        EMOTIONS[i]: {
            "precision": float(prec[i]),
            "recall":    float(rec[i]),
            "f1":        float(f1[i]),
            "support":   float(support[i]),
        }
        for i in range(5)
    }
    macro_f1 = float(f1.mean())
    print(f"\n--- Per-class metrics (aggregate, {len(subjects_data)} subjects) ---")
    print_per_class_table(metrics)
    print(f"\n  Macro-F1 (aggregate): {macro_f1:.3f}")
    print(f"  Overall accuracy    : {(all_preds == all_labels).mean():.3f}")

    # Per-subject macro-F1 distribution
    per_sub_f1 = []
    for d in subjects_data:
        _, _, f1s, _ = precision_recall_fscore_support(
            d["labels"], d["preds"], labels=list(range(5)),
            average=None, zero_division=0
        )
        per_sub_f1.append(float(f1s.mean()))
    per_sub_f1 = np.array(per_sub_f1)
    print(f"\n  Per-subject macro-F1: mean={per_sub_f1.mean():.3f}  "
          f"std={per_sub_f1.std():.3f}  "
          f"min={per_sub_f1.min():.3f}  max={per_sub_f1.max():.3f}")

    # 4. Degraded-modality accuracy table from CSV
    print(f"\n--- Degraded-modality accuracy ({len(subjects_data)} subjects) ---")
    if RESULTS_CSV.exists():
        csv_rows = []
        with open(RESULTS_CSV, newline="") as f:
            for row in csv.DictReader(f):
                csv_rows.append({k: float(v) if k != "subject" else int(v)
                                  for k, v in row.items()})
        print_degraded_modality_table(csv_rows)
    else:
        print(f"  (CSV not found: {RESULTS_CSV})")

    # 5. Per-emotion five-number accuracy summary
    print(f"\n--- Per-subject per-emotion accuracy (5-number summary) ---")
    print(f"{'Emotion':<14s} {'Min':>6s} {'Q1':>6s} {'Med':>6s} {'Q3':>6s} {'Max':>6s}")
    _divider(50)
    for i, emo in enumerate(EMOTIONS):
        per_sub_emo_acc = []
        for d in subjects_data:
            mask = d["labels"] == i
            if mask.sum() > 0:
                per_sub_emo_acc.append((d["preds"][mask] == i).mean())
        arr = np.array(per_sub_emo_acc)
        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        print(f"{emo:<14s} {arr.min():>6.3f} {q1:>6.3f} {med:>6.3f} {q3:>6.3f} {arr.max():>6.3f}")

    # 6. Save outputs
    RESULTS.mkdir(parents=True, exist_ok=True)

    # Confusion matrix CSV
    cm_path = RESULTS / "day7_confusion_matrix.csv"
    with open(cm_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred"] + EMOTIONS)
        for i, emo in enumerate(EMOTIONS):
            w.writerow([emo] + list(agg_cm[i]))
    print(f"\nSaved: {cm_path}")

    # Per-class metrics CSV
    metrics_path = RESULTS / "day7_per_class_metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["emotion", "precision", "recall", "f1", "support"])
        w.writeheader()
        for emo in EMOTIONS:
            row = {"emotion": emo}
            row.update({k: f"{v:.4f}" for k, v in metrics[emo].items()})
            w.writerow(row)
    print(f"Saved: {metrics_path}")


if __name__ == "__main__":
    main()
