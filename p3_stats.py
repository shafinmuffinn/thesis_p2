"""Every number the P3 report cites, with Holm-corrected tests and bootstrap CIs.

Reads the result CSVs and cached logits already on Drive -- no training -- and
writes report-ready tables to RESULTS/p3_stats/. Each section is one test
family, and Holm correction is applied within that family:

  1. cross_default     cross-subject, per-fold normalisation   (cv5_fusion.csv)
  2. cross_subjnorm    cross-subject, per-participant norm      (cv5_fusion_subjectnorm.csv)
  3. calib_effect      subject-norm vs default, per head        (both of the above)
  4. within_leakfree   within-subject, leak-free                (p2_leakfree.csv)
  5. calibration curve accuracy vs n calibration clips, with CIs (calibration_study.csv)
  6. per-class recall  pooled over held-out trials, both protocols (cv5_fusion_logits)

    python p3_stats.py
"""
from __future__ import annotations

import csv
import itertools
import sys
from pathlib import Path

import numpy as np

from paths import CHECKPOINTS, IDX_TO_EMOTION, RESULTS

OUT = RESULTS / "p3_stats"
BOOT = 10_000
RNG = np.random.default_rng(0)
HEADS = ["concat_mlp", "dropout_full", "cross_attn", "dropout_av"]
UNIMODAL = ["audio_only", "vision_only", "eeg_only"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def read(path: Path) -> list[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def latest_per_subject(rows: list[dict]) -> dict[str, dict[int, float]]:
    """variant -> {subject: acc}, keeping the last row per (fold, subject, variant)."""
    keep: dict[tuple, dict] = {}
    for r in rows:
        keep[(r["fold"], r["subject"], r["variant"])] = r
    out: dict[str, dict[int, float]] = {}
    for r in keep.values():
        out.setdefault(r["variant"], {})[int(r["subject"])] = float(r["test_acc"])
    return out


def boot_ci(x: np.ndarray) -> tuple[float, float]:
    idx = RNG.integers(0, len(x), size=(BOOT, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p); m = len(p); adj = [0.0] * m; run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (m - rank) * p[i])); adj[i] = run
    return adj


def paired(a: dict[int, float], b: dict[int, float]):
    from scipy.stats import wilcoxon
    s = sorted(set(a) & set(b))
    x, y = np.array([a[i] for i in s]), np.array([b[i] for i in s])
    p = 1.0 if np.allclose(x, y) else float(wilcoxon(x, y).pvalue)
    return (x.mean() - y.mean()) * 100, p, int((x > y).sum()), len(s)


def write(name: str, fields: list[str], rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def describe(name: str, data: dict[str, dict[int, float]], order: list[str]) -> None:
    rows = []
    print(f"\n{'variant':26s}{'n':>4s}{'mean':>9s}{'95% CI':>18s}")
    for v in order:
        if v not in data:
            continue
        x = np.array(list(data[v].values()))
        lo, hi = boot_ci(x)
        rows.append(dict(variant=v, n=len(x), mean=round(x.mean(), 4),
                         ci_low=round(lo, 4), ci_high=round(hi, 4)))
        print(f"{v:26s}{len(x):4d}{x.mean():9.4f}   [{lo:.4f}, {hi:.4f}]")
    write(f"{name}_means", ["variant", "n", "mean", "ci_low", "ci_high"], rows)


def test_family(name: str, pairs: list[tuple[str, str]],
                a_data: dict, b_data: dict | None = None) -> None:
    b_data = b_data if b_data is not None else a_data
    raw, meta = [], []
    for a, b in pairs:
        if a not in a_data or b not in b_data:
            continue
        d, p, better, n = paired(a_data[a], b_data[b])
        raw.append(p); meta.append((a, b, d, better, n))
    adj = holm(raw)
    rows = []
    print(f"\n  {name}: Holm over {len(raw)} tests")
    for (a, b, d, better, n), p0, p in zip(meta, raw, adj):
        star = " *" if p < 0.05 else ""
        print(f"    {a:24s} vs {b:24s} {d:+7.2f}pp  p={p0:.4g}  p_holm={p:.4g}{star}"
              f"  ({better}/{n})")
        rows.append(dict(a=a, b=b, delta_pp=round(d, 2), p_raw=p0, p_holm=p,
                         better=better, n=n))
    write(f"{name}_tests", ["a", "b", "delta_pp", "p_raw", "p_holm", "better", "n"], rows)


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def cross_sections() -> None:
    default = latest_per_subject(read(RESULTS / "cv5_fusion.csv"))
    subjn = latest_per_subject(read(RESULTS / "cv5_fusion_subjectnorm.csv"))
    fusion = ["naive_late"] + HEADS

    print("=" * 72 + "\n1. CROSS-SUBJECT, default normalisation\n" + "=" * 72)
    describe("cross_default", default, fusion + UNIMODAL)
    test_family("cross_default",
                [(h, "naive_late") for h in HEADS]
                + list(itertools.combinations(HEADS[:3], 2))
                + [("naive_late", "audio_only"), ("audio_only", "vision_only"),
                   ("vision_only", "eeg_only"), ("dropout_full", "dropout_av")],
                default)

    print("\n" + "=" * 72 + "\n2. CROSS-SUBJECT, per-participant calibration\n" + "=" * 72)
    describe("cross_subjnorm", subjn, fusion)
    test_family("cross_subjnorm",
                [(h, "naive_late") for h in HEADS]
                + list(itertools.combinations(HEADS[:3], 2))
                + [("dropout_full", "dropout_av")],
                subjn)

    print("\n" + "=" * 72 + "\n3. CALIBRATION EFFECT (subject-norm minus default, same head)\n"
          + "=" * 72)
    test_family("calib_effect", [(h, h) for h in HEADS], subjn, default)


def within_section() -> None:
    rows = read(RESULTS / "p2_leakfree.csv")
    by = {}
    for r in rows:
        key = r["variant"] if r["readout"] == "none" else f"{r['variant']}:{r['readout']}"
        by.setdefault(key, {})[int(r["subject"])] = float(r["acc"])
    print("\n" + "=" * 72 + "\n4. WITHIN-SUBJECT, leak-free (final_full primary)\n" + "=" * 72)
    order = ["naive_late"] + [f"{h}:{r}" for r in ("final_full", "val_selected", "test_selected")
                              for h in HEADS]
    describe("within_leakfree", by, order)
    test_family("within_leakfree",
                [(f"{h}:final_full", "naive_late") for h in HEADS]
                + [(f"{a}:final_full", f"{b}:final_full")
                   for a, b in itertools.combinations(HEADS[:3], 2)],
                by)


def calibration_curve() -> None:
    rows = read(RESULTS / "calibration_study.csv")
    cells: dict[tuple, dict[int, list[float]]] = {}
    for r in rows:
        key = (r["condition"], int(r["n_calib"]), r["variant"])
        cells.setdefault(key, {}).setdefault(int(r["subject"]), []).append(float(r["acc"]))
    out = []
    for (cond, n, v), d in sorted(cells.items()):
        x = np.array([np.mean(s) for s in d.values()])
        lo, hi = boot_ci(x)
        out.append(dict(condition=cond, n_calib=n, variant=v, n_subjects=len(x),
                        mean=round(x.mean(), 4), ci_low=round(lo, 4), ci_high=round(hi, 4),
                        reduced_sample=int(len(x) < 42)))
    write("calibration_curve",
          ["condition", "n_calib", "variant", "n_subjects", "mean", "ci_low", "ci_high",
           "reduced_sample"], out)
    flagged = sorted({(r["condition"], r["n_calib"]) for r in out if r["reduced_sample"]})
    print("\n" + "=" * 72 + "\n5. CALIBRATION CURVE -> p3_stats/calibration_curve.csv\n" + "=" * 72)
    print(f"  {len(out)} cells written.")
    if flagged:
        print(f"  Cells with fewer than 42 participants (do not plot unflagged): {flagged}")


def per_class() -> None:
    d = CHECKPOINTS / "cv5_fusion_logits"
    variants = ["naive_late", "concat_mlp", "dropout", "cross_attn"]
    out = []
    print("\n" + "=" * 72 + "\n6. PER-CLASS RECALL, pooled held-out trials\n" + "=" * 72)
    for tag, label in (("", "default"), ("_subjnorm", "subjnorm")):
        pooled = {v: [] for v in variants}; ys = []
        for k in range(5):
            path = d / f"fold{k}{tag}.npz"
            if not path.exists():
                print(f"  missing {path.name}; skipping {label}")
                break
            with np.load(path) as z:
                ys.append(z["y"])
                probs = []
                for m in ("audio", "vision", "eeg"):
                    lg = z[f"logits_{m}"]
                    e = np.exp(lg - lg.max(-1, keepdims=True)); probs.append(e / e.sum(-1, keepdims=True))
                pooled["naive_late"].append(np.mean(probs, 0).argmax(-1))
                for v in variants[1:]:
                    pooled[v].append(z[v].argmax(-1))
        else:
            y = np.concatenate(ys)
            print(f"\n  [{label}]  " + "".join(f"{IDX_TO_EMOTION[c][:9]:>11s}" for c in range(5))
                  + f"{'macro-F1':>10s}")
            for v in variants:
                p = np.concatenate(pooled[v])
                rec = [float((p[y == c] == c).mean()) for c in range(5)]
                f1s = []
                for c in range(5):
                    tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c))
                    fn = np.sum((p != c) & (y == c))
                    f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
                print(f"  {v:12s}" + "".join(f"{r:11.3f}" for r in rec) + f"{np.mean(f1s):10.3f}")
                out.append(dict(protocol=label, variant=v, macro_f1=round(float(np.mean(f1s)), 4),
                                **{IDX_TO_EMOTION[c]: round(rec[c], 4) for c in range(5)}))
    write("per_class", ["protocol", "variant", "macro_f1"]
          + [IDX_TO_EMOTION[c] for c in range(5)], out)


def main() -> int:
    cross_sections()
    within_section()
    calibration_curve()
    per_class()
    print(f"\nAll tables written to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
