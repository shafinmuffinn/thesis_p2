"""The four new figures of the P3 report, drawn from the result files only.

    fig_within_cross.png        per-method accuracy: within-subject (leak-free),
                                cross-subject default, cross-subject calibrated
    fig_cal_curve.png           accuracy vs number of unlabelled calibration clips
    fig_identity_probe.png      identity and emotion linear probes, default vs calibrated
    fig_audit_suppression.png   suppression matrix beside EEG's confident-error matrix

Needs results/p3_stats/*.csv (run p3_stats.py first), results/identity_probe.csv
and the day5 per-modality logits, so run it where those live (Colab):

    python make_p3_figures.py                       # writes RESULTS/p3_figures/
    python make_p3_figures.py --out LateX_P3/images

Copy the PNGs into LateX_P3/images/; the report picks them up by name.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from paths import IDX_TO_EMOTION, RESULTS

STATS = RESULTS / "p3_stats"
HEADS_CALIBRATED = {"concat_mlp", "dropout_full", "cross_attn", "dropout_av"}

# Within-subject per-modality means (Table tab:full_rollout_accuracy). The
# encoders were not epoch-selected, so these P2 figures stand; no CSV in
# p3_stats carries them.
WITHIN_UNIMODAL = {"audio_only": 0.571, "vision_only": 0.746, "eeg_only": 0.439}

LABEL = {"audio_only": "Audio", "vision_only": "Vision", "eeg_only": "EEG",
         "naive_late": "Naive late", "concat_mlp": "Concat-MLP",
         "dropout_full": "Dropout attn.", "cross_attn": "Attention",
         "dropout_av": "Dropout (AV)"}


def read(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"missing {path} -- run the script that produces it first")
    with open(path) as f:
        return list(csv.DictReader(f))


def means(name: str) -> dict[str, tuple[float, float, float]]:
    """variant -> (mean, ci_low, ci_high) from a p3_stats *_means.csv."""
    return {r["variant"]: (float(r["mean"]), float(r["ci_low"]), float(r["ci_high"]))
            for r in read(STATS / f"{name}_means.csv")}


def calibrated_naive_means() -> dict[int, float]:
    """n -> mean accuracy of centred naive late fusion (random draws only)."""
    path = RESULTS / "calibrated_naive.csv"
    if not path.exists():
        return {}
    per: dict[tuple, list[float]] = {}
    for r in read(path):
        if r["setting"] == "random" and r["variant"] == "naive_centred":
            per.setdefault((int(r["n_calib"]), int(r["subject"])), []).append(float(r["acc"]))
    by_n: dict[int, list[float]] = {}
    for (n, _), v in per.items():
        by_n.setdefault(n, []).append(np.mean(v))
    return {n: float(np.mean(v)) for n, v in by_n.items()}


def fig_within_cross(plt, out: Path) -> None:
    within = means("within_leakfree")
    cross = means("cross_default")
    calib = means("cross_subjnorm")
    order = ["audio_only", "vision_only", "eeg_only", "naive_late",
             "concat_mlp", "dropout_full", "cross_attn"]
    wkey = lambda v: v if v == "naive_late" else f"{v}:final_full"

    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = np.arange(len(order)); w = 0.27
    series = [
        ("Within-subject (leak-free)", [WITHIN_UNIMODAL.get(v) or within[wkey(v)][0] for v in order], "#4C72B0"),
        ("Cross-subject, default", [cross[v][0] for v in order], "#DD8452"),
        # Calibration acts on the learned heads' feature inputs only; unimodal
        # and naive methods are not calibrated, so no bar is drawn for them.
        ("Cross-subject, calibrated", [calib[v][0] if v in HEADS_CALIBRATED else np.nan
                                       for v in order], "#55A868"),
    ]
    for i, (name, vals, colour) in enumerate(series):
        ax.bar(x + (i - 1) * w, np.array(vals) * 100, w, label=name, color=colour)
    ax.axhline(20, color="grey", lw=0.8, ls=":")
    ax.text(-0.45, 21, "chance", color="grey", ha="left", fontsize=8)
    ax.set_xticks(x, [LABEL[v] for v in order])
    ax.set_ylabel("Mean accuracy over 42 participants (%)")
    ax.set_ylim(0, 90)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(out / "fig_within_cross.png", dpi=200); plt.close(fig)


def fig_cal_curve(plt, out: Path) -> None:
    rows = [r for r in read(STATS / "calibration_curve.csv") if r["reduced_sample"] == "0"]
    calib = means("cross_subjnorm")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, cond in zip(axes, ["random", "skewed"]):
        for v, colour in [("concat_mlp", "#55A868"), ("dropout_full", "#8172B3"),
                          ("cross_attn", "#C44E52"), ("naive_late", "#4C72B0")]:
            pts = sorted((int(r["n_calib"]), float(r["mean"]), float(r["ci_low"]), float(r["ci_high"]))
                         for r in rows if r["condition"] == cond and r["variant"] == v)
            if not pts:
                continue
            n, m, lo, hi = map(np.array, zip(*pts))
            label = "Averaging, uncalibrated" if v == "naive_late" else LABEL[v]
            ax.plot(n, m * 100, "o-", color=colour, label=label, ms=4)
            ax.fill_between(n, lo * 100, hi * 100, color=colour, alpha=0.15)
        ax.axhline(calib["concat_mlp"][0] * 100, color="#55A868", ls="--", lw=0.8)
        if cond == "random":
            # Calibrated averaging (experiment A), same random draws, n = 20 and 50.
            cn = calibrated_naive_means()
            if cn:
                n, m = zip(*sorted(cn.items()))
                ax.plot(n, np.array(m) * 100, "s--", color="#4C72B0", ms=6,
                        label="Averaging, calibrated")
        ax.set_xscale("log"); ax.set_xticks([5, 10, 20, 50, 100, 200], ["5", "10", "20", "50", "100", "200"])
        ax.set_xlabel("Unlabelled calibration clips (5 s each)")
        ax.set_title("Random clips" if cond == "random" else "Skewed clips, Dirichlet(0.3)", fontsize=10)
    axes[0].set_ylabel("Accuracy on remaining clips (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out / "fig_cal_curve.png", dpi=200); plt.close(fig)


def fig_identity_probe(plt, out: Path) -> None:
    rows = read(RESULTS / "identity_probe.csv")
    mods = ["audio", "vision", "eeg"]

    def avg(probe, m, norm):
        # Identity: one row per fold. Emotion: one row per held-out participant;
        # every participant is held out once, so the plain mean is per participant.
        return np.mean([float(r["acc"]) for r in rows
                        if r["probe"] == probe and r["modality"] == m and r["norm"] == norm])

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, probe, chance, title in [(axes[0], "identity", 1 / 42, "Participant identity (42-way)"),
                                     (axes[1], "emotion", 0.2, "Emotion, unseen participants (5-way)")]:
        x = np.arange(3); w = 0.38
        for i, (norm, name, colour) in enumerate([("global", "Default", "#DD8452"),
                                                  ("subject", "Calibrated", "#55A868")]):
            vals = [avg(probe, m, norm) * 100 for m in mods]
            ax.bar(x + (i - 0.5) * w, vals, w, label=name, color=colour)
        ax.axhline(chance * 100, color="grey", ls=":", lw=0.8)
        ax.set_xticks(x, ["Audio", "Vision", "EEG"]); ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("Linear-probe accuracy (%)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(out / "fig_identity_probe.png", dpi=200); plt.close(fig)


def fig_audit_suppression(plt, out: Path) -> None:
    from suppression_audit import eeg_confusion, events, load_within, offdiag
    recs = load_within()
    M = events(recs)[0]
    C = eeg_confusion(recs, only_av_correct=True)
    r = np.corrcoef(offdiag(M), offdiag(C))[0, 1]
    names = [IDX_TO_EMOTION[i] for i in range(5)]
    vmax = max(M.max(), C.max())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.subplots_adjust(wspace=0.45)
    for ax, mat, title, ylab in [
            (axes[0], M, f"Suppression matrix ({M.sum()} events)", "Audio-vision consensus"),
            (axes[1], C, "EEG confident errors, AV-correct trials", "Cued class")]:
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=vmax)
        for i in range(5):
            for j in range(5):
                ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if mat[i, j] > vmax / 2 else "black")
        ax.set_xticks(range(5), names, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(5), names, fontsize=8)
        ax.set_xlabel("EEG prediction"); ax.set_ylabel(ylab); ax.set_title(title, fontsize=10)
    fig.suptitle(f"Off-diagonal correlation r = {r:.3f}", fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.8)
    fig.savefig(out / "fig_audit_suppression.png", dpi=200, bbox_inches="tight"); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(RESULTS / "p3_figures"))
    ap.add_argument("--only", default="", help="comma list of: within_cross,cal_curve,identity,audit")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    figs = {"within_cross": fig_within_cross, "cal_curve": fig_cal_curve,
            "identity": fig_identity_probe, "audit": fig_audit_suppression}
    chosen = [k for k in args.only.split(",") if k] or list(figs)
    for k in chosen:
        figs[k](plt, out)
        print(f"  wrote {k}")
    print(f"figures -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
