"""Is the attention fusion actually dynamic, and does it attend sensibly?

The thesis title claims "dynamic attention-based fusion". This reads the
attention weights of the trained cross-subject heads on held-out participants
and asks three questions, before and after per-participant calibration.

Pre-specified design (fixed before any result was seen)
-------------------------------------------------------
Heads: `cross_attn` and `dropout` (the two attention models), trained by
cv_pipeline Stage C under the default protocol (fold{k}_*.pt, per-fold
standardiser) and under --subject-norm (fold{k}_subjnorm_*.pt, per-participant
z-scoring, transductive). Held-out participants only.

Weight given to each modality on one trial (PRIMARY): attention rollout
(Abnar & Zuidema 2020) through both encoder layers, with the residual
connection modelled as 0.5*A + 0.5*I per layer and attention averaged over
heads. Because the fused vector is the MEAN of the three output tokens, a
modality's weight is the mean over query tokens of its rollout column, so the
three weights sum to 1. SECONDARY (descriptive only): raw last-layer attention.

Q1  Level. Mean weight per modality, per protocol and head (bootstrap CI over
    participants).
Q2  Dynamics. Within-participant standard deviation of each modality's weight
    across trials. A fixed-weight fusion would give 0.
Q3  Sense. Within each participant, mean weight on modality m on trials where
    encoder m alone is correct minus trials where it is wrong. Positive means
    attention moves toward a modality when it is right. Paired Wilcoxon over
    participants against 0; Holm over the 12 tests (3 modalities x 2 heads x 2
    protocols).
Q4  Shift. Change in each modality's mean weight from default to calibrated,
    paired over participants; Holm over the 6 tests (3 modalities x 2 heads).

Caveat for the report: attention weights are not a complete explanation of a
model's decision (Jain & Wallace 2019); they describe how the tokens are mixed.

Inference only; CPU is fine.

    python attention_analysis.py
    python attention_analysis.py --selftest       # CPU check, no data needed
"""
from __future__ import annotations

import argparse
import csv
import sys

import numpy as np
import torch

MODS = ["audio", "vision", "eeg"]
HEADS = ["cross_attn", "dropout"]
PROTOCOLS = {"default": "", "calibrated": "_subjnorm"}
OUT_NAME = "attention_analysis.csv"
FIELDS = ["protocol", "head", "fold", "subject", "trial", "y", "head_correct"] + \
         [f"w_{m}" for m in MODS] + [f"raw_{m}" for m in MODS] + [f"{m}_correct" for m in MODS]


@torch.no_grad()
def attention_weights(model, xa, xv, xe) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(rollout weights (B,3), last-layer raw weights (B,3), logits (B,C)).

    Re-runs the model's forward pass layer by layer so the attention matrices can
    be read, and checks that the re-run reproduces the model's own logits.
    """
    a = model.proj_audio(xa) + model.modality_embed[0]
    v = model.proj_vision(xv) + model.modality_embed[1]
    e = model.proj_eeg(xe) + model.modality_embed[2]
    x = torch.stack([a, v, e], dim=1)
    eye = torch.eye(3).expand(x.shape[0], 3, 3)
    rollout = eye.clone()
    raw = None
    for layer in model.transformer.layers:
        h = layer.norm1(x) if layer.norm_first else x
        _, attn = layer.self_attn(h, h, h, need_weights=True, average_attn_weights=True)
        step = 0.5 * attn + 0.5 * eye
        step = step / step.sum(dim=-1, keepdim=True)
        rollout = step @ rollout
        raw = attn
        x = layer(x)
    if model.transformer.norm is not None:
        x = model.transformer.norm(x)
    logits = model.classifier(model.post_norm(x).mean(dim=1))
    ref = model(xa, xv, xe)["logits"]
    if not torch.allclose(logits, ref, atol=1e-4):
        raise RuntimeError("layer-by-layer re-run does not reproduce the model's logits")
    # Fused vector = mean over the three query tokens -> column means.
    return (rollout.mean(dim=1).numpy(), raw.mean(dim=1).numpy(), logits.numpy())


def run(folds: list[int]) -> list[dict]:
    import cv_pipeline as cv
    from cv_split import test_subjects

    dims = {"audio_dim": cv.FEAT_DIMS["audio"], "vision_dim": cv.FEAT_DIMS["vision"],
            "eeg_dim": cv.FEAT_DIMS["eeg"]}
    rows = []
    for k in folds:
        test = cv.load_fold_features(k, test_subjects(k))
        for proto, tag in PROTOCOLS.items():
            if proto == "calibrated":
                feats = cv.subject_standardize(test)
            else:
                path = cv.FUSION_STATE_DIR / f"fold{k}_standardizer.npz"
                if not path.exists():
                    raise FileNotFoundError(f"{path} missing -- re-run Stage C (default) for fold {k}")
                with np.load(path) as z:
                    stats = {m: (z[f"{m}_mu"], z[f"{m}_sd"]) for m in MODS}
                feats = cv.apply_standardizer(test, stats)
            x = {m: torch.from_numpy(np.asarray(feats[m], dtype=np.float32)) for m in MODS}
            for name in HEADS:
                path = cv.FUSION_STATE_DIR / f"fold{k}{tag}_{name}.pt"
                if not path.exists():
                    raise FileNotFoundError(f"{path} missing -- run Stage C for fold {k}")
                model = cv.TrimodalAttentionFusion(**dims)
                model.load_state_dict(torch.load(path, map_location="cpu"))
                model.eval()
                w, raw, logits = attention_weights(model, x["audio"], x["vision"], x["eeg"])
                pred = logits.argmax(-1)
                enc_ok = {m: test[f"logits_{m}"].argmax(-1) == test["y"] for m in MODS}
                for i in range(len(test["y"])):
                    rows.append(dict(
                        protocol=proto, head=name, fold=k, subject=int(test["subject"][i]),
                        trial=i, y=int(test["y"][i]), head_correct=int(pred[i] == test["y"][i]),
                        **{f"w_{m}": float(w[i, j]) for j, m in enumerate(MODS)},
                        **{f"raw_{m}": float(raw[i, j]) for j, m in enumerate(MODS)},
                        **{f"{m}_correct": int(enc_ok[m][i]) for m in MODS}))
                acc = float((pred == test["y"]).mean())
                print(f"  fold {k} {proto:10s} {name:10s} acc {acc:.4f}  "
                      f"mean w " + " ".join(f"{m[0]}={w[:, j].mean():.3f}"
                                            for j, m in enumerate(MODS)), flush=True)
    return rows


def _holm(p: list[float]) -> list[float]:
    order = np.argsort(p); m = len(p); adj = [0.0] * m; r = 0.0
    for rank, i in enumerate(order):
        r = max(r, min(1.0, (m - rank) * p[i])); adj[i] = r
    return adj


def summarise(rows: list[dict]) -> None:
    from scipy.stats import wilcoxon
    rng = np.random.default_rng(0)

    def by_subject(proto, head):
        d: dict[int, list[dict]] = {}
        for r in rows:
            if r["protocol"] == proto and r["head"] == head:
                d.setdefault(int(r["subject"]), []).append(r)
        return d

    print("\nQ1 level and Q2 dynamics (rollout; mean over participants)")
    print(f"  {'protocol':11s}{'head':11s}" + "".join(f"{'w_' + m:>22s}" for m in MODS)
          + "".join(f"{'sd_' + m:>11s}" for m in MODS))
    for proto in PROTOCOLS:
        for head in HEADS:
            d = by_subject(proto, head)
            cells, sds = [], []
            for m in MODS:
                per = np.array([np.mean([float(r[f"w_{m}"]) for r in rs]) for rs in d.values()])
                boots = per[rng.integers(0, len(per), (10_000, len(per)))].mean(1)
                cells.append(f"{per.mean():.3f}[{np.percentile(boots, 2.5):.3f},"
                             f"{np.percentile(boots, 97.5):.3f}]")
                sds.append(np.mean([np.std([float(r[f"w_{m}"]) for r in rs]) for rs in d.values()]))
            print(f"  {proto:11s}{head:11s}" + "".join(f"{c:>22s}" for c in cells)
                  + "".join(f"{s:11.3f}" for s in sds))

    print("\nQ3 does attention move toward a modality when that modality is right?")
    tests, labels = [], []
    for proto in PROTOCOLS:
        for head in HEADS:
            d = by_subject(proto, head)
            for m in MODS:
                diffs = []
                for rs in d.values():
                    right = [float(r[f"w_{m}"]) for r in rs if int(r[f"{m}_correct"])]
                    wrong = [float(r[f"w_{m}"]) for r in rs if not int(r[f"{m}_correct"])]
                    if right and wrong:
                        diffs.append(np.mean(right) - np.mean(wrong))
                diffs = np.array(diffs)
                tests.append(float(wilcoxon(diffs).pvalue))
                labels.append((proto, head, m, diffs.mean(), int((diffs > 0).sum()), len(diffs)))
    for (proto, head, m, dm, pos, n), p0, p in zip(labels, tests, _holm(tests)):
        print(f"  {proto:11s}{head:11s}{m:7s} right-minus-wrong {dm:+.4f}  "
              f"p={p0:.3g}  p_holm={p:.3g}{' *' if p < 0.05 else ''}  ({pos}/{n} positive)")

    print("\nQ4 shift in mean weight, calibrated minus default")
    tests, labels = [], []
    for head in HEADS:
        d0, d1 = by_subject("default", head), by_subject("calibrated", head)
        subs = sorted(set(d0) & set(d1))
        for m in MODS:
            a = np.array([np.mean([float(r[f"w_{m}"]) for r in d1[s]]) for s in subs])
            b = np.array([np.mean([float(r[f"w_{m}"]) for r in d0[s]]) for s in subs])
            tests.append(float(wilcoxon(a, b).pvalue))
            labels.append((head, m, (a - b).mean(), int((a > b).sum()), len(subs)))
    for (head, m, dm, pos, n), p0, p in zip(labels, tests, _holm(tests)):
        print(f"  {head:11s}{m:7s} {dm:+.4f}  p={p0:.3g}  p_holm={p:.3g}"
              f"{' *' if p < 0.05 else ''}  ({pos}/{n} increased)")


def selftest() -> int:
    from fusion.trimodal_attention import TrimodalAttentionFusion
    torch.manual_seed(0)
    model = TrimodalAttentionFusion().eval()
    xa, xv, xe = torch.randn(16, 768), torch.randn(16, 768), torch.randn(16, 960)
    w, raw, logits = attention_weights(model, xa, xv, xe)   # raises if re-run diverges
    assert w.shape == (16, 3) and np.allclose(w.sum(1), 1, atol=1e-5), w.sum(1)
    assert raw.shape == (16, 3) and np.allclose(raw.sum(1), 1, atol=1e-5)
    assert (w > 0).all() and w.std(0).max() > 0, "weights should vary across trials"
    print(f"selftest OK: re-run matches logits; weights sum to 1; "
          f"example trial weights {np.round(w[0], 3)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--summary-only", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    from paths import RESULTS
    out = RESULTS / OUT_NAME
    if args.summary_only:
        with open(out) as f:
            summarise(list(csv.DictReader(f)))
        return 0
    rows = run([int(x) for x in args.folds.split(",") if x.strip()])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {len(rows):,} rows -> {out}")
    summarise(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
