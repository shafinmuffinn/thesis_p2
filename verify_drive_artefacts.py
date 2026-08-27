"""Verify that the Drive artefacts from the Day-2 → Day-7 rollout are intact.

Run this in Colab (or anywhere MyDrive/Thesis_EAV is mounted) before starting
the P3 workstreams. It checks *presence and integrity* -- not just that files
exist, but that each .npz opens, carries the expected keys, and has the
expected shapes. A file that exists but was truncated by a Drive sync failure
will be reported as corrupt rather than passing silently.

Usage
-----
    # Colab, after mounting Drive and setting the env vars:
    !python verify_drive_artefacts.py

    # Quick pass, skips opening every archive:
    !python verify_drive_artefacts.py --fast

Exit code is 0 if every P3-critical artefact class is intact, 1 otherwise.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np

from paths import CHECKPOINTS, EAV_PICKLES, RESULTS, summary

SUBJECTS = list(range(1, 43))

FOLDER = {"audio": "Audio", "vision": "Vision", "eeg": "EEG"}
SUFFIX = {"audio": "aud", "vision": "vis", "eeg": "eeg"}
MODALITIES = ["audio", "vision", "eeg"]

# Per-subject EAV split: h_idx=56 -> 280 train / 120 test.
N_TRAIN, N_TEST, N_CLASSES = 280, 120, 5
FEAT_DIMS = {"audio": 768, "vision": 768, "eeg": 960}

RESULT_CSVS = [
    "day5_fusion.csv",
    "day3_late_fusion.csv",
    "suppression_matrix.csv",
    "suppression_per_trial.csv",
    "suppression_per_subject.csv",
]


class Report:
    """Accumulates per-class results and prints a verdict."""

    def __init__(self) -> None:
        self.classes: list[dict] = []

    def add(self, name: str, critical: bool, expected: int,
            missing: list[str], corrupt: list[tuple[str, str]],
            note: str = "") -> None:
        self.classes.append({
            "name": name, "critical": critical, "expected": expected,
            "missing": missing, "corrupt": corrupt, "note": note,
        })

    def print(self) -> bool:
        print("=" * 72)
        print("ARTEFACT VERIFICATION")
        print("=" * 72)
        ok_overall = True

        for c in self.classes:
            found = c["expected"] - len(c["missing"])
            bad = len(c["corrupt"])
            healthy = found - bad
            tag = "CRITICAL" if c["critical"] else "optional"
            status = "OK" if (not c["missing"] and not bad) else "PROBLEM"
            if status == "PROBLEM" and c["critical"]:
                ok_overall = False

            print(f"\n[{status:7s}] {c['name']}  ({tag})")
            print(f"           {healthy}/{c['expected']} intact"
                  f"   missing={len(c['missing'])}   corrupt={bad}")
            if c["note"]:
                print(f"           note: {c['note']}")

            for m in c["missing"][:8]:
                print(f"             missing: {m}")
            if len(c["missing"]) > 8:
                print(f"             ... and {len(c['missing']) - 8} more missing")
            for name, err in c["corrupt"][:8]:
                print(f"             corrupt: {name} -- {err}")
            if bad > 8:
                print(f"             ... and {bad - 8} more corrupt")

        print("\n" + "=" * 72)
        print("P3 READINESS")
        print("=" * 72)
        by_name = {c["name"]: c for c in self.classes}

        def intact(name: str) -> bool:
            c = by_name.get(name)
            return bool(c) and not c["missing"] and not c["corrupt"]

        pickles_ok = intact("EAV source pickles")
        feats_ok = intact("Fusion features (day5_features)")
        states_ok = intact("Encoder state dicts (day5_state_dicts)")
        logits_ok = intact("Per-modality logits (day5_per_modality_logits)")

        # Track A does NOT depend on the per-subject caches: those encoders each
        # saw their own test subject, so reusing them cross-subject would leak.
        # It needs the raw pickles to retrain fold-wise encoders from scratch.
        tracks = [
            ("Track A - 5-fold subject-wise CV",
             pickles_ok,
             "needs raw pickles (fold-wise encoders must be retrained; the "
             "cached per-subject encoders leak the held-out subject)"),
            ("Track B - sequence features + 6-way CMA",
             pickles_ok and states_ok,
             "needs pickles + state dicts to re-extract without re-training"),
            ("Track C - MERCL contrastive pre-training",
             feats_ok,
             "can start immediately on cached pooled features"),
            ("Reproduce P2 numbers / suppression matrix",
             logits_ok,
             "needs cached per-modality logits only"),
        ]
        for name, ready, why in tracks:
            print(f"  [{'READY   ' if ready else 'BLOCKED '}] {name}")
            print(f"             {why}")

        print("\n" + "=" * 72)
        if ok_overall:
            print("RESULT: PASS -- every critical artefact class is intact.")
        else:
            print("RESULT: FAIL -- critical artefacts are missing or corrupt.")
            print("        See the regeneration costs printed above.")
        print("=" * 72)
        return ok_overall


def check_npz(path: Path, expected: dict[str, tuple | None]) -> str | None:
    """Open an .npz and validate keys/shapes. Returns an error string or None."""
    try:
        with np.load(path, allow_pickle=False) as z:
            for key, shape in expected.items():
                if key not in z:
                    return f"missing key '{key}'"
                if shape is not None and z[key].shape != shape:
                    return f"{key} shape {z[key].shape}, expected {shape}"
    except Exception as e:  # truncated / not a zip / bad magic
        return f"{type(e).__name__}: {e}"
    return None


def verify_pickles(rep: Report, fast: bool) -> None:
    missing, corrupt = [], []
    for sub in SUBJECTS:
        for mod in MODALITIES:
            p = EAV_PICKLES / FOLDER[mod] / f"subject_{sub:02d}_{SUFFIX[mod]}.pkl"
            if not p.exists():
                missing.append(p.name)
            elif p.stat().st_size == 0:
                corrupt.append((p.name, "zero bytes"))
    rep.add("EAV source pickles", True, len(SUBJECTS) * 3, missing, corrupt,
            note="126 files = 42 subjects x 3 modalities. Regenerating these "
                 "means re-downloading ~46 GB from Zenodo.")


def verify_features(rep: Report, fast: bool) -> None:
    d = CHECKPOINTS / "day5_features"
    missing, corrupt = [], []
    expected = {
        "train_audio": (N_TRAIN, FEAT_DIMS["audio"]),
        "train_vision": (N_TRAIN, FEAT_DIMS["vision"]),
        "train_eeg": (N_TRAIN, FEAT_DIMS["eeg"]),
        "test_audio": (N_TEST, FEAT_DIMS["audio"]),
        "test_vision": (N_TEST, FEAT_DIMS["vision"]),
        "test_eeg": (N_TEST, FEAT_DIMS["eeg"]),
        "train_y": (N_TRAIN,),
        "test_y": (N_TEST,),
    }
    for sub in SUBJECTS:
        p = d / f"sub{sub:02d}.npz"
        if not p.exists():
            missing.append(p.name)
            continue
        if fast:
            continue
        err = check_npz(p, expected)
        if err:
            corrupt.append((p.name, err))
    rep.add("Fusion features (day5_features)", True, len(SUBJECTS), missing, corrupt,
            note="Pooled per-modality features. Track C runs directly off these.")


def verify_per_modality_logits(rep: Report, fast: bool) -> None:
    d = CHECKPOINTS / "day5_per_modality_logits"
    missing, corrupt = [], []
    expected = {"logits": (N_TEST, N_CLASSES), "labels": (N_TEST,)}
    for sub in SUBJECTS:
        for mod in MODALITIES:
            p = d / f"sub{sub:02d}_{mod}.npz"
            if not p.exists():
                missing.append(p.name)
                continue
            if fast:
                continue
            err = check_npz(p, expected)
            if err:
                corrupt.append((p.name, err))
    rep.add("Per-modality logits (day5_per_modality_logits)", True,
            len(SUBJECTS) * 3, missing, corrupt,
            note="Backs the suppression matrix and late-fusion baselines.")


def verify_state_dicts(rep: Report, fast: bool) -> None:
    d = CHECKPOINTS / "day5_state_dicts"
    missing, corrupt = [], []
    for sub in SUBJECTS:
        for mod in MODALITIES:
            p = d / f"sub{sub:02d}_{mod}.pt"
            if not p.exists():
                missing.append(p.name)
            elif p.stat().st_size < 1024:
                corrupt.append((p.name, f"only {p.stat().st_size} bytes"))
    rep.add("Encoder state dicts (day5_state_dicts)", True, len(SUBJECTS) * 3,
            missing, corrupt,
            note="Fine-tuned AST/ViT/EEGNet weights. Track B re-extracts "
                 "sequence features from these without re-training.")


def verify_fusion_logits(rep: Report, fast: bool) -> None:
    d = CHECKPOINTS / "day5_fusion_logits"
    missing, corrupt = [], []
    expected = {"logits": (N_TEST, N_CLASSES), "labels": (N_TEST,)}
    for sub in SUBJECTS:
        p = d / f"sub{sub:02d}.npz"
        if not p.exists():
            missing.append(p.name)
            continue
        if fast:
            continue
        err = check_npz(p, expected)
        if err:
            corrupt.append((p.name, err))
    rep.add("Fusion logits (day5_fusion_logits)", False, len(SUBJECTS),
            missing, corrupt,
            note="Cheap to regenerate from features if absent.")


def verify_results(rep: Report, fast: bool) -> None:
    missing, corrupt = [], []
    for name in RESULT_CSVS:
        p = RESULTS / name
        if not p.exists():
            missing.append(name)
        elif p.stat().st_size == 0:
            corrupt.append((name, "zero bytes"))
    rep.add("Result CSVs", False, len(RESULT_CSVS), missing, corrupt,
            note="Reported numbers; regenerable from logits.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fast", action="store_true",
                    help="check presence only; skip opening each archive")
    args = ap.parse_args()

    print(summary())
    print(f"\nmode: {'fast (presence only)' if args.fast else 'full (opens every archive)'}\n")

    for root, label in [(EAV_PICKLES, "EAV_PICKLES"), (CHECKPOINTS, "CHECKPOINTS")]:
        if not root.exists():
            print(f"WARNING: {label} does not exist at {root}")
            print("         Is Drive mounted and are the env vars set?\n")

    rep = Report()
    for fn in (verify_pickles, verify_features, verify_per_modality_logits,
               verify_state_dicts, verify_fusion_logits, verify_results):
        try:
            fn(rep, args.fast)
        except Exception:
            print(f"ERROR while running {fn.__name__}:")
            traceback.print_exc()
            return 2

    return 0 if rep.print() else 1


if __name__ == "__main__":
    sys.exit(main())
