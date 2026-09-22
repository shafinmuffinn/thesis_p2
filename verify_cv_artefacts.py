"""Verify that the cross-subject CV artefacts are complete and readable.

Run this after any interrupted Stage A/B, and before Stage C. Presence is not
enough: files written through a Drive FUSE mount can exist while being
truncated or still syncing, so every archive is opened and its keys and shapes
checked against what Stage B actually writes.

    python verify_cv_artefacts.py                # all folds
    python verify_cv_artefacts.py --folds 1,2    # just these
    python verify_cv_artefacts.py --fast         # presence only

Exit code 0 if every requested fold is ready for Stage C.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from cv_pipeline import (FEAT_DIR, LOGITS_DIR, MODALITIES, N_CLASSES, STATE_DIR)
from cv_split import N_FOLDS, SUBJECTS, test_subjects

N_TRIALS = 400          # all trials per subject under the subject-wise protocol
FEAT_DIMS = {"audio": 768, "vision": 768, "eeg": 960}


def check_feature_npz(path, fast: bool) -> str | None:
    """Return an error string, or None if the archive is intact."""
    if not path.exists():
        return "missing"
    if path.stat().st_size == 0:
        return "zero bytes"
    if fast:
        return None
    expected = {m: (N_TRIALS, FEAT_DIMS[m]) for m in MODALITIES}
    expected.update({f"logits_{m}": (N_TRIALS, N_CLASSES) for m in MODALITIES})
    expected["y"] = (N_TRIALS,)
    try:
        with np.load(path, allow_pickle=False) as z:
            for key, shape in expected.items():
                if key not in z:
                    return f"missing key '{key}'"
                if z[key].shape != shape:
                    return f"{key} shape {z[key].shape}, expected {shape}"
            if not np.isfinite(z["audio"]).all():
                return "non-finite values in audio features"
    except Exception as e:
        return f"{type(e).__name__}: {e}"
    return None


def check_fold(fold: int, fast: bool) -> bool:
    print(f"\n--- fold {fold} ---")
    ok = True

    missing_states = [m for m in MODALITIES
                      if not (STATE_DIR / f"fold{fold}_{m}.pt").exists()
                      or (STATE_DIR / f"fold{fold}_{m}.pt").stat().st_size < 1024]
    if missing_states:
        print(f"  [STAGE A] INCOMPLETE - missing/empty: {missing_states}")
        ok = False
    else:
        print("  [STAGE A] OK - 3/3 encoders present")

    problems = []
    for sub in SUBJECTS:
        err = check_feature_npz(FEAT_DIR / f"fold{fold}_sub{sub:02d}.npz", fast)
        if err:
            problems.append((sub, err))

    good = len(SUBJECTS) - len(problems)
    if problems:
        print(f"  [STAGE B] INCOMPLETE - {good}/{len(SUBJECTS)} subjects intact")
        for sub, err in problems[:10]:
            print(f"              sub{sub:02d}: {err}")
        if len(problems) > 10:
            print(f"              ... and {len(problems) - 10} more")
        ok = False
    else:
        print(f"  [STAGE B] OK - {good}/{len(SUBJECTS)} subjects intact")

    scored = (LOGITS_DIR / f"fold{fold}.npz").exists()
    print(f"  [STAGE C] {'already run' if scored else 'not yet run'}"
          f"{'' if scored else '  (expected until you run --stages C)'}")

    print(f"  => fold {fold} is {'READY for Stage C' if ok else 'NOT ready'}"
          f"   (held-out subjects: {test_subjects(fold)})")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--folds", default=",".join(str(k) for k in range(N_FOLDS)))
    ap.add_argument("--fast", action="store_true",
                    help="check presence only; do not open each archive")
    args = ap.parse_args()

    folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]
    print(f"state dicts : {STATE_DIR}")
    print(f"features    : {FEAT_DIR}")
    print(f"mode        : {'fast (presence only)' if args.fast else 'full (opens every archive)'}")

    results = {f: check_fold(f, args.fast) for f in folds}

    ready = [f for f, v in results.items() if v]
    not_ready = [f for f, v in results.items() if not v]
    print("\n" + "=" * 60)
    print(f"READY for Stage C : {ready if ready else 'none'}")
    if not_ready:
        print(f"NOT ready         : {not_ready}")
        print("  Re-run those folds with --stages A,B; completed subjects are skipped.")
    print("=" * 60)
    return 0 if not not_ready else 1


if __name__ == "__main__":
    sys.exit(main())
