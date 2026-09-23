"""Are the three modalities' trials in the same order, participant by participant?

The suppression matrix and every fusion model compare audio trial i, vision
trial i and EEG trial i as if they were the same interaction. P2 supported this
only with per-class COUNTS (heuristic_alignment_check.py), which is a necessary
condition but weak: any ordering with 80 trials per class passes it.

This checks the stronger condition -- the label SEQUENCES match element-wise --
and, crucially, reports whether that check is informative. If each pickle's
labels are sorted into class blocks, sequences will match trivially and the
check says nothing about whether trial i is the same interaction in each
modality. The number of label changes along the sequence tells the two apart:
~4 changes means class blocks (uninformative); many changes means an
interleaved order, where an element-wise match is genuine evidence.

    python label_sequence_check.py
"""
from __future__ import annotations

import pickle
import sys

import numpy as np

from paths import EAV_PICKLES

FOLDER = {"audio": "Audio", "vision": "Vision", "eeg": "EEG"}
SUFFIX = {"audio": "aud", "vision": "vis", "eeg": "eeg"}


def labels(sub: int, modality: str) -> tuple[np.ndarray, np.ndarray]:
    path = EAV_PICKLES / FOLDER[modality] / f"subject_{sub:02d}_{SUFFIX[modality]}.pkl"
    with open(path, "rb") as f:
        _, tr_y, _, te_y = pickle.load(f)
    fix = lambda y: (np.asarray(y).argmax(axis=1) if np.asarray(y).ndim == 2
                     else np.asarray(y).astype(int))
    return fix(tr_y), fix(te_y)


def main() -> int:
    print(f"{'sub':>4s}  {'split':5s}  {'A=V':>4s}  {'A=E':>4s}  {'changes':>8s}  order")
    all_match, informative, n = 0, 0, 0
    for sub in range(1, 43):
        seqs = {m: labels(sub, m) for m in FOLDER}
        for split, i in (("train", 0), ("test", 1)):
            a, v, e = seqs["audio"][i], seqs["vision"][i], seqs["eeg"][i]
            av = len(a) == len(v) and np.array_equal(a, v)
            ae = len(a) == len(e) and np.array_equal(a, e)
            changes = int((np.diff(a) != 0).sum())
            blocked = changes <= 5
            order = "class-blocked (match uninformative)" if blocked else "interleaved"
            print(f"{sub:4d}  {split:5s}  {'yes' if av else 'NO':>4s}  "
                  f"{'yes' if ae else 'NO':>4s}  {changes:8d}  {order}")
            n += 1
            all_match += av and ae
            informative += (av and ae) and not blocked
    print(f"\n{all_match}/{n} splits match element-wise across all three modalities")
    print(f"{informative}/{n} of those are interleaved, i.e. the match is real evidence")
    if informative < all_match:
        print("Class-blocked splits: an element-wise match is guaranteed by the "
              "blocking and says nothing about trial correspondence. Report "
              "alignment for those as unverified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
