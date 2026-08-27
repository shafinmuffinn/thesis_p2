"""Deterministic subject-wise fold assignment for cross-subject evaluation.

Why this module exists
----------------------
Every accuracy number in the P2 thesis is *within-subject*: the model trains
and tests on the same person. Limitation 2 of Chapter 8 acknowledges that this
cannot support any generalisation claim. This module defines the subject-wise
partition used to retire that limitation.

The unit of partition is the SUBJECT, never the trial. A subject appears in
exactly one test fold and in the training set of every other fold. Because each
EAV subject contributes a class-balanced 400 trials (80 per emotion), no
stratification is needed at the subject level -- any partition of subjects is
automatically class-balanced.

The leakage rule
----------------
For fold k, every model in the pipeline -- the three per-modality encoders AND
the fusion head -- must be fit using only subjects in `train_subjects(k)`. The
Day-5 caches under `day5_state_dicts/` and `day5_features/` CANNOT be reused
here: each of those encoders was fine-tuned on its own subject, so a cached
feature for a held-out subject was produced by a model that had already seen
that subject's data. Reusing them would leak the test subject into the encoder
and inflate the cross-subject number.

Usage
-----
    from cv_split import make_folds, train_subjects, test_subjects

    for k in range(5):
        tr, te = train_subjects(k), test_subjects(k)
"""
from __future__ import annotations

import numpy as np

SUBJECTS = list(range(1, 43))   # 42 EAV subjects
N_FOLDS = 5
SEED = 42


def make_folds(n_folds: int = N_FOLDS, seed: int = SEED) -> list[list[int]]:
    """Partition subjects into `n_folds` disjoint test groups.

    Deterministic for a given (n_folds, seed): the same assignment is produced
    on every machine and in every session, so the three team members running
    different folds in parallel are guaranteed to agree on the split.

    With 42 subjects and 5 folds the sizes are 9, 9, 8, 8, 8.
    """
    rng = np.random.default_rng(seed)
    shuffled = np.array(SUBJECTS)
    rng.shuffle(shuffled)
    return [sorted(int(s) for s in group)
            for group in np.array_split(shuffled, n_folds)]


def test_subjects(fold: int, n_folds: int = N_FOLDS, seed: int = SEED) -> list[int]:
    """Subjects held out for testing in `fold`."""
    folds = make_folds(n_folds, seed)
    if not 0 <= fold < len(folds):
        raise ValueError(f"fold must be in [0, {len(folds)}), got {fold}")
    return folds[fold]


def train_subjects(fold: int, n_folds: int = N_FOLDS, seed: int = SEED) -> list[int]:
    """Subjects used to fit every model in `fold` -- encoders and fusion head."""
    held_out = set(test_subjects(fold, n_folds, seed))
    return [s for s in SUBJECTS if s not in held_out]


def fold_of(subject: int, n_folds: int = N_FOLDS, seed: int = SEED) -> int:
    """The fold in which `subject` is held out. Inverse of test_subjects()."""
    for k, group in enumerate(make_folds(n_folds, seed)):
        if subject in group:
            return k
    raise ValueError(f"subject {subject} is not in SUBJECTS")


def validate(n_folds: int = N_FOLDS, seed: int = SEED) -> None:
    """Assert the partition is a genuine partition. Raises on violation."""
    folds = make_folds(n_folds, seed)
    flat = [s for group in folds for s in group]

    assert len(flat) == len(SUBJECTS), \
        f"partition covers {len(flat)} subjects, expected {len(SUBJECTS)}"
    assert len(set(flat)) == len(flat), "a subject appears in more than one fold"
    assert set(flat) == set(SUBJECTS), "partition does not cover every subject"

    for k in range(n_folds):
        tr, te = set(train_subjects(k, n_folds, seed)), set(test_subjects(k, n_folds, seed))
        assert not (tr & te), f"fold {k}: train and test subjects overlap"
        assert tr | te == set(SUBJECTS), f"fold {k}: train + test != all subjects"

    # Every subject is tested exactly once across the whole CV run, which is
    # what makes the 42 per-subject accuracies directly pairable against the
    # 42 within-subject accuracies from P2 in a Wilcoxon signed-rank test.
    for s in SUBJECTS:
        assert sum(s in test_subjects(k, n_folds, seed) for k in range(n_folds)) == 1, \
            f"subject {s} is not tested exactly once"


if __name__ == "__main__":
    validate()
    folds = make_folds()
    print(f"{len(SUBJECTS)} subjects -> {len(folds)} folds  (seed={SEED})\n")
    for k, group in enumerate(folds):
        tr = train_subjects(k)
        print(f"  fold {k}:  test = {len(group):2d} subjects  {group}")
        print(f"            train = {len(tr):2d} subjects  "
              f"({len(tr) * 400:,} trials)")
    print("\nvalidate(): PASS -- disjoint, exhaustive, each subject tested once.")
