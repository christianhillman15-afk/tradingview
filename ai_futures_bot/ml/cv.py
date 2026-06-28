"""Purged & embargoed K-fold cross-validation (López de Prado).

Standard K-fold leaks in finance: a training label that spans bars overlapping
the test set lets the model "see the future". Purged CV removes any training
sample whose label window (``[i, t1[i]]``) overlaps the test fold, and an
*embargo* drops a few samples immediately after the test fold to kill residual
serial-correlation leakage. The result is an honest out-of-sample score.

The splitter is pure stdlib (index logic); ``purged_cv_score`` lazily imports
scikit-learn to fit/evaluate a model per fold.
"""

from __future__ import annotations

from typing import Iterator, Sequence


class PurgedKFold:
    """K-fold splitter with purging and embargo, working in bar-index space."""

    def __init__(self, n_splits: int = 5, embargo: float = 0.01) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        self.n_splits = n_splits
        self.embargo = embargo

    def split(self, n: int, t1: Sequence[int]) -> Iterator[tuple[list[int], list[int]]]:
        """Yield ``(train_idx, test_idx)`` over ``0..n-1``. ``t1[i]`` is the index
        at which sample ``i``'s label resolves."""
        embargo_n = int(round(n * self.embargo))
        fold = n // self.n_splits
        for k in range(self.n_splits):
            test_start = k * fold
            test_end = n if k == self.n_splits - 1 else (k + 1) * fold
            test_idx = list(range(test_start, test_end))
            train_idx: list[int] = []
            for j in range(n):
                if test_start <= j < test_end:
                    continue  # belongs to the test fold
                # Purge: drop training samples whose label window overlaps the test fold.
                j_start, j_end = j, t1[j]
                overlaps = not (j_end < test_start or j_start >= test_end)
                # Embargo: drop training samples just after the test fold.
                in_embargo = test_end <= j < test_end + embargo_n
                if overlaps or in_embargo:
                    continue
                train_idx.append(j)
            yield train_idx, test_idx


def _require_sklearn():
    try:
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:  # pragma: no cover
        from .model import MLNotInstalledError

        raise MLNotInstalledError(
            "purged_cv_score needs numpy and scikit-learn (pip install numpy scikit-learn)."
        ) from exc
    return np, RandomForestClassifier


def purged_cv_score(
    feature_rows: Sequence[Sequence[float]],
    labels: Sequence[int | None],
    valid: Sequence[bool],
    t1: Sequence[int],
    *,
    n_splits: int = 5,
    embargo: float = 0.01,
) -> float | None:
    """Mean out-of-sample accuracy of a RandomForest under purged+embargoed CV.

    Returns ``None`` if there is not enough clean, two-class data.
    """
    np, RFC = _require_sklearn()
    n = len(feature_rows)
    usable = {i for i in range(n) if valid[i] and labels[i] is not None}
    if len(usable) < n_splits * 10:
        return None
    kf = PurgedKFold(n_splits=n_splits, embargo=embargo)
    accuracies: list[float] = []
    for train_idx, test_idx in kf.split(n, t1):
        tr = [i for i in train_idx if i in usable]
        te = [i for i in test_idx if i in usable]
        ytr = [labels[i] for i in tr]
        if len(tr) < 20 or len(te) < 5 or len(set(ytr)) < 2:
            continue
        clf = RFC(n_estimators=100, max_depth=5, min_samples_leaf=20, random_state=42, n_jobs=-1)
        clf.fit(np.asarray([feature_rows[i] for i in tr], dtype=float), np.asarray(ytr, dtype=int))
        pred = clf.predict(np.asarray([feature_rows[i] for i in te], dtype=float))
        yte = np.asarray([labels[i] for i in te], dtype=int)
        accuracies.append(float((pred == yte).mean()))
    if not accuracies:
        return None
    return sum(accuracies) / len(accuracies)
