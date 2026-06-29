"""Ensemble classifier for directional prediction.

Wraps a soft-voting ensemble of a RandomForest and a GradientBoosting
classifier (scikit-learn) behind a tiny interface so the strategy code does
not depend on sklearn directly. Imports are lazy and raise a clear, actionable
error if the ML extras are not installed.
"""

from __future__ import annotations

import pickle
from typing import Sequence


class MLNotInstalledError(RuntimeError):
    """Raised when the ML strategy is used without numpy/scikit-learn."""


def _require_sklearn():
    try:
        import numpy as np  # noqa: F401
        from sklearn.ensemble import (  # noqa: F401
            GradientBoostingClassifier,
            RandomForestClassifier,
            VotingClassifier,
        )
        from sklearn.preprocessing import StandardScaler  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise MLNotInstalledError(
            "The ML strategy needs numpy and scikit-learn. Install with:\n"
            "    pip install numpy scikit-learn\n"
            "or use any of the rule-based strategies, which need no extras."
        ) from exc
    return np


class EnsembleModel:
    """Soft-voting ensemble with feature standardisation."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self._pipeline = None
        self._trained = False

    def _build(self):
        from sklearn.ensemble import (
            GradientBoostingClassifier,
            RandomForestClassifier,
            VotingClassifier,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        ensemble = VotingClassifier(
            estimators=[
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=6,
                        min_samples_leaf=20,
                        random_state=self.random_state,
                        n_jobs=-1,
                    ),
                ),
                (
                    "gb",
                    GradientBoostingClassifier(
                        n_estimators=150,
                        max_depth=3,
                        learning_rate=0.05,
                        random_state=self.random_state,
                    ),
                ),
            ],
            voting="soft",
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", ensemble)])

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[int]) -> "EnsembleModel":
        np = _require_sklearn()
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        if len(set(y_arr.tolist())) < 2:
            raise ValueError("Training labels must contain both classes (up and down).")
        self._pipeline = self._build()
        self._pipeline.fit(X_arr, y_arr)
        self._trained = True
        return self

    def predict_proba_up(self, X: Sequence[Sequence[float]]) -> list[float]:
        """Probability of the 'up' class for each row."""
        if not self._trained or self._pipeline is None:
            raise RuntimeError("Model is not trained.")
        np = _require_sklearn()
        X_arr = np.asarray(X, dtype=float)
        proba = self._pipeline.predict_proba(X_arr)
        classes = list(self._pipeline.classes_)
        up_idx = classes.index(1) if 1 in classes else 1
        return proba[:, up_idx].tolist()

    @property
    def trained(self) -> bool:
        return self._trained

    def save(self, path: str) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self._pipeline, fh)

    @classmethod
    def load(cls, path: str) -> "EnsembleModel":
        model = cls()
        with open(path, "rb") as fh:
            model._pipeline = pickle.load(fh)
        model._trained = True
        return model
