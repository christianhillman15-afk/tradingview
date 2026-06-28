"""AI/ML ensemble strategy.

Uses an ensemble classifier (RandomForest + GradientBoosting) over technical
features to predict the probability that price is higher ``horizon`` bars ahead.

To avoid look-ahead bias, :meth:`prepare` trains the model **only** on the
first ``train_frac`` of the data and emits signals **only** on the held-out
remainder (a single train/test split — see the docstring caveats). For
production you would use proper walk-forward retraining and realistic costs.

Signal logic:
  * P(up) > ``long_threshold``  -> long
  * P(up) < ``short_threshold`` -> short
  * otherwise flat; ATR-based protective stop.

If a pre-trained model path is provided, it is loaded and used for the whole
series instead of doing an in-run split.
"""

from __future__ import annotations

from typing import Sequence

from ..data import Bar, atr_inputs
from ..indicators import atr
from ..ml.features import build_features, build_labels
from ..ml.labeling import binary_labels, triple_barrier_labels
from ..ml.model import EnsembleModel
from .base import Signal, Strategy


class MLEnsembleStrategy(Strategy):
    name = "ml_ensemble"
    category = "ml"
    intraday = False

    def __init__(
        self,
        horizon: int = 10,
        train_frac: float = 0.6,
        long_threshold: float = 0.58,
        short_threshold: float = 0.42,
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
        model_path: str | None = None,
        labeling: str = "triple_barrier",   # "triple_barrier" | "fixed"
    ) -> None:
        super().__init__(
            horizon=horizon,
            train_frac=train_frac,
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            atr_period=atr_period,
            atr_stop_mult=atr_stop_mult,
            model_path=model_path,
            labeling=labeling,
        )
        self.horizon = horizon
        self.train_frac = train_frac
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.model_path = model_path
        self.labeling = labeling

    def _labels(self, bars):
        """Training labels: path-aware triple-barrier (default) or fixed-horizon."""
        if self.labeling == "triple_barrier":
            tb, _ = triple_barrier_labels(
                bars, pt_mult=2.0, sl_mult=2.0, max_horizon=self.horizon, atr_period=self.atr_period
            )
            return binary_labels(tb)
        return build_labels(bars, horizon=self.horizon)
        self._proba: list[float | None] = []
        self._train_end = 0

    def prepare(self, bars: Sequence[Bar]) -> None:
        super().prepare(bars)
        h, l, c = atr_inputs(bars)
        self._atr = atr(h, l, c, self.atr_period)

        rows, valid = build_features(bars)
        n = len(bars)
        self._proba = [None] * n

        if self.model_path:
            model = EnsembleModel.load(self.model_path)
            self._train_end = self.warmup()
            self._fill_proba(model, rows, valid, start=0)
            return

        # In-run split: train on the first train_frac, predict on the rest.
        self._train_end = max(int(n * self.train_frac), self.warmup())
        if n < self.warmup():
            return  # not enough data to train; stay flat
        labels = self._labels(bars)
        X_train, y_train = [], []
        for i in range(min(self._train_end, n)):
            # Only use labels whose forward window stays inside the train region.
            if valid[i] and labels[i] is not None and i + self.horizon < self._train_end:
                X_train.append(rows[i])
                y_train.append(labels[i])
        if len(X_train) < 50 or len(set(y_train)) < 2:
            # Not enough clean training data; strategy stays flat.
            return
        model = EnsembleModel().fit(X_train, y_train)
        self._fill_proba(model, rows, valid, start=self._train_end)

    def _fill_proba(self, model: EnsembleModel, rows, valid, start: int) -> None:
        idxs = [i for i in range(start, len(rows)) if valid[i]]
        if not idxs:
            return
        probs = model.predict_proba_up([rows[i] for i in idxs])
        for i, p in zip(idxs, probs):
            self._proba[i] = p

    def warmup(self) -> int:
        return max(50, self.atr_period) + 1

    def on_bar(self, i: int) -> Signal | None:
        if i < self._train_end:
            return None
        p = self._proba[i]
        a = self._atr[i]
        if p is None or a is None:
            return None
        bar = self._bars[i]
        if p > self.long_threshold:
            return Signal(
                "long",
                reason=f"P(up)={p:.2f}",
                stop=bar.close - self.atr_stop_mult * a,
                strength=min(1.0, (p - 0.5) * 2),
            )
        if p < self.short_threshold:
            return Signal(
                "short",
                reason=f"P(up)={p:.2f}",
                stop=bar.close + self.atr_stop_mult * a,
                strength=min(1.0, (0.5 - p) * 2),
            )
        return None
