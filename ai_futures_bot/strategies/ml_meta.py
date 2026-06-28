"""Meta-labeling — let a secondary ML model decide *whether* to take the
primary strategy's trades (López de Prado, AFML ch. 3).

The idea: a primary rule-based strategy decides the *side* (long/short); a
secondary "meta" model, trained on the **triple-barrier outcomes** of the
primary's historical signals, predicts the *probability that taking the trade is
correct*. You only take signals the meta-model approves, and size by its
confidence. This typically raises precision and Sharpe by filtering the primary's
low-quality signals — without ever flipping its direction.

To avoid look-ahead the meta-model is trained only on signals in the first
``train_frac`` of the data and applied to the remainder. If there isn't enough
labelled data to train, it falls back to taking every primary signal (i.e.
behaves like the primary strategy).
"""

from __future__ import annotations

from typing import Sequence

from ..data import Bar
from ..ml.features import build_features
from ..ml.labeling import triple_barrier_labels
from ..ml.model import EnsembleModel
from .base import Signal, Strategy


class MetaLabelStrategy(Strategy):
    name = "ml_meta"
    category = "ml"
    intraday = False

    def __init__(
        self,
        primary: str = "supertrend",
        primary_params: dict | None = None,
        horizon: int = 20,
        pt_mult: float = 2.0,
        sl_mult: float = 2.0,
        train_frac: float = 0.6,
        take_threshold: float = 0.5,
        size_by_confidence: bool = True,
    ) -> None:
        super().__init__(
            primary=primary,
            primary_params=primary_params,
            horizon=horizon,
            pt_mult=pt_mult,
            sl_mult=sl_mult,
            train_frac=train_frac,
            take_threshold=take_threshold,
            size_by_confidence=size_by_confidence,
        )
        self.primary_name = primary
        self.primary_params = primary_params or {}
        self.horizon = horizon
        self.pt_mult = pt_mult
        self.sl_mult = sl_mult
        self.train_frac = train_frac
        self.take_threshold = take_threshold
        self.size_by_confidence = size_by_confidence
        self._decisions: dict[int, Signal] = {}
        self._train_end = 0

    def prepare(self, bars: Sequence[Bar]) -> None:
        super().prepare(bars)
        from . import get_strategy  # lazy import to dodge the registry cycle

        n = len(bars)
        self._decisions = {}
        self._train_end = max(int(n * self.train_frac), 60)

        # 1) Collect the primary strategy's directional signals.
        primary = get_strategy(self.primary_name, **self.primary_params)
        primary.prepare(bars)
        events: list[tuple[int, Signal]] = []
        exits: list[tuple[int, Signal]] = []
        for i in range(n):
            sig = primary.on_bar(i)
            if sig is None:
                continue
            if sig.action in ("long", "short"):
                events.append((i, sig))
            elif sig.action == "exit":
                exits.append((i, sig))
        # Always honour the primary's risk exits in the live region (no filtering).
        for i, sig in exits:
            if i >= self._train_end:
                self._decisions[i] = sig
        if not events:
            return

        # 2) Features and triple-barrier outcomes (the meta-label).
        rows, valid = build_features(bars)
        tb, _ = triple_barrier_labels(
            bars, pt_mult=self.pt_mult, sl_mult=self.sl_mult,
            max_horizon=self.horizon, atr_period=14,
        )

        X_train, y_train, test_events = [], [], []
        for i, sig in events:
            if not valid[i] or tb[i] is None:
                continue
            # Meta-label = was the primary's side the winning side at the barrier?
            won = 1 if ((sig.action == "long" and tb[i] == 1) or
                        (sig.action == "short" and tb[i] == -1)) else 0
            if i < self._train_end:
                X_train.append(rows[i])
                y_train.append(won)
            else:
                test_events.append((i, sig))

        # 3) Train the meta-model; fall back to "take everything" if we can't.
        if len(X_train) >= 30 and len(set(y_train)) == 2:
            model = EnsembleModel().fit(X_train, y_train)
            probs = model.predict_proba_up([rows[i] for i, _ in test_events]) if test_events else []
            for (i, sig), p in zip(test_events, probs):
                if p >= self.take_threshold:
                    self._decisions[i] = self._decorate(sig, p)
        else:
            for i, sig in test_events:
                self._decisions[i] = self._decorate(sig, 1.0)

    def _decorate(self, sig: Signal, prob: float) -> Signal:
        strength = max(0.0, min(1.0, (prob - 0.5) * 2)) if self.size_by_confidence else 1.0
        return Signal(
            action=sig.action,
            reason=f"{self.primary_name} signal, meta P(take)={prob:.2f}",
            stop=sig.stop,
            target=sig.target,
            strength=strength if self.size_by_confidence else 1.0,
            trail_atr_mult=sig.trail_atr_mult,
        )

    def warmup(self) -> int:
        return max(self._train_end, 60)

    def on_bar(self, i: int) -> Signal | None:
        return self._decisions.get(i)
