"""Time-series (absolute) momentum — the most robustly documented futures edge.

Moskowitz, Ooi & Pedersen (2012) showed that the sign of an instrument's own
past ~12-month return predicts its next-period return, positively across 58
liquid futures (52 of 58 significant), with continuation for ~1 year. This is
the canonical trend-following signal and the backbone of managed-futures (CTA)
programs.

Rules implemented:
  * Signal = sign(close_now − close_{lookback bars ago}). Long if the past
    return is positive, short if negative; flip on a sign change.
  * Optional ``skip`` of the most recent bars (the literature often skips the
    last month to sidestep short-term reversal).
  * ATR-based protective stop + ATR trailing stop.

NOTE on ``lookback``: it is in *bars*, so set it to match your timeframe — ~252
for daily bars, ~12 for monthly. The default (120) suits hourly/4h swing data;
the research's "12 months" is the intent, not a fixed bar count.
``category = "trend"``.
"""

from __future__ import annotations

from typing import Sequence

from ..data import Bar, atr_inputs, closes
from ..indicators import atr
from .base import Signal, Strategy


class TimeSeriesMomentumStrategy(Strategy):
    name = "tsmom"
    category = "trend"
    intraday = False

    def __init__(
        self,
        lookback: int = 120,
        skip: int = 0,
        atr_period: int = 14,
        atr_stop_mult: float = 3.0,
        trail_atr_mult: float = 4.0,
    ) -> None:
        super().__init__(
            lookback=lookback,
            skip=skip,
            atr_period=atr_period,
            atr_stop_mult=atr_stop_mult,
            trail_atr_mult=trail_atr_mult,
        )
        self.lookback = lookback
        self.skip = skip
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.trail_atr_mult = trail_atr_mult

    def prepare(self, bars: Sequence[Bar]) -> None:
        super().prepare(bars)
        self._close = closes(bars)
        h, l, c = atr_inputs(bars)
        self._atr = atr(h, l, c, self.atr_period)
        self._side = 0

    def reset(self) -> None:
        super().reset()
        self._side = 0

    def warmup(self) -> int:
        return self.lookback + self.skip + self.atr_period + 1

    def on_bar(self, i: int) -> Signal | None:
        if i < self.warmup():
            return None
        a = self._atr[i]
        if a is None:
            return None
        now = self._close[i - self.skip]
        past = self._close[i - self.skip - self.lookback]
        mom = now - past
        desired = 1 if mom > 0 else (-1 if mom < 0 else 0)
        if desired == 0 or desired == self._side:
            return None
        self._side = desired
        bar = self._bars[i]
        ret_pct = (mom / past * 100.0) if past else 0.0
        if desired > 0:
            return Signal(
                "long",
                reason=f"12m-style momentum +{ret_pct:.1f}%",
                stop=bar.close - self.atr_stop_mult * a,
                trail_atr_mult=self.trail_atr_mult,
            )
        return Signal(
            "short",
            reason=f"12m-style momentum {ret_pct:.1f}%",
            stop=bar.close + self.atr_stop_mult * a,
            trail_atr_mult=self.trail_atr_mult,
        )
