"""Supertrend trend-following with an ADX strength filter.

Rules (researched Supertrend playbook):
  * Supertrend(period, mult) flips its line from below price (uptrend) to above
    price (downtrend). Enter on the flip.
  * Only take flips when ADX >= ``adx_min`` — in chop (low ADX) every flip loses.
  * The Supertrend line is the stop; additionally an ATR trailing stop ratchets
    the exit. Opposite flip reverses the position.

The most common professional use of Supertrend is as a *trailing stop*, which is
exactly how this strategy treats it. ``category = "trend"``.
"""

from __future__ import annotations

from typing import Sequence

from ..data import Bar, atr_inputs
from ..indicators import adx, supertrend
from .base import Signal, Strategy


class SupertrendStrategy(Strategy):
    name = "supertrend"
    category = "trend"
    intraday = False

    def __init__(
        self,
        period: int = 10,
        mult: float = 3.0,
        adx_period: int = 14,
        adx_min: float = 20.0,
        trail_atr_mult: float = 3.0,
    ) -> None:
        super().__init__(
            period=period,
            mult=mult,
            adx_period=adx_period,
            adx_min=adx_min,
            trail_atr_mult=trail_atr_mult,
        )
        self.period = period
        self.mult = mult
        self.adx_period = adx_period
        self.adx_min = adx_min
        self.trail_atr_mult = trail_atr_mult

    def prepare(self, bars: Sequence[Bar]) -> None:
        super().prepare(bars)
        h, l, c = atr_inputs(bars)
        self._st, self._dir = supertrend(h, l, c, self.period, self.mult)
        self._adx, _, _ = adx(h, l, c, self.adx_period)

    def warmup(self) -> int:
        return max(self.period, self.adx_period) * 2 + 1

    def on_bar(self, i: int) -> Signal | None:
        if i < self.warmup():
            return None
        d_now = self._dir[i]
        d_prev = self._dir[i - 1]
        line = self._st[i]
        adx_val = self._adx[i]
        if d_now is None or d_prev is None or line is None or adx_val is None:
            return None
        if d_now == d_prev:
            return None  # only act on a flip
        if adx_val < self.adx_min:
            return None  # weak/no trend — skip the flip
        if d_now == 1:
            return Signal(
                "long",
                reason=f"Supertrend flip up (ADX {adx_val:.0f})",
                stop=line,
                trail_atr_mult=self.trail_atr_mult,
            )
        return Signal(
            "short",
            reason=f"Supertrend flip down (ADX {adx_val:.0f})",
            stop=line,
            trail_atr_mult=self.trail_atr_mult,
        )
