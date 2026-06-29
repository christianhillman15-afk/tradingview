"""Z-score statistical mean-reversion.

Rules:
  * Compute the rolling z-score of price vs its ``period`` mean.
  * Long when z <= -``entry_z`` (stretched cheap), short when z >= ``entry_z``.
  * Exit when price reverts to the mean (z crosses back toward 0) or the
    ``exit_z`` band, or an ATR stop is hit, or a time-stop expires.

A cleaner statistical cousin of the Bollinger strategy — it fades extremes
measured in standard deviations. ``category = "reversion"``.
"""

from __future__ import annotations

from typing import Sequence

from ..data import Bar, atr_inputs, closes
from ..indicators import atr, sma, zscore
from .base import Signal, Strategy


class ZScoreReversionStrategy(Strategy):
    name = "zscore_reversion"
    category = "reversion"
    intraday = False

    def __init__(
        self,
        period: int = 20,
        entry_z: float = 2.0,
        exit_z: float = 0.3,
        atr_period: int = 14,
        atr_stop_mult: float = 1.5,
        max_hold: int = 30,
    ) -> None:
        super().__init__(
            period=period,
            entry_z=entry_z,
            exit_z=exit_z,
            atr_period=atr_period,
            atr_stop_mult=atr_stop_mult,
            max_hold=max_hold,
        )
        self.period = period
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.max_hold = max_hold

    def prepare(self, bars: Sequence[Bar]) -> None:
        super().prepare(bars)
        c = closes(bars)
        h, l, cc = atr_inputs(bars)
        self._z = zscore(c, self.period)
        self._mean = sma(c, self.period)
        self._atr = atr(h, l, cc, self.atr_period)
        self._side = 0
        self._held = 0

    def reset(self) -> None:
        super().reset()
        self._side = 0
        self._held = 0

    def warmup(self) -> int:
        return max(self.period, self.atr_period) + 1

    def on_bar(self, i: int) -> Signal | None:
        if i < self.warmup():
            return None
        z = self._z[i]
        mean = self._mean[i]
        a = self._atr[i]
        if z is None or mean is None or a is None:
            return None
        bar = self._bars[i]

        if self._side != 0:
            self._held += 1
            reverted = abs(z) <= self.exit_z
            timed_out = self._held >= self.max_hold
            if reverted or timed_out:
                self._side = 0
                self._held = 0
                return Signal("exit", reason="reverted to mean" if reverted else "time-stop")
            return None

        if z <= -self.entry_z:
            self._side = 1
            self._held = 0
            return Signal(
                "long",
                reason=f"z={z:.2f} (cheap)",
                stop=bar.close - self.atr_stop_mult * a,
                target=mean,
            )
        if z >= self.entry_z:
            self._side = -1
            self._held = 0
            return Signal(
                "short",
                reason=f"z={z:.2f} (rich)",
                stop=bar.close + self.atr_stop_mult * a,
                target=mean,
            )
        return None
