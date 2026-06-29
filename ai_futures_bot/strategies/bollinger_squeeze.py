"""TTM-style squeeze breakout (Bollinger Bands inside Keltner Channels).

Rules (researched TTM Squeeze playbook, John Carter):
  * A "squeeze" is ON when the Bollinger Bands contract *inside* the Keltner
    Channels — volatility is coiled and a directional move is likely.
  * When the squeeze "fires" (BB expand back outside KC after being inside),
    enter in the direction of momentum (rate-of-change sign).
  * Stop at the opposite extreme of the squeeze range; ATR trailing stop rides
    the breakout. ``category = "breakout"``.

Defaults: BB(20, 2.0), KC(20, 1.5×ATR), momentum via 12-bar ROC.
"""

from __future__ import annotations

from typing import Sequence

from ..data import Bar, atr_inputs, closes
from ..indicators import bollinger_bands, keltner_channels, roc
from .base import Signal, Strategy


class BollingerSqueezeStrategy(Strategy):
    name = "bollinger_squeeze"
    category = "breakout"
    intraday = False

    def __init__(
        self,
        bb_period: int = 20,
        bb_std: float = 2.0,
        kc_period: int = 20,
        kc_atr_period: int = 10,
        kc_mult: float = 1.5,
        roc_period: int = 12,
        trail_atr_mult: float = 2.5,
    ) -> None:
        super().__init__(
            bb_period=bb_period,
            bb_std=bb_std,
            kc_period=kc_period,
            kc_atr_period=kc_atr_period,
            kc_mult=kc_mult,
            roc_period=roc_period,
            trail_atr_mult=trail_atr_mult,
        )
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.kc_period = kc_period
        self.kc_atr_period = kc_atr_period
        self.kc_mult = kc_mult
        self.roc_period = roc_period
        self.trail_atr_mult = trail_atr_mult

    def prepare(self, bars: Sequence[Bar]) -> None:
        super().prepare(bars)
        c = closes(bars)
        h, l, cc = atr_inputs(bars)
        self._bb_u, _, self._bb_l = bollinger_bands(c, self.bb_period, self.bb_std)
        self._kc_u, _, self._kc_l = keltner_channels(
            h, l, cc, self.kc_period, self.kc_atr_period, self.kc_mult
        )
        self._roc = roc(c, self.roc_period)
        self._squeeze_on = False
        self._range_hi: float | None = None
        self._range_lo: float | None = None

    def reset(self) -> None:
        super().reset()
        self._squeeze_on = False
        self._range_hi = None
        self._range_lo = None

    def warmup(self) -> int:
        return max(self.bb_period, self.kc_period, self.roc_period) + 1

    def _in_squeeze(self, i: int) -> bool | None:
        bu, bl = self._bb_u[i], self._bb_l[i]
        ku, kl = self._kc_u[i], self._kc_l[i]
        if None in (bu, bl, ku, kl):
            return None
        return bu < ku and bl > kl

    def on_bar(self, i: int) -> Signal | None:
        if i < self.warmup():
            return None
        bar = self._bars[i]
        sq = self._in_squeeze(i)
        if sq is None:
            return None

        if sq:
            # Accumulate the squeeze range to anchor the stop.
            self._squeeze_on = True
            self._range_hi = bar.high if self._range_hi is None else max(self._range_hi, bar.high)
            self._range_lo = bar.low if self._range_lo is None else min(self._range_lo, bar.low)
            return None

        # Squeeze just released -> fire in the momentum direction.
        if self._squeeze_on:
            self._squeeze_on = False
            momentum = self._roc[i]
            hi, lo = self._range_hi, self._range_lo
            self._range_hi = self._range_lo = None
            if momentum is None or hi is None or lo is None:
                return None
            if momentum > 0:
                return Signal(
                    "long",
                    reason=f"squeeze fired up (ROC {momentum:.2f})",
                    stop=lo,
                    trail_atr_mult=self.trail_atr_mult,
                )
            if momentum < 0:
                return Signal(
                    "short",
                    reason=f"squeeze fired down (ROC {momentum:.2f})",
                    stop=hi,
                    trail_atr_mult=self.trail_atr_mult,
                )
        return None
