"""VWAP mean-reversion — fade intraday stretches back to fair value.

Rules (researched VWAP playbook):
  * VWAP is the intraday "fair value" anchor and resets each session.
  * Long when price is stretched ``extension_mult`` * ATR *below* VWAP and
    RSI < ``rsi_oversold`` (default 25); target is VWAP.
  * Short when stretched above VWAP with RSI > ``rsi_overbought`` (default 75).
  * Most reliable in the first 90 and last 60 minutes; the optional
    ``avoid_midday`` filter skips the 11:00-14:00 ET lull.
  * Protective stop ``atr_stop_mult`` * ATR beyond entry.

``intraday`` is True (flatten at session end).
"""

from __future__ import annotations

from typing import Sequence

from ..data import Bar, atr_inputs, closes, session_ids, volumes
from ..indicators import atr, rsi, session_vwap
from .base import Signal, Strategy


class VwapReversionStrategy(Strategy):
    name = "vwap_reversion"
    category = "reversion"
    intraday = True

    def __init__(
        self,
        rsi_period: int = 14,
        rsi_oversold: float = 25.0,
        rsi_overbought: float = 75.0,
        atr_period: int = 14,
        extension_mult: float = 1.5,
        atr_stop_mult: float = 1.0,
        bars_per_day: int = 390,
        avoid_midday: bool = True,
    ) -> None:
        super().__init__(
            rsi_period=rsi_period,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            atr_period=atr_period,
            extension_mult=extension_mult,
            atr_stop_mult=atr_stop_mult,
            bars_per_day=bars_per_day,
            avoid_midday=avoid_midday,
        )
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.atr_period = atr_period
        self.extension_mult = extension_mult
        self.atr_stop_mult = atr_stop_mult
        self.bars_per_day = bars_per_day
        self.avoid_midday = avoid_midday

    def prepare(self, bars: Sequence[Bar]) -> None:
        super().prepare(bars)
        h, l, c = atr_inputs(bars)
        v = volumes(bars)
        sids = session_ids(bars)
        self._vwap = session_vwap(h, l, c, v, sids)
        self._rsi = rsi(c, self.rsi_period)
        self._atr = atr(h, l, c, self.atr_period)

        # Minute-in-session index for the time-of-day filter.
        n = len(bars)
        self._minute = [0] * n
        current: str | None = None
        idx = 0
        for i in range(n):
            if sids[i] != current:
                current = sids[i]
                idx = 0
            else:
                idx += 1
            self._minute[i] = idx
        self._side = 0

    def reset(self) -> None:
        super().reset()
        self._side = 0

    def warmup(self) -> int:
        return max(self.rsi_period, self.atr_period) + 1

    def _in_active_window(self, minute: int) -> bool:
        if not self.avoid_midday:
            return True
        first_90 = minute <= 90
        last_60 = minute >= self.bars_per_day - 60
        return first_90 or last_60

    def on_bar(self, i: int) -> Signal | None:
        if i < self.warmup():
            return None
        bar = self._bars[i]
        vwap = self._vwap[i]
        r = self._rsi[i]
        a = self._atr[i]
        if None in (vwap, r, a):
            return None
        minute = self._minute[i]

        # Manage an open position: exit at VWAP (the mean) or at session window end.
        if self._side > 0 and bar.high >= vwap:
            self._side = 0
            return Signal("exit", reason="reverted to VWAP (long)")
        if self._side < 0 and bar.low <= vwap:
            self._side = 0
            return Signal("exit", reason="reverted to VWAP (short)")

        if self._side != 0 or not self._in_active_window(minute):
            return None

        stretch = bar.close - vwap
        threshold = self.extension_mult * a
        if stretch <= -threshold and r < self.rsi_oversold:
            self._side = 1
            return Signal(
                "long",
                reason=f"{abs(stretch):.1f} below VWAP, RSI={r:.0f}",
                stop=bar.close - self.atr_stop_mult * a,
                target=vwap,
            )
        if stretch >= threshold and r > self.rsi_overbought:
            self._side = -1
            return Signal(
                "short",
                reason=f"{stretch:.1f} above VWAP, RSI={r:.0f}",
                stop=bar.close + self.atr_stop_mult * a,
                target=vwap,
            )
        return None
