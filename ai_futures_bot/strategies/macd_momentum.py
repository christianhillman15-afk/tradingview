"""MACD momentum strategy with a trend filter.

Rules:
  * Long when the MACD line crosses above its signal line, *and* price is above
    the long-term EMA trend filter (default 200).
  * Short on the opposite cross below signal while price is below the EMA.
  * Exit when MACD crosses back through the signal line.
  * ATR-based protective stop.

The trend filter is the key refinement: raw MACD crosses whipsaw badly in
ranges, so we only take crosses aligned with the higher-timeframe trend.
"""

from __future__ import annotations

from typing import Sequence

from ..data import Bar, atr_inputs, closes
from ..indicators import atr, crossed_above, crossed_below, ema, macd
from .base import Signal, Strategy


class MacdMomentumStrategy(Strategy):
    name = "macd_momentum"
    category = "trend"
    intraday = False

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        trend_period: int = 200,
        atr_period: int = 14,
        atr_stop_mult: float = 2.5,
    ) -> None:
        super().__init__(
            fast=fast,
            slow=slow,
            signal=signal,
            trend_period=trend_period,
            atr_period=atr_period,
            atr_stop_mult=atr_stop_mult,
        )
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.trend_period = trend_period
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult

    def prepare(self, bars: Sequence[Bar]) -> None:
        super().prepare(bars)
        c = closes(bars)
        self._macd, self._sig, _ = macd(c, self.fast, self.slow, self.signal)
        self._trend = ema(c, self.trend_period)
        h, l, cc = atr_inputs(bars)
        self._atr = atr(h, l, cc, self.atr_period)

    def warmup(self) -> int:
        return max(self.slow + self.signal, self.trend_period, self.atr_period) + 1

    def on_bar(self, i: int) -> Signal | None:
        if i < self.warmup():
            return None
        bar = self._bars[i]
        trend = self._trend[i]
        a = self._atr[i]
        if trend is None or a is None:
            return None

        up = crossed_above(self._macd, self._sig, i)
        down = crossed_below(self._macd, self._sig, i)
        above_trend = bar.close > trend

        if up and above_trend:
            return Signal(
                "long",
                reason="MACD cross up in uptrend",
                stop=bar.close - self.atr_stop_mult * a,
            )
        if down and not above_trend:
            return Signal(
                "short",
                reason="MACD cross down in downtrend",
                stop=bar.close + self.atr_stop_mult * a,
            )
        if up or down:
            # A cross against the position is our exit trigger.
            return Signal("exit", reason="MACD cross against position")
        return None
