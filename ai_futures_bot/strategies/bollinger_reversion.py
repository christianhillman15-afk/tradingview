"""Bollinger Band + RSI mean-reversion strategy.

Rules (researched mean-reversion playbook):
  * Long when price closes below the lower band *and* RSI < oversold (default 30).
  * Short when price closes above the upper band *and* RSI > overbought (70).
  * Target is the middle band (the mean); the trade thesis is exhausted there.
  * Protective stop 0.5*ATR beyond the band that was pierced.
  * Time-stop: exit if the move has not begun reverting within ``max_hold`` bars.

Mean reversion fades extremes, so it is the natural complement to the breakout
and momentum strategies — they tend to win in different regimes.
"""

from __future__ import annotations

from typing import Sequence

from ..data import Bar, atr_inputs, closes
from ..indicators import atr, bollinger_bands, rsi
from .base import Signal, Strategy


class BollingerReversionStrategy(Strategy):
    name = "bollinger_reversion"
    category = "reversion"
    intraday = False

    def __init__(
        self,
        period: int = 20,
        num_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        atr_period: int = 14,
        atr_stop_mult: float = 0.5,
        max_hold: int = 10,
    ) -> None:
        super().__init__(
            period=period,
            num_std=num_std,
            rsi_period=rsi_period,
            rsi_oversold=rsi_oversold,
            rsi_overbought=rsi_overbought,
            atr_period=atr_period,
            atr_stop_mult=atr_stop_mult,
            max_hold=max_hold,
        )
        self.period = period
        self.num_std = num_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.max_hold = max_hold

    def prepare(self, bars: Sequence[Bar]) -> None:
        super().prepare(bars)
        c = closes(bars)
        self._upper, self._mid, self._lower = bollinger_bands(c, self.period, self.num_std)
        self._rsi = rsi(c, self.rsi_period)
        h, l, cc = atr_inputs(bars)
        self._atr = atr(h, l, cc, self.atr_period)
        self._bars_in_trade = 0
        self._position_side = 0  # -1, 0, +1 (tracked for the time-stop)

    def reset(self) -> None:
        super().reset()
        self._bars_in_trade = 0
        self._position_side = 0

    def warmup(self) -> int:
        return max(self.period, self.rsi_period, self.atr_period) + 1

    def on_bar(self, i: int) -> Signal | None:
        if i < self.warmup():
            return None
        bar = self._bars[i]
        upper, mid, lower = self._upper[i], self._mid[i], self._lower[i]
        r = self._rsi[i]
        a = self._atr[i]
        if None in (upper, mid, lower, r, a):
            return None

        # Time-stop while in a position.
        if self._position_side != 0:
            self._bars_in_trade += 1
            if self._bars_in_trade >= self.max_hold:
                self._position_side = 0
                self._bars_in_trade = 0
                return Signal("exit", reason=f"time-stop after {self.max_hold} bars")
            # Take profit at the mean.
            if self._position_side > 0 and bar.high >= mid:
                self._position_side = 0
                self._bars_in_trade = 0
                return Signal("exit", reason="reverted to mean (long target)")
            if self._position_side < 0 and bar.low <= mid:
                self._position_side = 0
                self._bars_in_trade = 0
                return Signal("exit", reason="reverted to mean (short target)")

        # Fresh entries only when flat.
        if self._position_side == 0:
            if bar.close < lower and r < self.rsi_oversold:
                self._position_side = 1
                self._bars_in_trade = 0
                return Signal(
                    "long",
                    reason=f"oversold: close<{lower:.2f}, RSI={r:.0f}",
                    stop=lower - self.atr_stop_mult * a,
                    target=mid,
                )
            if bar.close > upper and r > self.rsi_overbought:
                self._position_side = -1
                self._bars_in_trade = 0
                return Signal(
                    "short",
                    reason=f"overbought: close>{upper:.2f}, RSI={r:.0f}",
                    stop=upper + self.atr_stop_mult * a,
                    target=mid,
                )
        return None
