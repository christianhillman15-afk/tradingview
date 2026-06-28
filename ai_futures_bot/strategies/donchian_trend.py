"""Donchian-channel breakout trend following — the classic "Turtle" system.

Rules (Dennis/Eckhardt Turtle system, adapted):
  * Enter long on a break above the N-bar high (default 20), short below the
    N-bar low.
  * Exit on a break of the shorter M-bar channel in the opposite direction
    (default 10) — i.e. a long exits on a new 10-bar low.
  * Protective stop placed 2*ATR from entry (volatility-based).

This is a swing/position system, so ``intraday`` is False (positions can be
held across sessions). See STRATEGIES.md for sources.
"""

from __future__ import annotations

from typing import Sequence

from ..data import Bar, atr_inputs
from ..indicators import atr, rolling_max, rolling_min
from .base import Signal, Strategy


class DonchianTrendStrategy(Strategy):
    name = "donchian_trend"
    intraday = False

    def __init__(
        self,
        entry_period: int = 20,
        exit_period: int = 10,
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
    ) -> None:
        super().__init__(
            entry_period=entry_period,
            exit_period=exit_period,
            atr_period=atr_period,
            atr_stop_mult=atr_stop_mult,
        )
        self.entry_period = entry_period
        self.exit_period = exit_period
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult

    def prepare(self, bars: Sequence[Bar]) -> None:
        super().prepare(bars)
        h, l, c = atr_inputs(bars)
        self._upper = rolling_max(h, self.entry_period)
        self._lower = rolling_min(l, self.entry_period)
        # Exit channels are computed on the *prior* bar to avoid look-ahead;
        # we shift by reading index i-1 in on_bar.
        self._exit_high = rolling_max(h, self.exit_period)
        self._exit_low = rolling_min(l, self.exit_period)
        self._atr = atr(h, l, c, self.atr_period)

    def warmup(self) -> int:
        return max(self.entry_period, self.exit_period, self.atr_period) + 1

    def on_bar(self, i: int) -> Signal | None:
        if i < self.warmup():
            return None
        bar = self._bars[i]
        # Compare against the channel defined by bars strictly before i so the
        # breakout is detectable in real time.
        prev_upper = self._upper[i - 1]
        prev_lower = self._lower[i - 1]
        prev_exit_high = self._exit_high[i - 1]
        prev_exit_low = self._exit_low[i - 1]
        a = self._atr[i]
        if None in (prev_upper, prev_lower, prev_exit_high, prev_exit_low, a):
            return None

        # Breakout entries.
        if bar.high >= prev_upper:
            return Signal(
                "long",
                reason=f"break {self.entry_period}-bar high {prev_upper:.2f}",
                stop=bar.close - self.atr_stop_mult * a,
            )
        if bar.low <= prev_lower:
            return Signal(
                "short",
                reason=f"break {self.entry_period}-bar low {prev_lower:.2f}",
                stop=bar.close + self.atr_stop_mult * a,
            )

        # Opposite shorter-channel break => exit (handled as a flat request;
        # the backtester only acts on it if a position is open in that side).
        if bar.low <= prev_exit_low:
            return Signal("exit", reason=f"long exit on {self.exit_period}-bar low")
        if bar.high >= prev_exit_high:
            return Signal("exit", reason=f"short exit on {self.exit_period}-bar high")
        return None
