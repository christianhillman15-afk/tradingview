"""Opening Range Breakout (ORB) — a staple index-futures day-trading strategy.

Rules (researched ORB playbook for ES/MES):
  * Define the opening range as the high/low of the first ``opening_minutes``
    bars of the session (default 5).
  * Go long when price breaks above the opening-range high while above session
    VWAP; go short below the opening-range low while below VWAP.
  * Stop at the opposite side / midpoint of the opening range.
  * Target a ``target_mult`` extension of the range height (default 1.5x).
  * One trade per session per direction; flatten by ``exit_minute`` (default
    90 minutes in, ~11:00 ET) and at session end.

``intraday`` is True so no position is ever carried overnight.
"""

from __future__ import annotations

from typing import Sequence

from ..data import Bar, atr_inputs, closes, highs, lows, session_ids, volumes
from ..indicators import session_vwap
from .base import Signal, Strategy


class OpeningRangeBreakoutStrategy(Strategy):
    name = "opening_range_breakout"
    category = "breakout"
    intraday = True

    def __init__(
        self,
        opening_minutes: int = 5,
        exit_minute: int = 90,
        target_mult: float = 1.5,
        require_vwap: bool = True,
    ) -> None:
        super().__init__(
            opening_minutes=opening_minutes,
            exit_minute=exit_minute,
            target_mult=target_mult,
            require_vwap=require_vwap,
        )
        self.opening_minutes = opening_minutes
        self.exit_minute = exit_minute
        self.target_mult = target_mult
        self.require_vwap = require_vwap

    def prepare(self, bars: Sequence[Bar]) -> None:
        super().prepare(bars)
        h, l, c = atr_inputs(bars)
        v = volumes(bars)
        sids = session_ids(bars)
        self._vwap = session_vwap(h, l, c, v, sids)

        n = len(bars)
        self._or_high: list[float | None] = [None] * n
        self._or_low: list[float | None] = [None] * n
        self._minute: list[int] = [0] * n

        # Walk sessions, building the opening range incrementally.
        current: str | None = None
        idx_in_session = 0
        or_hi = or_lo = None
        for i in range(n):
            if sids[i] != current:
                current = sids[i]
                idx_in_session = 0
                or_hi = h[i]
                or_lo = l[i]
            else:
                idx_in_session += 1
                if idx_in_session < self.opening_minutes:
                    or_hi = max(or_hi, h[i])  # type: ignore[arg-type]
                    or_lo = min(or_lo, l[i])  # type: ignore[arg-type]
            self._minute[i] = idx_in_session
            # Range is only "set" once the opening window has fully formed.
            if idx_in_session >= self.opening_minutes - 1:
                self._or_high[i] = or_hi
                self._or_low[i] = or_lo

        self._entered_session: set[str] = set()

    def reset(self) -> None:
        super().reset()
        self._entered_session = set()

    def warmup(self) -> int:
        return self.opening_minutes

    def on_bar(self, i: int) -> Signal | None:
        bar = self._bars[i]
        sid = bar.session_id
        minute = self._minute[i]

        # Force flat near/after the cutoff.
        if minute >= self.exit_minute:
            return Signal("exit", reason="ORB session cutoff")

        or_hi = self._or_high[i]
        or_lo = self._or_low[i]
        if or_hi is None or or_lo is None:
            return None
        # Don't trade during the opening-range formation itself.
        if minute < self.opening_minutes:
            return None
        if sid in self._entered_session:
            return None

        vwap = self._vwap[i]
        rng = or_hi - or_lo
        if rng <= 0:
            return None

        if bar.high >= or_hi and (not self.require_vwap or (vwap is not None and bar.close >= vwap)):
            self._entered_session.add(sid)
            return Signal(
                "long",
                reason=f"ORB long >{or_hi:.2f}",
                stop=or_lo,
                target=or_hi + self.target_mult * rng,
            )
        if bar.low <= or_lo and (not self.require_vwap or (vwap is not None and bar.close <= vwap)):
            self._entered_session.add(sid)
            return Signal(
                "short",
                reason=f"ORB short <{or_lo:.2f}",
                stop=or_hi,
                target=or_lo - self.target_mult * rng,
            )
        return None
