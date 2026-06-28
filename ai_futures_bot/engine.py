"""Shared execution engine used by both the backtester and the live trader.

Holds one bar's worth of decision logic in :meth:`ExecutionEngine.step`:
roll risk counters, fill protective stops/targets intrabar, act on the
strategy signal subject to risk gating, and flatten intraday positions at the
session close. Keeping this in one place guarantees backtest and live behave
identically.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from .contracts import ContractSpec
from .data import Bar, atr_inputs
from .indicators import atr
from .portfolio import Portfolio, Trade
from .risk import RiskManager
from .strategies.base import Signal, Strategy


class ExecutionEngine:
    def __init__(
        self,
        strategy: Strategy,
        spec: ContractSpec,
        risk: RiskManager,
        portfolio: Portfolio,
        *,
        default_atr_period: int = 14,
        default_atr_stop_mult: float = 2.0,
        max_events: int = 5000,
    ) -> None:
        self.strategy = strategy
        self.spec = spec
        self.risk = risk
        self.portfolio = portfolio
        self.default_atr_period = default_atr_period
        self.default_atr_stop_mult = default_atr_stop_mult
        self.max_events = max_events
        self.events: list[dict] = []
        self._atr_series: list[float | None] = []

    def prepare(self, bars: Sequence[Bar]) -> None:
        self.strategy.reset()
        self.strategy.prepare(bars)
        h, l, c = atr_inputs(bars)
        self._atr_series = atr(h, l, c, self.default_atr_period)

    def step(self, bars: Sequence[Bar], i: int, is_session_end: bool) -> None:
        bar = bars[i]
        self.risk.start_bar(bar.session_id, self.portfolio.equity(bar.close))

        # 1) Protective stop / target (intrabar).
        self._check_protective_exits(bar)

        # 2) Strategy signal.
        signal = self.strategy.on_bar(i)
        if signal is not None:
            atr_val = self._atr_series[i] if i < len(self._atr_series) else None
            self._handle_signal(bar, signal, atr_val)

        # 3) Flatten intraday strategies at the session close.
        if is_session_end and self.strategy.intraday and self.portfolio.position is not None:
            trade = self.portfolio.close(bar.close, bar.timestamp, reason="session close")
            self._on_close(trade)

        # 4) Mark equity for the curve.
        self.portfolio.mark(bar.timestamp, bar.close)

    def flatten(self, bar: Bar, reason: str = "flatten") -> None:
        if self.portfolio.position is not None:
            trade = self.portfolio.close(bar.close, bar.timestamp, reason=reason)
            self._on_close(trade)
            self.portfolio.mark(bar.timestamp, bar.close)

    # --- internals -----------------------------------------------------
    def _check_protective_exits(self, bar: Bar) -> None:
        pos = self.portfolio.position
        if pos is None:
            return
        stop, target = pos.stop, pos.target
        hit_price = None
        reason = ""
        if pos.side > 0:  # long
            if stop is not None and bar.low <= stop:
                hit_price, reason = stop, "stop"
            elif target is not None and bar.high >= target:
                hit_price, reason = target, "target"
        else:  # short
            if stop is not None and bar.high >= stop:
                hit_price, reason = stop, "stop"
            elif target is not None and bar.low <= target:
                hit_price, reason = target, "target"
        if hit_price is not None:
            trade = self.portfolio.close(hit_price, bar.timestamp, reason=reason)
            self._on_close(trade)

    def _handle_signal(self, bar: Bar, signal: Signal, atr_val: float | None) -> None:
        pos = self.portfolio.position

        if signal.action == "exit":
            if pos is not None:
                trade = self.portfolio.close(bar.close, bar.timestamp, reason=signal.reason or "exit")
                self._on_close(trade)
            return

        target_side = 1 if signal.action == "long" else -1
        if pos is not None and pos.side == target_side:
            return
        if pos is not None and pos.side != target_side:
            trade = self.portfolio.close(bar.close, bar.timestamp, reason="reverse")
            self._on_close(trade)

        if not self.risk.can_trade():
            self._log(bar.timestamp, "blocked", f"risk halt: {signal.action} skipped")
            return

        entry = bar.close
        stop = signal.stop
        if stop is None:
            if atr_val is None:
                return
            stop = entry - target_side * self.default_atr_stop_mult * atr_val

        equity = self.portfolio.equity(bar.close)
        qty = self.risk.position_size(equity, entry, stop, self.spec, signal.strength)
        if qty <= 0:
            self._log(bar.timestamp, "skip", "size=0 (risk/margin)")
            return

        self.portfolio.open(
            target_side,
            qty,
            entry,
            bar.timestamp,
            stop=stop,
            target=signal.target,
            strategy=self.strategy.name,
            reason=signal.reason,
        )
        self._log(
            bar.timestamp,
            "open",
            f"{signal.action} {qty} @ {entry:.2f} stop {stop:.2f}"
            + (f" tgt {signal.target:.2f}" if signal.target else "")
            + (f" — {signal.reason}" if signal.reason else ""),
        )

    def _on_close(self, trade: Trade | None) -> None:
        if trade is None:
            return
        self.risk.register_trade(trade.pnl)
        self._log(
            trade.exit_time,
            "close",
            f"{trade.side} {trade.quantity} @ {trade.exit_price:.2f} "
            f"pnl {trade.pnl:+.2f} ({trade.exit_reason})",
        )

    def _log(self, time: datetime, kind: str, message: str) -> None:
        self.events.append({"time": time.isoformat(), "kind": kind, "message": message})
        if len(self.events) > self.max_events * 2:
            del self.events[: -self.max_events]
