"""Live (paper) trader.

Streams bars through the shared :class:`ExecutionEngine` one at a time,
mirroring exactly what the backtester does per bar, and writes a dashboard
state snapshot as it goes. This is the engine behind ``cli live`` — it paper
trades a (synthetic or CSV) feed against a $50,000 simulated account by default.

To keep per-bar ``prepare`` cost bounded during long runs it operates over a
rolling window of the most recent ``history_window`` bars. Rule-based
strategies only look back tens of bars, so this is lossless for them; the ML
strategy is best pre-trained and loaded via ``model_path`` rather than retrained
each bar.
"""

from __future__ import annotations

import time as _time
from typing import Callable, Iterable, Sequence

from .contracts import ContractSpec
from .data import Bar
from .engine import ExecutionEngine
from .portfolio import Portfolio
from .risk import RiskManager
from .state import snapshot, write_state
from .strategies.base import Strategy


class LiveTrader:
    def __init__(
        self,
        strategy: Strategy,
        spec: ContractSpec,
        risk: RiskManager,
        *,
        starting_cash: float = 50_000.0,
        commission_per_contract: float = 2.50,
        slippage_ticks: float = 1.0,
        history_window: int = 2000,
        state_path: str | None = None,
        update_every: int = 1,
    ) -> None:
        self.strategy = strategy
        self.spec = spec
        self.risk = risk
        self.portfolio = Portfolio(
            spec,
            starting_cash=starting_cash,
            commission_per_contract=commission_per_contract,
            slippage_ticks=slippage_ticks,
        )
        self.engine = ExecutionEngine(strategy, spec, risk, self.portfolio)
        self.history_window = history_window
        self.state_path = state_path
        self.update_every = max(1, update_every)
        self._buffer: list[Bar] = []
        self._bars_processed = 0

    def on_bar(self, bar: Bar, is_session_end: bool = False) -> None:
        """Process a single incoming bar (the live event-loop entry point)."""
        self._buffer.append(bar)
        if len(self._buffer) > self.history_window:
            self._buffer = self._buffer[-self.history_window :]
        # Re-derive indicator series over the window, then step the latest bar.
        self.engine.prepare(self._buffer)
        self.engine.step(self._buffer, len(self._buffer) - 1, is_session_end)
        self._bars_processed += 1

    def run(
        self,
        bars: Iterable[Bar],
        *,
        speed: float = 0.0,
        on_update: Callable[[dict], None] | None = None,
    ) -> Portfolio:
        """Replay a bar stream as if live. ``speed`` is seconds slept per bar."""
        bars = list(bars)
        n = len(bars)
        for i, bar in enumerate(bars):
            is_session_end = (i == n - 1) or (bars[i + 1].session_id != bar.session_id)
            self.on_bar(bar, is_session_end)
            if self._bars_processed % self.update_every == 0 or i == n - 1:
                state = self.build_state(bar)
                if self.state_path:
                    write_state(self.state_path, state)
                if on_update:
                    on_update(state)
            if speed > 0:
                _time.sleep(speed)
        # Flatten at the very end so realised PnL is complete.
        if self.portfolio.position is not None and bars:
            self.engine.flatten(bars[-1], reason="session end")
            final = self.build_state(bars[-1])
            if self.state_path:
                write_state(self.state_path, final)
            if on_update:
                on_update(final)
        return self.portfolio

    def seed_history(self, bars: list[Bar]) -> None:
        """Warm up the indicator buffer without trading (for a fresh account)."""
        self._buffer = list(bars)[-self.history_window:]

    def export_account(self, feed=None) -> dict:
        """Serialize the full account for persistence/resume."""
        from .data import bar_to_dict

        acct = {
            "portfolio": self.portfolio.export_state(),
            "buffer": [bar_to_dict(b) for b in self._buffer[-self.history_window:]],
            "bars_processed": self._bars_processed,
        }
        if feed is not None:
            acct["feed"] = feed.export()
        return acct

    def restore_account(self, account: dict) -> None:
        from .data import bar_from_dict

        self.portfolio.restore_state(account.get("portfolio", {}))
        self._buffer = [bar_from_dict(d) for d in account.get("buffer", [])]
        self._bars_processed = int(account.get("bars_processed", 0))

    def build_state(self, last_bar: Bar) -> dict:
        return snapshot(
            portfolio=self.portfolio,
            spec=self.spec,
            strategy_name=self.strategy.name,
            last_price=last_bar.close,
            last_time=last_bar.timestamp,
            events=self.engine.events,
            mode="live-paper",
            bars_processed=self._bars_processed,
            risk_status={
                "can_trade": self.risk.can_trade(),
                "halted": self.risk.halted,
                "day_realized": round(self.risk.day_realized, 2),
                "equity_peak": round(self.risk.equity_peak, 2),
            },
        )
