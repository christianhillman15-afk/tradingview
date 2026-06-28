"""Event-driven backtester — drives the shared :class:`ExecutionEngine` over a
full bar history and bundles the result (portfolio, metrics, equity curve,
trade blotter, event log) for reporting and the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from .contracts import ContractSpec
from .data import Bar
from .engine import ExecutionEngine
from .metrics import compute_metrics
from .portfolio import Portfolio, Trade
from .risk import RiskManager
from .strategies.base import Strategy


@dataclass
class BacktestResult:
    spec: ContractSpec
    strategy_name: str
    portfolio: Portfolio
    metrics: dict
    events: list[dict] = field(default_factory=list)

    @property
    def trades(self) -> list[Trade]:
        return self.portfolio.trades

    @property
    def equity_curve(self) -> list[tuple[datetime, float]]:
        return self.portfolio.equity_curve


class Backtester:
    def __init__(
        self,
        strategy: Strategy,
        spec: ContractSpec,
        risk: RiskManager,
        *,
        starting_cash: float = 50_000.0,
        commission_per_contract: float = 2.50,
        slippage_ticks: float = 1.0,
        default_atr_period: int = 14,
        default_atr_stop_mult: float = 2.0,
    ) -> None:
        self.strategy = strategy
        self.spec = spec
        self.risk = risk
        self.starting_cash = starting_cash
        self.commission_per_contract = commission_per_contract
        self.slippage_ticks = slippage_ticks
        self.default_atr_period = default_atr_period
        self.default_atr_stop_mult = default_atr_stop_mult

    def run(self, bars: Sequence[Bar]) -> BacktestResult:
        if not bars:
            raise ValueError("No bars to backtest.")
        portfolio = Portfolio(
            self.spec,
            starting_cash=self.starting_cash,
            commission_per_contract=self.commission_per_contract,
            slippage_ticks=self.slippage_ticks,
        )
        engine = ExecutionEngine(
            self.strategy,
            self.spec,
            self.risk,
            portfolio,
            default_atr_period=self.default_atr_period,
            default_atr_stop_mult=self.default_atr_stop_mult,
        )
        engine.prepare(bars)

        n = len(bars)
        for i in range(n):
            is_session_end = (i == n - 1) or (bars[i + 1].session_id != bars[i].session_id)
            engine.step(bars, i, is_session_end)

        # Final flatten so realised PnL is complete.
        if portfolio.position is not None:
            engine.flatten(bars[-1], reason="end of data")

        metrics = compute_metrics(portfolio.equity_curve, portfolio.trades, self.starting_cash)
        return BacktestResult(
            spec=self.spec,
            strategy_name=self.strategy.name,
            portfolio=portfolio,
            metrics=metrics,
            events=engine.events,
        )
