"""Multi-market portfolio backtesting — the biggest documented Sharpe lever.

The research is unambiguous: a single futures market gives a trend Sharpe of only
~0.4 gross, but combining many *low-correlation* markets (avg pairwise correlation
~0.08) with risk-balanced sizing lifts the portfolio Sharpe toward ~1.4 (AQR;
Moskowitz/Ooi/Pedersen). This module runs one strategy across a basket of
contracts, each as an independent risk-balanced "sleeve", and aggregates them
into a portfolio — then reports the diversification benefit explicitly.

Risk balancing: capital is split across sleeves and each sleeve sizes positions
by ATR/1%-risk, so per-trade dollar risk is equalised across instruments
regardless of their raw volatility (risk parity at the trade level). Two capital
weightings are offered: ``equal`` (default) and ``inverse_vol`` (allocate more
capital to lower-volatility markets — the inverse-volatility precursor to full
risk parity).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Sequence

from .backtester import Backtester
from .contracts import get_contract
from .data import Bar, closes
from .metrics import compute_metrics
from .portfolio import Trade
from .risk import RiskConfig, RiskManager
from .strategies import get_strategy


@dataclass
class Sleeve:
    symbol: str
    start_capital: float
    metrics: dict
    equity_curve: list[tuple[datetime, float]]
    trades: list[Trade]


@dataclass
class PortfolioResult:
    strategy: str
    weighting: str
    starting_cash: float
    sleeves: list[Sleeve] = field(default_factory=list)
    portfolio_equity: list[tuple[datetime, float]] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    avg_correlation: float = 0.0
    mean_sleeve_sharpe: float = 0.0
    diversification_ratio: float = 0.0

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "weighting": self.weighting,
            "starting_cash": round(self.starting_cash, 2),
            "metrics": self.metrics,
            "avg_correlation": round(self.avg_correlation, 4),
            "mean_sleeve_sharpe": round(self.mean_sleeve_sharpe, 3),
            "diversification_ratio": round(self.diversification_ratio, 3),
            "sleeves": [
                {"symbol": s.symbol, "start_capital": round(s.start_capital, 2), "metrics": s.metrics}
                for s in self.sleeves
            ],
        }


def _annualized_vol(bars: Sequence[Bar]) -> float:
    c = closes(bars)
    rets = [c[i] / c[i - 1] - 1.0 for i in range(1, len(c)) if c[i - 1] > 0]
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var)


def _weights(symbol_bars: dict[str, list[Bar]], weighting: str) -> dict[str, float]:
    symbols = list(symbol_bars)
    if weighting == "inverse_vol":
        inv = {}
        for s in symbols:
            v = _annualized_vol(symbol_bars[s])
            inv[s] = (1.0 / v) if v > 0 else 0.0
        total = sum(inv.values())
        if total > 0:
            return {s: inv[s] / total for s in symbols}
    # equal weight (default / fallback)
    return {s: 1.0 / len(symbols) for s in symbols}


def portfolio_backtest(
    strategy_name: str,
    symbol_bars: dict[str, list[Bar]],
    *,
    starting_cash: float = 50_000.0,
    weighting: str = "equal",
    strategy_params: dict | None = None,
    risk_config: RiskConfig | None = None,
    commission_per_contract: float = 0.50,
    slippage_ticks: float = 1.0,
) -> PortfolioResult:
    if len(symbol_bars) < 1:
        raise ValueError("Need at least one symbol to run a portfolio backtest.")
    strategy_params = strategy_params or {}
    risk_config = risk_config or RiskConfig()
    weights = _weights(symbol_bars, weighting)

    result = PortfolioResult(strategy=strategy_name, weighting=weighting, starting_cash=starting_cash)
    for sym, bars in symbol_bars.items():
        spec = get_contract(sym)
        capital = starting_cash * weights[sym]
        strat = get_strategy(strategy_name, **strategy_params)
        bt = Backtester(
            strat, spec, RiskManager(config=risk_config), starting_cash=capital,
            commission_per_contract=commission_per_contract, slippage_ticks=slippage_ticks,
        )
        res = bt.run(bars)
        result.sleeves.append(
            Sleeve(symbol=sym, start_capital=capital, metrics=res.metrics,
                   equity_curve=res.equity_curve, trades=res.trades)
        )

    # Aggregate sleeves onto a common timeline (two-pointer forward-fill).
    port_equity, sleeve_series = _aggregate(result.sleeves)
    all_trades = [t for s in result.sleeves for t in s.trades]
    result.portfolio_equity = port_equity
    result.metrics = compute_metrics(port_equity, all_trades, starting_cash)
    result.avg_correlation = _avg_pairwise_correlation(sleeve_series)
    sharpes = [s.metrics["sharpe"] for s in result.sleeves]
    result.mean_sleeve_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0.0
    if result.mean_sleeve_sharpe != 0:
        result.diversification_ratio = result.metrics["sharpe"] / result.mean_sleeve_sharpe
    return result


def _aggregate(sleeves: list[Sleeve]):
    all_ts = sorted({ts for s in sleeves for ts, _ in s.equity_curve})
    pointers = [0] * len(sleeves)
    last = [s.start_capital for s in sleeves]
    port: list[tuple[datetime, float]] = []
    sleeve_series: list[list[float]] = [[] for _ in sleeves]
    for ts in all_ts:
        total = 0.0
        for i, s in enumerate(sleeves):
            ec = s.equity_curve
            while pointers[i] < len(ec) and ec[pointers[i]][0] <= ts:
                last[i] = ec[pointers[i]][1]
                pointers[i] += 1
            total += last[i]
            sleeve_series[i].append(last[i])
        port.append((ts, total))
    return port, sleeve_series


def _avg_pairwise_correlation(sleeve_series: list[list[float]]) -> float:
    # Convert each sleeve's equity to returns, then average pairwise correlation.
    returns = []
    for eq in sleeve_series:
        r = [eq[i] / eq[i - 1] - 1.0 for i in range(1, len(eq)) if eq[i - 1] > 0]
        returns.append(r)
    n = len(returns)
    if n < 2:
        return 0.0
    corrs = []
    for i in range(n):
        for j in range(i + 1, n):
            c = _pearson(returns[i], returns[j])
            if c is not None:
                corrs.append(c)
    return sum(corrs) / len(corrs) if corrs else 0.0


def _pearson(a: list[float], b: list[float]) -> float | None:
    m = min(len(a), len(b))
    if m < 3:
        return None
    a, b = a[:m], b[:m]
    ma, mb = sum(a) / m, sum(b) / m
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((x - mb) ** 2 for x in b)
    if va <= 0 or vb <= 0:
        return None
    cov = sum((a[k] - ma) * (b[k] - mb) for k in range(m))
    return cov / math.sqrt(va * vb)
