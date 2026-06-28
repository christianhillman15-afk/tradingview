"""Walk-forward analysis — the honest way to validate a strategy.

For each step it (optionally) optimises parameters on the in-sample history, then
evaluates them on the *next, unseen* out-of-sample segment, compounding capital
across segments. Stitching the out-of-sample segments together gives a realistic
picture of how the strategy would have performed trading forward with parameters
chosen only from the past — the test that exposes curve-fitting.

Each window's backtest runs from the start of data up to that window's end so the
strategy's indicator warm-up is satisfied; only the out-of-sample slice is scored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .backtester import Backtester
from .contracts import ContractSpec
from .data import Bar
from .metrics import compute_metrics
from .optimize import grid_search
from .risk import RiskConfig, RiskManager
from .strategies import get_strategy


@dataclass
class WFWindow:
    index: int
    test_start_time: str
    test_end_time: str
    params: dict
    start_equity: float
    end_equity: float
    oos_metrics: dict


@dataclass
class WalkForwardResult:
    windows: list[WFWindow] = field(default_factory=list)
    aggregate: dict = field(default_factory=dict)
    final_equity: float = 0.0

    def to_dict(self) -> dict:
        return {
            "windows": [w.__dict__ for w in self.windows],
            "aggregate": self.aggregate,
            "final_equity": round(self.final_equity, 2),
        }


def walk_forward(
    strategy_name: str,
    bars: Sequence[Bar],
    spec: ContractSpec,
    *,
    param_grid: dict[str, Sequence] | None = None,
    base_params: dict | None = None,
    n_splits: int = 5,
    train_frac: float = 0.5,
    objective: str = "sharpe",
    risk_config: RiskConfig | None = None,
    starting_cash: float = 50_000.0,
    commission_per_contract: float = 0.50,
    slippage_ticks: float = 1.0,
) -> WalkForwardResult:
    n = len(bars)
    if n < n_splits * 30:
        raise ValueError("Not enough bars for walk-forward; use more data or fewer splits.")
    base_params = base_params or {}
    risk_config = risk_config or RiskConfig()

    test_total = int(n * (1 - train_frac))
    seg = max(1, test_total // n_splits)
    first_test = n - seg * n_splits

    result = WalkForwardResult()
    capital = starting_cash
    stitched_equity: list = []
    stitched_trades: list = []

    for k in range(n_splits):
        test_start = first_test + k * seg
        test_end = min(test_start + seg, n)
        is_bars = bars[:test_start]

        # 1) Choose parameters using only in-sample data.
        if param_grid:
            best = grid_search(
                strategy_name, param_grid, is_bars, spec,
                objective=objective, risk_config=risk_config,
                starting_cash=starting_cash,
                commission_per_contract=commission_per_contract,
                slippage_ticks=slippage_ticks, top_n=1,
            )
            params = best[0].params if best else dict(base_params)
        else:
            params = dict(base_params)

        # 2) Evaluate out-of-sample (full run for warm-up; score only the slice).
        strat = get_strategy(strategy_name, **params)
        risk = RiskManager(config=risk_config)
        full = Backtester(
            strat, spec, risk, starting_cash=starting_cash,
            commission_per_contract=commission_per_contract, slippage_ticks=slippage_ticks,
        ).run(bars[:test_end])

        eq = full.equity_curve
        if test_start >= len(eq):
            continue
        base_eq = eq[test_start][1] or starting_cash
        oos_slice = eq[test_start:test_end]
        if not oos_slice:
            continue
        # Rebase the OOS slice onto the running compounded capital.
        rebased = [(ts, capital * (e / base_eq)) for ts, e in oos_slice]
        t0 = bars[test_start].timestamp
        t1 = bars[test_end - 1].timestamp
        oos_trades = [t for t in full.trades if t0 <= t.entry_time <= t1]
        win_metrics = compute_metrics(rebased, oos_trades, capital)

        result.windows.append(
            WFWindow(
                index=k,
                test_start_time=t0.isoformat(),
                test_end_time=t1.isoformat(),
                params=params,
                start_equity=round(capital, 2),
                end_equity=round(rebased[-1][1], 2),
                oos_metrics=win_metrics,
            )
        )
        stitched_equity.extend(rebased)
        stitched_trades.extend(oos_trades)
        capital = rebased[-1][1]

    result.aggregate = compute_metrics(stitched_equity, stitched_trades, starting_cash)
    result.final_equity = capital
    return result
