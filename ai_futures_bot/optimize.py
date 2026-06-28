"""Parameter optimisation via grid / random search.

Searches a strategy's parameter space to maximise an objective metric (Sharpe,
profit factor, Calmar, …) on a dataset. Used standalone and as the inner loop of
walk-forward analysis.

WARNING: optimising on a single dataset is exactly how strategies get overfit.
Always confirm chosen parameters out-of-sample (see ``walkforward.py``). This
module deliberately keeps the search small and the objectives risk-adjusted.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Sequence

from .backtester import Backtester
from .contracts import ContractSpec
from .data import Bar
from .risk import RiskConfig, RiskManager
from .strategies import get_strategy

_OBJECTIVES = {
    "sharpe": lambda m: m["sharpe"],
    "sortino": lambda m: m["sortino"],
    "calmar": lambda m: m["calmar"],
    "profit_factor": lambda m: 0.0 if m["profit_factor"] == "inf" else m["profit_factor"],
    "net_profit": lambda m: m["net_profit"],
    "expectancy": lambda m: m["expectancy"],
}


@dataclass
class OptResult:
    params: dict
    score: float
    metrics: dict

    def to_dict(self) -> dict:
        return {"params": self.params, "score": round(self.score, 4), "metrics": self.metrics}


def _score(metrics: dict, objective: str) -> float:
    fn = _OBJECTIVES.get(objective)
    if fn is None:
        raise ValueError(f"Unknown objective {objective!r}. Options: {sorted(_OBJECTIVES)}")
    # Penalise degenerate runs with too few trades so noise can't win.
    if metrics["num_trades"] < 5:
        return float("-inf")
    return float(fn(metrics))


def _run(strategy_name, params, bars, spec, risk_config, starting_cash, costs) -> dict:
    strat = get_strategy(strategy_name, **params)
    risk = RiskManager(config=risk_config or RiskConfig())
    bt = Backtester(
        strat, spec, risk,
        starting_cash=starting_cash,
        commission_per_contract=costs[0],
        slippage_ticks=costs[1],
    )
    return bt.run(bars).metrics


def grid_search(
    strategy_name: str,
    param_grid: dict[str, Sequence],
    bars: Sequence[Bar],
    spec: ContractSpec,
    *,
    objective: str = "sharpe",
    risk_config: RiskConfig | None = None,
    starting_cash: float = 50_000.0,
    commission_per_contract: float = 0.50,
    slippage_ticks: float = 1.0,
    top_n: int = 10,
) -> list[OptResult]:
    """Exhaustively search the cartesian product of ``param_grid``.

    Returns the top ``top_n`` results sorted by ``objective`` (best first).
    """
    keys = list(param_grid)
    combos = list(itertools.product(*(param_grid[k] for k in keys)))
    results: list[OptResult] = []
    for combo in combos:
        params = dict(zip(keys, combo))
        metrics = _run(strategy_name, params, bars, spec, risk_config,
                       starting_cash, (commission_per_contract, slippage_ticks))
        results.append(OptResult(params=params, score=_score(metrics, objective), metrics=metrics))
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_n]


def random_search(
    strategy_name: str,
    param_space: dict[str, Sequence],
    bars: Sequence[Bar],
    spec: ContractSpec,
    *,
    n_iter: int = 30,
    objective: str = "sharpe",
    seed: int = 7,
    risk_config: RiskConfig | None = None,
    starting_cash: float = 50_000.0,
    commission_per_contract: float = 0.50,
    slippage_ticks: float = 1.0,
    top_n: int = 10,
) -> list[OptResult]:
    """Sample ``n_iter`` random parameter combinations from ``param_space``."""
    keys = list(param_space)
    state = seed & 0xFFFFFFFF
    results: list[OptResult] = []
    seen: set[tuple] = set()
    for _ in range(n_iter):
        combo = []
        for k in keys:
            options = list(param_space[k])
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            combo.append(options[state % len(options)])
        combo_t = tuple(combo)
        if combo_t in seen:
            continue
        seen.add(combo_t)
        params = dict(zip(keys, combo))
        metrics = _run(strategy_name, params, bars, spec, risk_config,
                       starting_cash, (commission_per_contract, slippage_ticks))
        results.append(OptResult(params=params, score=_score(metrics, objective), metrics=metrics))
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_n]
