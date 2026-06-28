"""Monte Carlo stress testing of a strategy's trade sequence.

A single backtest equity curve is one sample from a distribution. By resampling
(bootstrap) or reshuffling the realised trade P&Ls thousands of times we estimate
the *distribution* of outcomes — the probability of profit, the likely-worst
drawdown, and the risk of ruin — which a single equity curve hides.

Pure stdlib: a seeded LCG RNG so results are reproducible without numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .portfolio import Trade


@dataclass
class MonteCarloResult:
    simulations: int
    method: str
    prob_profit: float
    risk_of_ruin: float
    ruin_threshold: float
    final_equity_pctiles: dict[str, float]   # p5/p25/p50/p75/p95
    max_drawdown_pctiles: dict[str, float]    # p50/p95 (fraction)
    mean_final_equity: float

    def to_dict(self) -> dict:
        return {
            "simulations": self.simulations,
            "method": self.method,
            "prob_profit": round(self.prob_profit, 4),
            "risk_of_ruin": round(self.risk_of_ruin, 4),
            "ruin_threshold": round(self.ruin_threshold, 2),
            "final_equity_pctiles": {k: round(v, 2) for k, v in self.final_equity_pctiles.items()},
            "max_drawdown_pctiles": {k: round(v, 4) for k, v in self.max_drawdown_pctiles.items()},
            "mean_final_equity": round(self.mean_final_equity, 2),
        }


class _LCG:
    """Tiny seeded linear-congruential RNG (Numerical Recipes constants)."""

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF

    def randint(self, n: int) -> int:
        self.state = (1664525 * self.state + 1013904223) & 0xFFFFFFFF
        return self.state % n


def monte_carlo(
    trades: Sequence[Trade],
    starting_cash: float,
    *,
    simulations: int = 2000,
    method: str = "resample",      # "resample" (bootstrap) | "shuffle"
    ruin_fraction: float = 0.5,    # ruin = equity falls to this fraction of start
    seed: int = 12345,
) -> MonteCarloResult:
    """Run a Monte Carlo simulation over the trade P&L sequence."""
    pnls = [t.pnl for t in trades]
    ruin_threshold = starting_cash * ruin_fraction
    if not pnls:
        return MonteCarloResult(
            simulations=0, method=method, prob_profit=0.0, risk_of_ruin=0.0,
            ruin_threshold=ruin_threshold,
            final_equity_pctiles={k: starting_cash for k in ("p5", "p25", "p50", "p75", "p95")},
            max_drawdown_pctiles={"p50": 0.0, "p95": 0.0},
            mean_final_equity=starting_cash,
        )

    rng = _LCG(seed)
    n = len(pnls)
    finals: list[float] = []
    drawdowns: list[float] = []
    ruined = 0
    profitable = 0

    for _ in range(simulations):
        if method == "shuffle":
            order = list(range(n))
            for i in range(n - 1, 0, -1):     # Fisher-Yates
                j = rng.randint(i + 1)
                order[i], order[j] = order[j], order[i]
            seq = (pnls[k] for k in order)
        else:  # bootstrap resample with replacement
            seq = (pnls[rng.randint(n)] for _ in range(n))

        equity = starting_cash
        peak = starting_cash
        max_dd = 0.0
        hit_ruin = False
        for pnl in seq:
            equity += pnl
            peak = max(peak, equity)
            if peak > 0:
                max_dd = max(max_dd, (peak - equity) / peak)
            if equity <= ruin_threshold:
                hit_ruin = True
        finals.append(equity)
        drawdowns.append(max_dd)
        if hit_ruin:
            ruined += 1
        if equity > starting_cash:
            profitable += 1

    finals.sort()
    drawdowns.sort()
    return MonteCarloResult(
        simulations=simulations,
        method=method,
        prob_profit=profitable / simulations,
        risk_of_ruin=ruined / simulations,
        ruin_threshold=ruin_threshold,
        final_equity_pctiles={
            "p5": _pctile(finals, 0.05),
            "p25": _pctile(finals, 0.25),
            "p50": _pctile(finals, 0.50),
            "p75": _pctile(finals, 0.75),
            "p95": _pctile(finals, 0.95),
        },
        max_drawdown_pctiles={
            "p50": _pctile(drawdowns, 0.50),
            "p95": _pctile(drawdowns, 0.95),
        },
        mean_final_equity=sum(finals) / len(finals),
    )


def _pctile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]
