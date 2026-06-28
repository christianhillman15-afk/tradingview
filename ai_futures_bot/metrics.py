"""Performance analytics computed from an equity curve and a trade blotter.

Pure stdlib. Produces the numbers shown on the dashboard's Overview and
Performance tabs: ROI, net profit, win rate, profit factor, expectancy,
max drawdown, and an annualised Sharpe ratio.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Sequence

from .portfolio import Trade


def compute_metrics(
    equity_curve: Sequence[tuple[datetime, float]],
    trades: Sequence[Trade],
    starting_cash: float,
) -> dict:
    final_equity = equity_curve[-1][1] if equity_curve else starting_cash
    net_profit = final_equity - starting_cash
    roi = (net_profit / starting_cash) if starting_cash else 0.0

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl < 0]
    num_trades = len(trades)
    win_rate = (len(wins) / num_trades) if num_trades else 0.0

    gross_profit = sum(t.pnl for t in wins)
    gross_loss = -sum(t.pnl for t in losses)
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
        math.inf if gross_profit > 0 else 0.0
    )
    avg_win = (gross_profit / len(wins)) if wins else 0.0
    avg_loss = (gross_loss / len(losses)) if losses else 0.0
    expectancy = (sum(t.pnl for t in trades) / num_trades) if num_trades else 0.0
    payoff = (avg_win / avg_loss) if avg_loss > 0 else 0.0

    max_dd, max_dd_dollars = _max_drawdown(equity_curve)
    sharpe = _sharpe(equity_curve)
    cagr = _cagr(equity_curve, starting_cash)
    max_consec_losses = _max_consecutive_losses(trades)

    return {
        "starting_equity": round(starting_cash, 2),
        "final_equity": round(final_equity, 2),
        "net_profit": round(net_profit, 2),
        "roi_pct": round(roi * 100, 3),
        "cagr_pct": round(cagr * 100, 3),
        "num_trades": num_trades,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(win_rate * 100, 2),
        "profit_factor": _round_or_inf(profit_factor),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "payoff_ratio": round(payoff, 3),
        "expectancy": round(expectancy, 2),
        "max_drawdown_pct": round(max_dd * 100, 3),
        "max_drawdown_dollars": round(max_dd_dollars, 2),
        "sharpe": round(sharpe, 3),
        "max_consecutive_losses": max_consec_losses,
    }


def _max_drawdown(equity_curve: Sequence[tuple[datetime, float]]) -> tuple[float, float]:
    peak = -math.inf
    max_dd = 0.0
    max_dd_dollars = 0.0
    for _, eq in equity_curve:
        peak = max(peak, eq)
        if peak > 0:
            dd = (peak - eq) / peak
            if dd > max_dd:
                max_dd = dd
                max_dd_dollars = peak - eq
    return max_dd, max_dd_dollars


def _sharpe(equity_curve: Sequence[tuple[datetime, float]]) -> float:
    if len(equity_curve) < 3:
        return 0.0
    rets: list[float] = []
    for (_, prev), (_, cur) in zip(equity_curve, equity_curve[1:]):
        if prev > 0:
            rets.append(cur / prev - 1.0)
    if len(rets) < 2:
        return 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    std = math.sqrt(var)
    if std == 0:
        return 0.0
    periods_per_year = _infer_periods_per_year(equity_curve)
    return (mean / std) * math.sqrt(periods_per_year)


def _infer_periods_per_year(equity_curve: Sequence[tuple[datetime, float]]) -> float:
    first, last = equity_curve[0][0], equity_curve[-1][0]
    span_seconds = (last - first).total_seconds()
    if span_seconds <= 0:
        return 252.0
    span_years = span_seconds / (365.25 * 24 * 3600)
    # Effective sampling rate; compresses weekend/overnight gaps automatically.
    return max(len(equity_curve) / max(span_years, 1e-9), 1.0)


def _cagr(equity_curve: Sequence[tuple[datetime, float]], starting_cash: float) -> float:
    if len(equity_curve) < 2 or starting_cash <= 0:
        return 0.0
    first, last = equity_curve[0][0], equity_curve[-1][0]
    years = (last - first).total_seconds() / (365.25 * 24 * 3600)
    final_equity = equity_curve[-1][1]
    # Annualising a sub-day window explodes the exponent and is meaningless, so
    # only report CAGR once there is at least a day of data.
    if years < 1.0 / 365.25 or final_equity <= 0:
        return 0.0
    try:
        return (final_equity / starting_cash) ** (1 / years) - 1
    except OverflowError:
        return 0.0


def _max_consecutive_losses(trades: Sequence[Trade]) -> int:
    worst = 0
    run = 0
    for t in trades:
        if t.pnl < 0:
            run += 1
            worst = max(worst, run)
        else:
            run = 0
    return worst


def _round_or_inf(x: float) -> float | str:
    return "inf" if x == math.inf else round(x, 3)
