"""Dashboard state snapshots.

A single JSON document describes everything the dashboard renders: account
summary, open position, recent trades, equity curve, metrics, event log, and
risk status. Both the backtester (one final snapshot) and the live trader
(updated each bar) produce the same shape, so the dashboard is agnostic to the
source.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from typing import Sequence

from .contracts import ContractSpec
from .metrics import compute_metrics
from .portfolio import Portfolio

_MAX_CURVE_POINTS = 1500
_MAX_TRADES = 200
_MAX_EVENTS = 300


def snapshot(
    *,
    portfolio: Portfolio,
    spec: ContractSpec,
    strategy_name: str,
    last_price: float,
    last_time: datetime | None,
    events: Sequence[dict],
    mode: str = "backtest",
    bars_processed: int = 0,
    risk_status: dict | None = None,
    generated_at: datetime | None = None,
) -> dict:
    equity = portfolio.equity(last_price)
    metrics = compute_metrics(portfolio.equity_curve, portfolio.trades, portfolio.starting_cash)
    curve = _downsample(portfolio.equity_curve, _MAX_CURVE_POINTS)
    unrealized = portfolio.position.unrealized(last_price, spec) if portfolio.position else 0.0
    return {
        "mode": mode,
        "generated_at": (generated_at or _now()).isoformat(),
        "symbol": spec.symbol,
        "contract_name": spec.name,
        "exchange": spec.exchange,
        "tradingview_symbol": _tv_symbol(spec),
        "strategy": strategy_name,
        "bars_processed": bars_processed,
        "last_price": round(last_price, 4),
        "last_bar_time": last_time.isoformat() if last_time else None,
        "account": {
            "starting_equity": round(portfolio.starting_cash, 2),
            "equity": round(equity, 2),
            "cash": round(portfolio.cash, 2),
            "realized_pnl": round(portfolio.realized_pnl, 2),
            "unrealized_pnl": round(unrealized, 2),
            "roi_pct": metrics["roi_pct"],
        },
        "position": portfolio.open_position_dict(last_price),
        "metrics": metrics,
        "trades": [t.to_dict() for t in portfolio.trades[-_MAX_TRADES:]][::-1],
        "equity_curve": [[t.isoformat(), round(e, 2)] for t, e in curve],
        "events": list(events[-_MAX_EVENTS:])[::-1],
        "risk": risk_status or {},
    }


def portfolio_snapshot(pr, *, generated_at: datetime | None = None) -> dict:
    """Build a dashboard state document from a multi-market portfolio result.

    Keeps the standard fields so the existing tabs render, and adds a
    ``portfolio`` block (sleeves + correlation matrix) for the Portfolio tab.
    """
    from .contracts import get_contract

    m = pr.metrics
    curve = _downsample(pr.portfolio_equity, _MAX_CURVE_POINTS)
    symbols = [s.symbol for s in pr.sleeves]
    first_tv = _tv_symbol(get_contract(symbols[0])) if symbols else "CME_MINI:ES1!"

    trades = []
    for s in pr.sleeves:
        for t in s.trades[-40:]:
            d = t.to_dict()
            d["strategy"] = s.symbol  # surface the market in the blotter
            trades.append(d)
    trades.sort(key=lambda d: d["exit_time"], reverse=True)

    sleeves = []
    for s in pr.sleeves:
        sm = s.metrics
        sleeves.append({
            "symbol": s.symbol,
            "start_capital": round(s.start_capital, 2),
            "final_equity": sm["final_equity"],
            "roi_pct": sm["roi_pct"],
            "sharpe": sm["sharpe"],
            "num_trades": sm["num_trades"],
            "win_rate_pct": sm["win_rate_pct"],
            "max_drawdown_pct": sm["max_drawdown_pct"],
            "equity_curve": [[t.isoformat(), round(e, 2)] for t, e in _downsample(s.equity_curve, 200)],
        })

    return {
        "mode": "portfolio",
        "generated_at": (generated_at or _now()).isoformat(),
        "symbol": "+".join(symbols) if symbols else "PORTFOLIO",
        "contract_name": f"{len(symbols)}-market basket ({pr.weighting}-weighted)",
        "exchange": "multi",
        "tradingview_symbol": first_tv,
        "strategy": pr.strategy,
        "bars_processed": len(pr.portfolio_equity),
        "last_price": round(curve[-1][1], 2) if curve else 0,
        "last_bar_time": curve[-1][0].isoformat() if curve else None,
        "account": {
            "starting_equity": round(pr.starting_cash, 2),
            "equity": m["final_equity"],
            "cash": m["final_equity"],
            "realized_pnl": m["net_profit"],
            "unrealized_pnl": 0.0,
            "roi_pct": m["roi_pct"],
        },
        "position": None,
        "metrics": m,
        "trades": trades[:_MAX_TRADES],
        "equity_curve": [[t.isoformat(), round(e, 2)] for t, e in curve],
        "events": [],
        "risk": {},
        "portfolio": {
            "weighting": pr.weighting,
            "avg_correlation": round(pr.avg_correlation, 3),
            "mean_sleeve_sharpe": round(pr.mean_sleeve_sharpe, 3),
            "diversification_ratio": round(pr.diversification_ratio, 3),
            "symbols": symbols,
            "correlation_matrix": pr.correlation_matrix,
            "sleeves": sleeves,
        },
    }


def write_state(path: str, state: dict) -> None:
    """Atomically write the state JSON (so the dashboard never reads a partial file)."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(state, fh, separators=(",", ":"))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def read_state(path: str) -> dict | None:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _downsample(curve, max_points: int):
    if len(curve) <= max_points:
        return curve
    step = len(curve) / max_points
    out = [curve[int(i * step)] for i in range(max_points)]
    out.append(curve[-1])
    return out


def _tv_symbol(spec: ContractSpec) -> str:
    """Best-effort TradingView symbol for the live chart widget.

    TradingView lists CME index futures as continuous contracts like
    ``CME_MINI:ES1!``. Falls back to the exchange-qualified root.
    """
    mapping = {
        "ES": "CME_MINI:ES1!",
        "MES": "CME_MINI:MES1!",
        "NQ": "CME_MINI:NQ1!",
        "MNQ": "CME_MINI:MNQ1!",
        "RTY": "CME_MINI:RTY1!",
        "M2K": "CME_MINI:M2K1!",
        "YM": "CBOT_MINI:YM1!",
        "MYM": "CBOT_MINI:MYM1!",
        "CL": "NYMEX:CL1!",
        "MCL": "NYMEX:MCL1!",
        "NG": "NYMEX:NG1!",
        "GC": "COMEX:GC1!",
        "MGC": "COMEX:MGC1!",
        "SI": "COMEX:SI1!",
        "ZN": "CBOT:ZN1!",
    }
    return mapping.get(spec.symbol, f"{spec.exchange}:{spec.symbol}1!")


def _now() -> datetime:
    return datetime.now(timezone.utc)
