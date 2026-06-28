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
