"""Position, Trade, and Portfolio — accounting for a single-symbol futures bot.

The portfolio holds at most one open position at a time (typical for a single
contract). It tracks realised PnL, marks unrealised PnL each bar, applies
commissions and slippage on fills, and records a full trade blotter plus an
equity curve for analytics and the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .contracts import ContractSpec


@dataclass
class Position:
    side: int                 # +1 long, -1 short
    quantity: int
    entry_price: float
    entry_time: datetime
    stop: float | None = None
    target: float | None = None
    strategy: str = ""
    reason: str = ""

    def unrealized(self, price: float, spec: ContractSpec) -> float:
        return (price - self.entry_price) * spec.point_value * self.side * self.quantity


@dataclass
class Trade:
    """A completed round-trip trade."""

    strategy: str
    side: str                 # "long" / "short"
    quantity: int
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    pnl: float                # net of costs
    commission: float
    entry_reason: str = ""
    exit_reason: str = ""

    @property
    def return_pct(self) -> float:
        denom = abs(self.entry_price) or 1e-9
        sign = 1 if self.side == "long" else -1
        return sign * (self.exit_price - self.entry_price) / denom

    @property
    def won(self) -> bool:
        return self.pnl > 0

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "side": self.side,
            "quantity": self.quantity,
            "entry_time": self.entry_time.isoformat(),
            "entry_price": round(self.entry_price, 4),
            "exit_time": self.exit_time.isoformat(),
            "exit_price": round(self.exit_price, 4),
            "pnl": round(self.pnl, 2),
            "commission": round(self.commission, 2),
            "return_pct": round(self.return_pct * 100, 3),
            "won": self.won,
            "entry_reason": self.entry_reason,
            "exit_reason": self.exit_reason,
        }


class Portfolio:
    """Cash + at most one open position, with full PnL accounting."""

    def __init__(
        self,
        spec: ContractSpec,
        starting_cash: float = 50_000.0,
        commission_per_contract: float = 2.50,
        slippage_ticks: float = 1.0,
    ) -> None:
        self.spec = spec
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.commission_per_contract = commission_per_contract
        self.slippage_ticks = slippage_ticks
        self.position: Position | None = None
        self.trades: list[Trade] = []
        self.equity_curve: list[tuple[datetime, float]] = []
        self._last_price: float = 0.0

    # --- marking & equity ---------------------------------------------
    def mark(self, time: datetime, price: float) -> float:
        self._last_price = price
        eq = self.equity(price)
        self.equity_curve.append((time, eq))
        return eq

    def equity(self, price: float | None = None) -> float:
        price = self._last_price if price is None else price
        unreal = self.position.unrealized(price, self.spec) if self.position else 0.0
        return self.cash + unreal

    # --- fills --------------------------------------------------------
    def _apply_slippage(self, price: float, side: int) -> float:
        """Worsen the fill by ``slippage_ticks`` in the adverse direction."""
        return price + side * self.slippage_ticks * self.spec.tick_size

    def open(
        self,
        side: int,
        quantity: int,
        price: float,
        time: datetime,
        *,
        stop: float | None = None,
        target: float | None = None,
        strategy: str = "",
        reason: str = "",
    ) -> None:
        if self.position is not None:
            raise RuntimeError("A position is already open; close it first.")
        if quantity <= 0:
            return
        fill = self._apply_slippage(price, side)
        # Commissions are charged round-turn at close so trade.pnl and cash
        # reconcile exactly; only mark-to-market moves equity while open.
        self.position = Position(
            side=side,
            quantity=quantity,
            entry_price=fill,
            entry_time=time,
            stop=stop,
            target=target,
            strategy=strategy,
            reason=reason,
        )

    def close(self, price: float, time: datetime, reason: str = "") -> Trade | None:
        if self.position is None:
            return None
        pos = self.position
        fill = self._apply_slippage(price, -pos.side)
        gross = (fill - pos.entry_price) * self.spec.point_value * pos.side * pos.quantity
        commission = self.commission_per_contract * pos.quantity * 2  # round turn
        net = gross - commission
        self.cash += net
        trade = Trade(
            strategy=pos.strategy,
            side="long" if pos.side > 0 else "short",
            quantity=pos.quantity,
            entry_time=pos.entry_time,
            entry_price=pos.entry_price,
            exit_time=time,
            exit_price=fill,
            pnl=net,
            commission=commission,
            entry_reason=pos.reason,
            exit_reason=reason,
        )
        self.trades.append(trade)
        self.position = None
        return trade

    # --- snapshots for the dashboard ----------------------------------
    @property
    def realized_pnl(self) -> float:
        return self.cash - self.starting_cash

    def open_position_dict(self, price: float | None = None) -> dict | None:
        if self.position is None:
            return None
        price = self._last_price if price is None else price
        pos = self.position
        return {
            "side": "long" if pos.side > 0 else "short",
            "quantity": pos.quantity,
            "entry_price": round(pos.entry_price, 4),
            "entry_time": pos.entry_time.isoformat(),
            "stop": round(pos.stop, 4) if pos.stop is not None else None,
            "target": round(pos.target, 4) if pos.target is not None else None,
            "price": round(price, 4),
            "unrealized": round(pos.unrealized(price, self.spec), 2),
            "strategy": pos.strategy,
            "reason": pos.reason,
        }
