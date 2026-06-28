"""A self-contained simulated broker for paper trading and tests.

Wraps a :class:`~ai_futures_bot.portfolio.Portfolio` and fills market orders at
the supplied price hint (plus the portfolio's slippage model). It implements
the :class:`Broker` interface so the same calling code could later target a
real adapter.
"""

from __future__ import annotations

from datetime import datetime

from ..contracts import ContractSpec
from ..portfolio import Portfolio
from .base import Broker, Fill, Order, OrderSide


class PaperBroker(Broker):
    def __init__(
        self,
        spec: ContractSpec,
        starting_cash: float = 50_000.0,
        commission_per_contract: float = 2.50,
        slippage_ticks: float = 1.0,
    ) -> None:
        self._spec = spec
        self.portfolio = Portfolio(
            spec,
            starting_cash=starting_cash,
            commission_per_contract=commission_per_contract,
            slippage_ticks=slippage_ticks,
        )
        self._connected = False

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def submit_market_order(self, order: Order, price_hint: float, time: datetime) -> Fill:
        side = 1 if order.side == OrderSide.BUY else -1
        pos = self.portfolio.position
        # Reduce/close first if the order opposes the current position.
        if pos is not None and pos.side != side:
            self.portfolio.close(price_hint, time, reason="broker order")
        if self.portfolio.position is None and order.quantity > 0:
            self.portfolio.open(side, order.quantity, price_hint, time)
        return Fill(order=order, price=price_hint, quantity=order.quantity, time=time)

    def net_position(self) -> int:
        pos = self.portfolio.position
        return 0 if pos is None else pos.side * pos.quantity

    def account_equity(self, mark_price: float) -> float:
        return self.portfolio.equity(mark_price)

    @property
    def spec(self) -> ContractSpec:
        return self._spec
