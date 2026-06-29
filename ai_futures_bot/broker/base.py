"""Broker execution interface and order/fill value types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from ..contracts import ContractSpec


class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    side: OrderSide
    quantity: int
    symbol: str
    type: str = "market"
    limit_price: float | None = None


@dataclass
class Fill:
    order: Order
    price: float
    quantity: int
    time: datetime
    commission: float = 0.0


class Broker(ABC):
    """Minimal execution interface shared by paper and live adapters."""

    @abstractmethod
    def connect(self) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def submit_market_order(self, order: Order, price_hint: float, time: datetime) -> Fill: ...

    @abstractmethod
    def net_position(self) -> int:
        """Signed contract count (+long / -short / 0 flat)."""

    @abstractmethod
    def account_equity(self, mark_price: float) -> float: ...

    @property
    @abstractmethod
    def spec(self) -> ContractSpec: ...
