"""Broker adapters.

``Broker`` is the execution interface. :class:`PaperBroker` is a fully
simulated account used for risk-free paper trading and tests. :class:`IBBroker`
is the Interactive Brokers adapter (via ``ib_insync``) — wire it in only when
you intend to trade a real (or IB paper) account.
"""

from .base import Broker, Fill, Order, OrderSide
from .paper import PaperBroker

__all__ = ["Broker", "Fill", "Order", "OrderSide", "PaperBroker"]
