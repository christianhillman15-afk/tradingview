"""Interactive Brokers adapter (via ib_insync) — real-execution surface.

This is a thin, intentionally conservative adapter. It is NOT exercised by the
default paper workflow; wire it in only when you deliberately want to route
orders to an IB (paper or live) account through Trader Workstation (TWS) or the
IB Gateway.

Prerequisites:
  * ``pip install ib_insync``
  * TWS or IB Gateway running with the API enabled.
  * Connect to the **paper** port (7497) until you have validated everything.

Safety: trading real money is entirely at your own risk. Start on the IB paper
account, keep size tiny, and verify every fill against the simulator first.
"""

from __future__ import annotations

from datetime import datetime, timezone

from ..contracts import ContractSpec
from .base import Broker, Fill, Order, OrderSide


class IBBroker(Broker):
    def __init__(
        self,
        spec: ContractSpec,
        host: str = "127.0.0.1",
        port: int = 7497,            # 7497 = paper, 7496 = live (be careful!)
        client_id: int = 1,
        expiry: str | None = None,   # e.g. "202503" for a specific contract month
    ) -> None:
        self._spec = spec
        self.host = host
        self.port = port
        self.client_id = client_id
        self.expiry = expiry
        self._ib = None
        self._contract = None

    def _require_ib(self):
        try:
            import ib_insync  # noqa: F401
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "IBBroker needs ib_insync. Install with: pip install ib_insync\n"
                "and run TWS or IB Gateway with the API enabled."
            ) from exc
        return ib_insync

    def connect(self) -> None:
        ib_insync = self._require_ib()
        self._ib = ib_insync.IB()
        self._ib.connect(self.host, self.port, clientId=self.client_id)
        contract = ib_insync.Future(
            symbol=self._spec.symbol,
            exchange=self._spec.exchange,
            currency=self._spec.currency,
            lastTradeDateOrContractMonth=self.expiry or "",
        )
        # Resolve the front-month / specified contract.
        details = self._ib.reqContractDetails(contract)
        if not details:
            raise RuntimeError(f"No IB contract found for {self._spec.symbol}.")
        self._contract = details[0].contract

    def disconnect(self) -> None:
        if self._ib is not None:
            self._ib.disconnect()
            self._ib = None

    def submit_market_order(self, order: Order, price_hint: float, time: datetime) -> Fill:
        ib_insync = self._require_ib()
        if self._ib is None or self._contract is None:
            raise RuntimeError("IBBroker is not connected. Call connect() first.")
        action = "BUY" if order.side == OrderSide.BUY else "SELL"
        ib_order = ib_insync.MarketOrder(action, order.quantity)
        trade = self._ib.placeOrder(self._contract, ib_order)
        self._ib.sleep(1.0)  # allow the fill to come back
        avg = trade.orderStatus.avgFillPrice or price_hint
        commission = sum(
            f.commissionReport.commission
            for f in trade.fills
            if f.commissionReport
        ) if trade.fills else 0.0
        return Fill(
            order=order,
            price=float(avg),
            quantity=order.quantity,
            time=datetime.now(timezone.utc),
            commission=float(commission),
        )

    def net_position(self) -> int:
        if self._ib is None or self._contract is None:
            return 0
        for pos in self._ib.positions():
            if pos.contract.conId == self._contract.conId:
                return int(pos.position)
        return 0

    def account_equity(self, mark_price: float) -> float:
        if self._ib is None:
            return 0.0
        for v in self._ib.accountValues():
            if v.tag == "NetLiquidation":
                return float(v.value)
        return 0.0

    @property
    def spec(self) -> ContractSpec:
        return self._spec
