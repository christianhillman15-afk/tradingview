"""Futures contract specifications.

A futures contract's economics are defined by its *tick size* (the minimum
price increment) and *tick value* (the dollar value of one tick). Profit/loss
in dollars is computed from price moves using these specs, so getting them
right is essential for correct position sizing and PnL.

Values below reflect CME specifications at the time of writing. Always verify
against the exchange before trading real money — contract specs change.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ContractSpec:
    """Specification for a single futures contract.

    Attributes:
        symbol: Root symbol (e.g. ``"ES"``).
        name: Human-readable name.
        exchange: Listing exchange (e.g. ``"CME"``).
        tick_size: Minimum price increment (e.g. ``0.25`` index points for ES).
        tick_value: Dollar value of one tick move for one contract.
        currency: Quote currency.
        initial_margin: Approximate exchange initial margin per contract (USD).
            Indicative only; brokers set their own day-trade margins.
        point_value: Dollar value of a full 1.0 move in price for one contract.
    """

    symbol: str
    name: str
    exchange: str
    tick_size: float
    tick_value: float
    currency: str = "USD"
    initial_margin: float = 0.0

    @property
    def point_value(self) -> float:
        """Dollar value of a 1.0 price move for one contract."""
        return self.tick_value / self.tick_size

    def round_to_tick(self, price: float) -> float:
        """Round a raw price to the nearest valid tick."""
        ticks = round(price / self.tick_size)
        return round(ticks * self.tick_size, 10)

    def pnl(self, entry: float, exit_: float, quantity: int) -> float:
        """Dollar PnL for ``quantity`` contracts (signed: +long, -short)."""
        return (exit_ - entry) * self.point_value * quantity


# Registry of commonly traded index, energy, metal, and rate futures.
# Tick/point values are standard CME contract specs.
_REGISTRY: dict[str, ContractSpec] = {
    spec.symbol: spec
    for spec in [
        # --- Equity index ---
        ContractSpec("ES", "E-mini S&P 500", "CME", 0.25, 12.50, initial_margin=13_200),
        ContractSpec("MES", "Micro E-mini S&P 500", "CME", 0.25, 1.25, initial_margin=1_320),
        ContractSpec("NQ", "E-mini Nasdaq-100", "CME", 0.25, 5.00, initial_margin=22_000),
        ContractSpec("MNQ", "Micro E-mini Nasdaq-100", "CME", 0.25, 0.50, initial_margin=2_200),
        ContractSpec("YM", "E-mini Dow", "CBOT", 1.0, 5.00, initial_margin=11_000),
        ContractSpec("MYM", "Micro E-mini Dow", "CBOT", 1.0, 0.50, initial_margin=1_100),
        ContractSpec("RTY", "E-mini Russell 2000", "CME", 0.10, 5.00, initial_margin=8_000),
        ContractSpec("M2K", "Micro E-mini Russell 2000", "CME", 0.10, 0.50, initial_margin=800),
        # --- Energy ---
        ContractSpec("CL", "Crude Oil WTI", "NYMEX", 0.01, 10.00, initial_margin=6_500),
        ContractSpec("MCL", "Micro WTI Crude Oil", "NYMEX", 0.01, 1.00, initial_margin=650),
        ContractSpec("NG", "Natural Gas", "NYMEX", 0.001, 10.00, initial_margin=3_500),
        # --- Metals ---
        ContractSpec("GC", "Gold", "COMEX", 0.10, 10.00, initial_margin=11_000),
        ContractSpec("MGC", "Micro Gold", "COMEX", 0.10, 1.00, initial_margin=1_100),
        ContractSpec("SI", "Silver", "COMEX", 0.005, 25.00, initial_margin=14_000),
        # --- Rates ---
        ContractSpec("ZN", "10-Year T-Note", "CBOT", 0.015625, 15.625, initial_margin=2_000),
    ]
}


def get_contract(symbol: str) -> ContractSpec:
    """Look up a contract spec by root symbol (case-insensitive).

    Raises:
        KeyError: if the symbol is not in the registry.
    """
    key = symbol.upper().strip()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown contract {symbol!r}. Available: {available}")
    return _REGISTRY[key]


def register_contract(spec: ContractSpec) -> None:
    """Add or override a contract spec in the registry."""
    _REGISTRY[spec.symbol.upper()] = spec


def list_contracts() -> list[ContractSpec]:
    """Return all registered contract specs, sorted by symbol."""
    return [_REGISTRY[k] for k in sorted(_REGISTRY)]
