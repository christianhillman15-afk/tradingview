"""Risk management: volatility-based position sizing and a hard kill switch.

Implements the professional risk rules surfaced in the research:
  * Risk a fixed small fraction of equity per trade (default 1%).
  * Size positions from the **stop distance**, not from available margin —
    ``contracts = floor(risk_dollars / (stop_points * point_value))``.
  * A daily-loss "kill switch": once realised losses for the session exceed
    ``daily_loss_limit`` of start-of-day equity, no new trades that session.
  * A global max-drawdown stop that halts all trading if equity falls too far
    from its peak.
  * Caps on contract count and margin utilisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .contracts import ContractSpec


@dataclass
class RiskConfig:
    risk_per_trade: float = 0.01          # fraction of equity risked per trade
    max_contracts: int = 50               # hard cap on position size
    max_margin_fraction: float = 0.50     # cap on equity tied up in margin
    daily_loss_limit: float = 0.03        # halt for the day past this loss frac
    max_drawdown_stop: float = 0.20       # halt entirely past this drawdown
    scale_by_strength: bool = False       # multiply risk by signal.strength
    min_stop_points: float = 0.0          # floor on stop distance (0 = none)


@dataclass
class RiskManager:
    """Stateful risk manager. One instance per trading run."""

    config: RiskConfig = field(default_factory=RiskConfig)
    _equity_peak: float = 0.0
    _day: str | None = None
    _day_start_equity: float = 0.0
    _day_realized: float = 0.0
    halted: bool = False

    def position_size(
        self,
        equity: float,
        entry_price: float,
        stop_price: float,
        spec: ContractSpec,
        strength: float = 1.0,
    ) -> int:
        """Number of contracts to trade for this setup (0 = skip)."""
        stop_points = abs(entry_price - stop_price)
        if self.config.min_stop_points:
            stop_points = max(stop_points, self.config.min_stop_points)
        if stop_points <= 0 or equity <= 0:
            return 0

        risk_dollars = equity * self.config.risk_per_trade
        if self.config.scale_by_strength:
            risk_dollars *= max(0.0, min(1.0, strength))

        per_contract_risk = stop_points * spec.point_value
        qty = int(math.floor(risk_dollars / per_contract_risk))

        # Cap by margin utilisation.
        if spec.initial_margin > 0:
            max_by_margin = int(
                math.floor(equity * self.config.max_margin_fraction / spec.initial_margin)
            )
            qty = min(qty, max_by_margin)

        qty = min(qty, self.config.max_contracts)
        return max(qty, 0)

    # --- session / drawdown gating ------------------------------------
    def start_bar(self, day: str, equity: float) -> None:
        """Update peak equity and roll daily counters at each new session."""
        if equity > self._equity_peak:
            self._equity_peak = equity
        if self._day != day:
            self._day = day
            self._day_start_equity = equity
            self._day_realized = 0.0

        # Global drawdown kill switch.
        if self._equity_peak > 0:
            drawdown = (self._equity_peak - equity) / self._equity_peak
            if drawdown >= self.config.max_drawdown_stop:
                self.halted = True

    def register_trade(self, realized_pnl: float) -> None:
        """Record a closed trade's realised PnL against the daily counter."""
        self._day_realized += realized_pnl

    def can_trade(self) -> bool:
        """Whether a *new* position may be opened right now."""
        if self.halted:
            return False
        if self._day_start_equity <= 0:
            return True
        loss_frac = -self._day_realized / self._day_start_equity
        return loss_frac < self.config.daily_loss_limit

    @property
    def equity_peak(self) -> float:
        return self._equity_peak

    @property
    def day_realized(self) -> float:
        return self._day_realized
