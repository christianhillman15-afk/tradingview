"""Strategy base class and the :class:`Signal` it emits.

Design: the backtester (and live loop) calls :meth:`Strategy.prepare` once with
the full bar history to precompute indicator series, then calls
:meth:`Strategy.on_bar` for each bar index. This keeps per-bar work O(1) instead
of recomputing indicators on every call.

A strategy never sizes positions or tracks PnL — it only emits a *desired state*
(:class:`Signal`). The :class:`~ai_futures_bot.risk.RiskManager` decides how many
contracts to trade and the backtester/broker enforces stops and PnL. This clean
separation is what lets the same strategy run in backtest and live unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal, Sequence

from ..data import Bar

Action = Literal["long", "short", "exit"]


@dataclass
class Signal:
    """A desired position state emitted by a strategy.

    Attributes:
        action: ``"long"`` / ``"short"`` to enter or reverse into that side,
            ``"exit"`` to flatten.
        reason: Human-readable explanation (shown in logs / trade blotter).
        stop: Suggested protective stop price. If ``None`` the risk manager
            derives one from ATR.
        target: Suggested take-profit price (optional).
        strength: Confidence in ``[0, 1]``; may scale position size.
        trail_atr_mult: If set, the engine ratchets an ATR-based trailing stop at
            this multiple of ATR for the life of the trade (never loosening it).
    """

    action: Action
    reason: str = ""
    stop: float | None = None
    target: float | None = None
    strength: float = 1.0
    trail_atr_mult: float | None = None


class Strategy(ABC):
    """Base class for all trading strategies."""

    #: Human-readable strategy name (set by subclasses).
    name: str = "base"
    #: Behavioural family, used by the ensemble's regime filter:
    #: "trend" | "breakout" | "reversion" | "ml" | "other".
    category: str = "other"
    #: If True the backtester/live loop flattens any open position at the end
    #: of each trading session (no overnight risk).
    intraday: bool = False

    def __init__(self, **params: object) -> None:
        self.params = params
        self._bars: Sequence[Bar] = []

    # --- lifecycle -----------------------------------------------------
    def prepare(self, bars: Sequence[Bar]) -> None:
        """Precompute indicator series over the full history. Override to cache
        derived arrays, then call ``super().prepare(bars)`` (or set ``_bars``)."""
        self._bars = bars

    def reset(self) -> None:
        """Clear any per-run state (called before a fresh backtest)."""
        self._bars = []

    @abstractmethod
    def on_bar(self, i: int) -> Signal | None:
        """Return a :class:`Signal` for bar index ``i`` (or ``None`` for hold)."""
        raise NotImplementedError

    @abstractmethod
    def warmup(self) -> int:
        """Number of leading bars needed before signals are valid."""
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        params = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.__class__.__name__}({params})"
