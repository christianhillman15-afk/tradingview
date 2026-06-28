"""Strategy registry.

Each strategy registers under a short name so it can be selected from config or
the CLI (``--strategy donchian_trend``). New strategies just need to subclass
:class:`~ai_futures_bot.strategies.base.Strategy` and be added to ``_REGISTRY``.
"""

from __future__ import annotations

from typing import Type

from .base import Signal, Strategy
from .bollinger_reversion import BollingerReversionStrategy
from .bollinger_squeeze import BollingerSqueezeStrategy
from .donchian_trend import DonchianTrendStrategy
from .ensemble import EnsembleStrategy
from .macd_momentum import MacdMomentumStrategy
from .ml_ensemble import MLEnsembleStrategy
from .opening_range_breakout import OpeningRangeBreakoutStrategy
from .supertrend import SupertrendStrategy
from .tsmom import TimeSeriesMomentumStrategy
from .vwap_reversion import VwapReversionStrategy
from .zscore_reversion import ZScoreReversionStrategy

_REGISTRY: dict[str, Type[Strategy]] = {
    cls.name: cls
    for cls in [
        OpeningRangeBreakoutStrategy,
        VwapReversionStrategy,
        DonchianTrendStrategy,
        MacdMomentumStrategy,
        BollingerReversionStrategy,
        SupertrendStrategy,
        BollingerSqueezeStrategy,
        ZScoreReversionStrategy,
        TimeSeriesMomentumStrategy,
        EnsembleStrategy,
        MLEnsembleStrategy,
    ]
}


def get_strategy(name: str, **params: object) -> Strategy:
    """Instantiate a strategy by name with optional parameter overrides."""
    key = name.lower().strip()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown strategy {name!r}. Available: {available}")
    return _REGISTRY[key](**params)


def list_strategies() -> list[str]:
    """Return the names of all registered strategies."""
    return sorted(_REGISTRY)


def strategy_class(name: str) -> Type[Strategy]:
    return _REGISTRY[name.lower().strip()]


__all__ = [
    "Signal",
    "Strategy",
    "get_strategy",
    "list_strategies",
    "strategy_class",
    "OpeningRangeBreakoutStrategy",
    "VwapReversionStrategy",
    "DonchianTrendStrategy",
    "MacdMomentumStrategy",
    "BollingerReversionStrategy",
    "SupertrendStrategy",
    "BollingerSqueezeStrategy",
    "ZScoreReversionStrategy",
    "TimeSeriesMomentumStrategy",
    "EnsembleStrategy",
    "MLEnsembleStrategy",
]
