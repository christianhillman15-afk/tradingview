"""Configuration loading.

A single :class:`Config` describes a run: which contract, which strategy (and
its parameters), account size, costs, risk limits, the data source, and the
dashboard endpoint. Loadable from YAML and overridable from the CLI. The
default starting balance is **$50,000** of paper money.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .risk import RiskConfig

try:
    import yaml  # PyYAML is available in the standard image; optional otherwise.
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class DataConfig:
    source: str = "synthetic"     # "synthetic" | "csv"
    csv_path: str | None = None
    timeframe_minutes: int = 60   # resample to this timeframe (1 = raw minutes)
    days: int = 400
    bars_per_day: int = 390
    seed: int = 42
    start_price: float = 5000.0
    annual_drift: float = 0.05
    annual_vol: float = 0.20


@dataclass
class DashboardConfig:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class Config:
    # Defaults: the flagship regime-aware ensemble on Micro E-mini S&P 500
    # (MES), the right contract size for a $50k account, on an hourly timeframe.
    # Intraday strategies (opening_range_breakout / vwap_reversion) should be run
    # with timeframe_minutes=1 instead.
    symbol: str = "MES"
    strategy: str = "ensemble"
    strategy_params: dict[str, Any] = field(default_factory=dict)
    starting_cash: float = 50_000.0
    commission_per_contract: float = 2.50
    slippage_ticks: float = 1.0
    state_path: str = "runtime/state.json"
    risk: RiskConfig = field(default_factory=RiskConfig)
    data: DataConfig = field(default_factory=DataConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        d = dict(d or {})
        risk = RiskConfig(**(d.pop("risk", {}) or {}))
        data = DataConfig(**(d.pop("data", {}) or {}))
        dash = DashboardConfig(**(d.pop("dashboard", {}) or {}))
        known = {f for f in cls.__dataclass_fields__ if f not in ("risk", "data", "dashboard")}
        kwargs = {k: v for k, v in d.items() if k in known}
        return cls(risk=risk, data=data, dashboard=dash, **kwargs)

    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        if yaml is None:  # pragma: no cover
            raise RuntimeError("PyYAML is required to read YAML config files.")
        with open(path) as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
