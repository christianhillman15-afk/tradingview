"""Regime-aware multi-strategy ensemble — the bot's "best" composite.

Why this exists: markets only trend ~30% of the time. No single strategy wins
in every regime, so professionals diversify across uncorrelated strategies and
turn each on only when its regime is favourable. This meta-strategy does exactly
that:

  * It runs several member strategies in parallel as *opinion generators*, each
    maintaining its own desired position (long/short/flat) from its own signals.
  * An ADX + efficiency-ratio **regime filter** classifies each bar as trending,
    ranging, or neutral.
  * In a **trending** regime only trend/breakout/ml members vote; in a
    **ranging** regime only reversion members vote. Votes are weighted and
    netted to a single ensemble position.
  * Entries fire when the (regime-gated) weighted net conviction clears a
    threshold; the position flattens when conviction collapses toward zero.

Defaults combine three trend members and two reversion members, so the ensemble
always has something appropriate for the current regime. ``category = "other"``.
"""

from __future__ import annotations

from typing import Sequence

from ..data import Bar, atr_inputs, closes
from ..indicators import adx, atr, efficiency_ratio
from .base import Signal, Strategy

_DEFAULT_MEMBERS = [
    {"name": "supertrend", "weight": 1.0},
    {"name": "donchian_trend", "weight": 1.0},
    {"name": "macd_momentum", "weight": 1.0},
    {"name": "bollinger_reversion", "weight": 1.0},
    {"name": "zscore_reversion", "weight": 1.0},
]


class EnsembleStrategy(Strategy):
    name = "ensemble"
    category = "other"
    intraday = False

    def __init__(
        self,
        members: list[dict] | None = None,
        trend_adx: float = 25.0,
        range_adx: float = 20.0,
        enter_threshold: float = 0.5,
        exit_threshold: float = 0.2,
        adx_period: int = 14,
        atr_period: int = 14,
        atr_stop_mult: float = 2.0,
        trail_atr_mult: float = 3.0,
    ) -> None:
        super().__init__(
            members=members,
            trend_adx=trend_adx,
            range_adx=range_adx,
            enter_threshold=enter_threshold,
            exit_threshold=exit_threshold,
            adx_period=adx_period,
            atr_period=atr_period,
            atr_stop_mult=atr_stop_mult,
            trail_atr_mult=trail_atr_mult,
        )
        self.member_specs = members or _DEFAULT_MEMBERS
        self.trend_adx = trend_adx
        self.range_adx = range_adx
        self.enter_threshold = enter_threshold
        self.exit_threshold = exit_threshold
        self.adx_period = adx_period
        self.atr_period = atr_period
        self.atr_stop_mult = atr_stop_mult
        self.trail_atr_mult = trail_atr_mult
        self._members: list[Strategy] = []
        self._weights: list[float] = []
        self._sub_state: list[int] = []

    def prepare(self, bars: Sequence[Bar]) -> None:
        super().prepare(bars)
        # Lazy import to avoid a circular import with the registry.
        from . import get_strategy

        self._members = []
        self._weights = []
        for spec in self.member_specs:
            params = {k: v for k, v in spec.items() if k not in ("name", "weight")}
            member = get_strategy(spec["name"], **params)
            member.prepare(bars)
            self._members.append(member)
            self._weights.append(float(spec.get("weight", 1.0)))
        self._sub_state = [0] * len(self._members)

        h, l, c = atr_inputs(bars)
        self._adx, _, _ = adx(h, l, c, self.adx_period)
        self._eff = efficiency_ratio(closes(bars), 10)
        self._atr = atr(h, l, c, self.atr_period)

    def reset(self) -> None:
        super().reset()
        for m in self._members:
            m.reset()
        self._sub_state = [0] * len(self._members)

    def warmup(self) -> int:
        member_warm = max((m.warmup() for m in self._members), default=0) if self._members else 0
        return max(member_warm, self.adx_period * 2) + 1

    def _regime(self, i: int) -> str:
        a = self._adx[i]
        e = self._eff[i]
        if a is None:
            return "neutral"
        eff_trend = (e is not None and e >= 0.4)
        if a >= self.trend_adx or eff_trend:
            return "trending"
        if a < self.range_adx:
            return "ranging"
        return "neutral"

    def _enabled(self, category: str, regime: str) -> bool:
        if regime == "trending":
            return category in ("trend", "breakout", "ml")
        if regime == "ranging":
            return category == "reversion"
        return False  # neutral regime: no new conviction

    def on_bar(self, i: int) -> Signal | None:
        # Always advance member state so each keeps its own running opinion.
        for k, member in enumerate(self._members):
            sig = member.on_bar(i)
            if sig is not None:
                if sig.action == "long":
                    self._sub_state[k] = 1
                elif sig.action == "short":
                    self._sub_state[k] = -1
                elif sig.action == "exit":
                    self._sub_state[k] = 0

        if i < self.warmup():
            return None
        a = self._atr[i]
        if a is None:
            return None

        regime = self._regime(i)
        net = 0.0
        total = 0.0
        for k, member in enumerate(self._members):
            if not self._enabled(member.category, regime):
                continue
            total += self._weights[k]
            net += self._weights[k] * self._sub_state[k]
        if total <= 0:
            # Nothing votes this regime; collapse conviction -> allow an exit.
            return Signal("exit", reason=f"{regime}: no active members") if regime == "neutral" else None

        conviction = net / total  # in [-1, 1]
        bar = self._bars[i]
        if conviction >= self.enter_threshold:
            return Signal(
                "long",
                reason=f"{regime} conviction {conviction:+.2f}",
                stop=bar.close - self.atr_stop_mult * a,
                trail_atr_mult=self.trail_atr_mult,
            )
        if conviction <= -self.enter_threshold:
            return Signal(
                "short",
                reason=f"{regime} conviction {conviction:+.2f}",
                stop=bar.close + self.atr_stop_mult * a,
                trail_atr_mult=self.trail_atr_mult,
            )
        if abs(conviction) <= self.exit_threshold:
            return Signal("exit", reason=f"{regime} conviction faded {conviction:+.2f}")
        return None
