"""Market data: the :class:`Bar` model, CSV loading, and a synthetic generator.

The synthetic generator lets the whole pipeline run end-to-end with no external
data or network access — useful for demos, tests, and CI. Real data can be
loaded from CSV (or wired to a broker feed) using the same :class:`Bar` model.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, Iterator, Sequence


@dataclass(frozen=True)
class Bar:
    """A single OHLCV price bar."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def session_id(self) -> str:
        """Identifier that changes per trading day (used to reset VWAP)."""
        return self.timestamp.strftime("%Y-%m-%d")

    @property
    def typical(self) -> float:
        return (self.high + self.low + self.close) / 3.0


def opens(bars: Sequence[Bar]) -> list[float]:
    return [b.open for b in bars]


def highs(bars: Sequence[Bar]) -> list[float]:
    return [b.high for b in bars]


def lows(bars: Sequence[Bar]) -> list[float]:
    return [b.low for b in bars]


def closes(bars: Sequence[Bar]) -> list[float]:
    return [b.close for b in bars]


def volumes(bars: Sequence[Bar]) -> list[float]:
    return [b.volume for b in bars]


def session_ids(bars: Sequence[Bar]) -> list[str]:
    return [b.session_id for b in bars]


def atr_inputs(bars: Sequence[Bar]) -> tuple[list[float], list[float], list[float]]:
    """Return ``(highs, lows, closes)`` — the inputs ATR-based indicators need."""
    return highs(bars), lows(bars), closes(bars)


def load_csv(path: str, *, tz: timezone = timezone.utc) -> list[Bar]:
    """Load bars from a CSV with columns: timestamp,open,high,low,close,volume.

    ``timestamp`` may be an ISO-8601 string or a UNIX epoch (seconds). Extra
    columns are ignored; header is required.
    """
    bars: list[Bar] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required - {c.lower() for c in (reader.fieldnames or [])}
        if missing:
            raise ValueError(f"CSV missing columns: {sorted(missing)}")
        # Build a case-insensitive column map.
        for row in reader:
            r = {k.lower(): v for k, v in row.items()}
            bars.append(
                Bar(
                    timestamp=_parse_ts(r["timestamp"], tz),
                    open=float(r["open"]),
                    high=float(r["high"]),
                    low=float(r["low"]),
                    close=float(r["close"]),
                    volume=float(r["volume"]),
                )
            )
    bars.sort(key=lambda b: b.timestamp)
    return bars


def _parse_ts(raw: str, tz: timezone) -> datetime:
    raw = raw.strip()
    try:
        # Epoch seconds?
        return datetime.fromtimestamp(float(raw), tz)
    except ValueError:
        pass
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt


def write_csv(path: str, bars: Iterable[Bar]) -> None:
    """Write bars to a CSV (inverse of :func:`load_csv`)."""
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for b in bars:
            writer.writerow(
                [b.timestamp.isoformat(), b.open, b.high, b.low, b.close, b.volume]
            )


class SyntheticDataGenerator:
    """Deterministic synthetic intraday OHLCV generator.

    Produces realistic-looking minute bars during a regular trading session
    (default 9:30-16:00 ET, here represented in UTC for simplicity) using a
    seeded LCG random source so output is reproducible without numpy.

    The price path mixes a slow drift, a mean-reverting component, and
    intraday volatility clustering so trend, momentum, and mean-reversion
    strategies all have something to work with.
    """

    def __init__(
        self,
        *,
        seed: int = 42,
        start_price: float = 5000.0,
        annual_drift: float = 0.05,
        annual_vol: float = 0.20,
    ) -> None:
        self._state = seed & 0xFFFFFFFF
        self.start_price = start_price
        self.annual_drift = annual_drift
        self.annual_vol = annual_vol

    # --- Seeded RNG (LCG + Box-Muller) so we avoid a numpy dependency. ---
    def _rand(self) -> float:
        # Numerical Recipes LCG constants.
        self._state = (1664525 * self._state + 1013904223) & 0xFFFFFFFF
        return self._state / 0x100000000

    def _gauss(self) -> float:
        u1 = max(self._rand(), 1e-12)
        u2 = self._rand()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def generate(
        self,
        *,
        days: int = 30,
        bars_per_day: int = 390,
        start: datetime | None = None,
    ) -> list[Bar]:
        """Generate ``days`` sessions of ``bars_per_day`` one-minute bars."""
        if start is None:
            start = datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)  # ~9:30 ET
        minutes_per_year = 252 * bars_per_day
        mu = self.annual_drift / minutes_per_year
        sigma = self.annual_vol / math.sqrt(minutes_per_year)

        bars: list[Bar] = []
        price = self.start_price
        # Persistent multi-day trend ("regime") as an AR(1) process so the data
        # has both trending and ranging stretches, like a real market — not a
        # pure intraday mean-reverter. Tuned for realism, not for any strategy.
        regime = 0.0
        day = 0
        while day < days:
            session_start = start + timedelta(days=day)
            # Skip weekends to look like a real calendar.
            if session_start.weekday() >= 5:
                day += 1
                continue
            # Evolve the regime once per session (AR(1), stationary std ~0.9).
            regime = 0.92 * regime + 0.35 * self._gauss()
            # Modest daily drift (~0.4% per unit of regime) so intraday noise
            # still dominates — trends are present but far from clean.
            trend_per_min = (regime * 0.004) / bars_per_day
            session_anchor = price  # weak intraday pull toward the session open
            vol_scale = 1.0
            for minute in range(bars_per_day):
                ts = session_start + timedelta(minutes=minute)
                # Volatility clustering: smile shape (open/close busier).
                tod = minute / bars_per_day
                intraday_vol = 1.4 - 1.2 * math.sin(math.pi * tod)
                vol_scale = 0.9 * vol_scale + 0.1 * intraday_vol
                shock = self._gauss() * sigma * vol_scale
                reversion = 0.0008 * (session_anchor - price) / max(price, 1e-9)
                ret = mu + trend_per_min + shock + reversion
                new_price = price * math.exp(ret)
                # Build an OHLC bar around the open->close move.
                o = price
                c = new_price
                wick = abs(self._gauss()) * sigma * vol_scale * price * 0.6
                hi = max(o, c) + wick
                lo = min(o, c) - wick
                base_vol = 1500 + 4000 * intraday_vol
                vol = base_vol * (0.7 + 0.6 * self._rand())
                bars.append(
                    Bar(
                        timestamp=ts,
                        open=round(o, 2),
                        high=round(hi, 2),
                        low=round(lo, 2),
                        close=round(c, 2),
                        volume=round(vol),
                    )
                )
                price = new_price
            day += 1
        return bars


def stream(bars: Iterable[Bar]) -> Iterator[Bar]:
    """Yield bars one at a time (the live/backtest event loop interface)."""
    yield from bars
