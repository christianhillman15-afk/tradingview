"""Technical indicators implemented in pure-Python stdlib.

Every function operates on a sequence of floats (or :class:`~ai_futures_bot.data.Bar`
objects for the OHLC-aware indicators) and returns a list aligned to the input,
with ``None`` for warm-up positions where the indicator is not yet defined.

Keeping these dependency-free means the whole strategy/backtest core runs with
nothing but the standard library, and the functions are trivially unit-testable.
"""

from __future__ import annotations

import math
from typing import Sequence

Number = float


def sma(values: Sequence[float], period: int) -> list[float | None]:
    """Simple moving average."""
    if period <= 0:
        raise ValueError("period must be positive")
    out: list[float | None] = [None] * len(values)
    running = 0.0
    for i, v in enumerate(values):
        running += v
        if i >= period:
            running -= values[i - period]
        if i >= period - 1:
            out[i] = running / period
    return out


def ema(values: Sequence[float], period: int) -> list[float | None]:
    """Exponential moving average, seeded with the first SMA."""
    if period <= 0:
        raise ValueError("period must be positive")
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    k = 2.0 / (period + 1)
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(values)):
        prev = values[i] * k + prev * (1 - k)
        out[i] = prev
    return out


def stdev(values: Sequence[float], period: int) -> list[float | None]:
    """Rolling population standard deviation."""
    out: list[float | None] = [None] * len(values)
    for i in range(period - 1, len(values)):
        window = values[i - period + 1 : i + 1]
        mean = sum(window) / period
        var = sum((x - mean) ** 2 for x in window) / period
        out[i] = math.sqrt(var)
    return out


def rsi(values: Sequence[float], period: int = 14) -> list[float | None]:
    """Wilder's Relative Strength Index.

    RSI > 70 is conventionally "overbought", < 30 "oversold".
    """
    out: list[float | None] = [None] * len(values)
    if len(values) <= period:
        return out
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        change = values[i] - values[i - 1]
        gains += max(change, 0.0)
        losses += max(-change, 0.0)
    avg_gain = gains / period
    avg_loss = losses / period
    out[period] = _rsi_from(avg_gain, avg_loss)
    for i in range(period + 1, len(values)):
        change = values[i] - values[i - 1]
        gain = max(change, 0.0)
        loss = max(-change, 0.0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        out[i] = _rsi_from(avg_gain, avg_loss)
    return out


def _rsi_from(avg_gain: float, avg_loss: float) -> float:
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def macd(
    values: Sequence[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """MACD line, signal line, and histogram.

    Returns ``(macd_line, signal_line, histogram)``.
    """
    fast_ema = ema(values, fast)
    slow_ema = ema(values, slow)
    macd_line: list[float | None] = [
        (f - s) if (f is not None and s is not None) else None
        for f, s in zip(fast_ema, slow_ema)
    ]
    # Signal line is an EMA of the (defined portion of the) MACD line.
    defined = [m for m in macd_line if m is not None]
    start = next((i for i, m in enumerate(macd_line) if m is not None), len(macd_line))
    sig_defined = ema(defined, signal)
    signal_line: list[float | None] = [None] * len(values)
    for offset, val in enumerate(sig_defined):
        signal_line[start + offset] = val
    hist: list[float | None] = [
        (m - s) if (m is not None and s is not None) else None
        for m, s in zip(macd_line, signal_line)
    ]
    return macd_line, signal_line, hist


def bollinger_bands(
    values: Sequence[float],
    period: int = 20,
    num_std: float = 2.0,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    """Bollinger Bands. Returns ``(upper, middle, lower)``."""
    middle = sma(values, period)
    sd = stdev(values, period)
    upper: list[float | None] = [None] * len(values)
    lower: list[float | None] = [None] * len(values)
    for i in range(len(values)):
        if middle[i] is not None and sd[i] is not None:
            upper[i] = middle[i] + num_std * sd[i]
            lower[i] = middle[i] - num_std * sd[i]
    return upper, middle, lower


def true_range(highs: Sequence[float], lows: Sequence[float], closes: Sequence[float]) -> list[float]:
    """True Range series (first element uses high-low only)."""
    tr: list[float] = []
    for i in range(len(closes)):
        if i == 0:
            tr.append(highs[i] - lows[i])
        else:
            prev_close = closes[i - 1]
            tr.append(
                max(
                    highs[i] - lows[i],
                    abs(highs[i] - prev_close),
                    abs(lows[i] - prev_close),
                )
            )
    return tr


def atr(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int = 14,
) -> list[float | None]:
    """Average True Range (Wilder smoothing). The volatility unit used for
    position sizing and stop placement throughout the bot."""
    tr = true_range(highs, lows, closes)
    out: list[float | None] = [None] * len(closes)
    if len(tr) < period:
        return out
    prev = sum(tr[:period]) / period
    out[period - 1] = prev
    for i in range(period, len(tr)):
        prev = (prev * (period - 1) + tr[i]) / period
        out[i] = prev
    return out


def rolling_max(values: Sequence[float], period: int) -> list[float | None]:
    """Rolling maximum over ``period`` (the Donchian upper channel)."""
    out: list[float | None] = [None] * len(values)
    for i in range(period - 1, len(values)):
        out[i] = max(values[i - period + 1 : i + 1])
    return out


def rolling_min(values: Sequence[float], period: int) -> list[float | None]:
    """Rolling minimum over ``period`` (the Donchian lower channel)."""
    out: list[float | None] = [None] * len(values)
    for i in range(period - 1, len(values)):
        out[i] = min(values[i - period + 1 : i + 1])
    return out


def donchian_channels(
    highs: Sequence[float],
    lows: Sequence[float],
    period: int = 20,
) -> tuple[list[float | None], list[float | None]]:
    """Donchian channel. Returns ``(upper, lower)`` over the lookback.

    The core of the Turtle trend-following system: a breakout above ``upper``
    is a long entry, below ``lower`` a short entry.
    """
    return rolling_max(highs, period), rolling_min(lows, period)


def session_vwap(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    volumes: Sequence[float],
    session_ids: Sequence[object],
) -> list[float | None]:
    """Volume-Weighted Average Price that resets at each new session.

    ``session_ids`` is any sequence whose value changes when a new session
    starts (e.g. the trading date). VWAP is the intraday "fair value" anchor
    used by the VWAP mean-reversion strategy.
    """
    out: list[float | None] = [None] * len(closes)
    cum_pv = 0.0
    cum_vol = 0.0
    current: object = object()
    for i in range(len(closes)):
        if session_ids[i] != current:
            current = session_ids[i]
            cum_pv = 0.0
            cum_vol = 0.0
        typical = (highs[i] + lows[i] + closes[i]) / 3.0
        cum_pv += typical * volumes[i]
        cum_vol += volumes[i]
        out[i] = (cum_pv / cum_vol) if cum_vol > 0 else closes[i]
    return out


def crossed_above(series: Sequence[float | None], level: Sequence[float | None] | float, i: int) -> bool:
    """True if ``series`` crossed above ``level`` between bar ``i-1`` and ``i``."""
    if i <= 0:
        return False
    a_prev, a_now = series[i - 1], series[i]
    if a_prev is None or a_now is None:
        return False
    b_prev, b_now = _level_at(level, i - 1), _level_at(level, i)
    if b_prev is None or b_now is None:
        return False
    return a_prev <= b_prev and a_now > b_now


def crossed_below(series: Sequence[float | None], level: Sequence[float | None] | float, i: int) -> bool:
    """True if ``series`` crossed below ``level`` between bar ``i-1`` and ``i``."""
    if i <= 0:
        return False
    a_prev, a_now = series[i - 1], series[i]
    if a_prev is None or a_now is None:
        return False
    b_prev, b_now = _level_at(level, i - 1), _level_at(level, i)
    if b_prev is None or b_now is None:
        return False
    return a_prev >= b_prev and a_now < b_now


def _level_at(level: Sequence[float | None] | float, i: int) -> float | None:
    if isinstance(level, (int, float)):
        return float(level)
    return level[i]
