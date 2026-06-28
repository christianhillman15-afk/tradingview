"""Feature engineering for the ML strategy.

Turns a bar history into a numeric feature matrix using the same technical
indicators the rule-based strategies use (returns, RSI, MACD histogram,
Bollinger position, ATR-normalised range, momentum, volume ratio, distance
from moving averages). Built in pure Python — only the model itself needs
numpy/scikit-learn.

The label is the sign of the forward return over ``horizon`` bars, with a
dead-band so tiny moves are excluded from training.
"""

from __future__ import annotations

from typing import Sequence

from ..data import Bar, atr_inputs, closes, volumes
from ..indicators import atr, bollinger_bands, ema, macd, rsi, sma

FEATURE_NAMES = [
    "ret_1",
    "ret_5",
    "ret_15",
    "rsi_14",
    "macd_hist",
    "bb_pos",
    "atr_pct",
    "mom_10",
    "dist_ema_20",
    "dist_ema_50",
    "vol_ratio",
]


def build_features(bars: Sequence[Bar]) -> tuple[list[list[float]], list[bool]]:
    """Return ``(feature_rows, valid_mask)`` aligned to ``bars``.

    ``valid_mask[i]`` is False during indicator warm-up where features are not
    fully defined; callers should skip those rows.
    """
    c = closes(bars)
    v = volumes(bars)
    h, l, cc = atr_inputs(bars)
    n = len(bars)

    rsi14 = rsi(c, 14)
    _, _, hist = macd(c)
    upper, mid, lower = bollinger_bands(c, 20, 2.0)
    atr14 = atr(h, l, cc, 14)
    ema20 = ema(c, 20)
    ema50 = ema(c, 50)
    vol_sma = sma(v, 20)

    rows: list[list[float]] = []
    valid: list[bool] = []
    for i in range(n):
        ok = (
            i >= 50
            and rsi14[i] is not None
            and hist[i] is not None
            and upper[i] is not None
            and atr14[i] is not None
            and ema20[i] is not None
            and ema50[i] is not None
            and vol_sma[i] is not None
            and vol_sma[i] > 0
            and c[i] > 0
        )
        if not ok:
            rows.append([0.0] * len(FEATURE_NAMES))
            valid.append(False)
            continue
        band = (upper[i] - lower[i]) or 1e-9
        row = [
            _ret(c, i, 1),
            _ret(c, i, 5),
            _ret(c, i, 15),
            rsi14[i] / 100.0,
            hist[i] / c[i],
            (c[i] - lower[i]) / band,           # 0=lower band, 1=upper band
            atr14[i] / c[i],
            _ret(c, i, 10),
            (c[i] - ema20[i]) / c[i],
            (c[i] - ema50[i]) / c[i],
            v[i] / vol_sma[i],
        ]
        rows.append(row)
        valid.append(True)
    return rows, valid


def build_labels(bars: Sequence[Bar], horizon: int = 10, deadband: float = 0.0005) -> list[int | None]:
    """Forward-return classification labels: 1 (up), 0 (down), or ``None``.

    ``None`` marks rows with no future bar or a move inside the dead-band.
    """
    c = closes(bars)
    n = len(bars)
    labels: list[int | None] = []
    for i in range(n):
        j = i + horizon
        if j >= n:
            labels.append(None)
            continue
        fwd = (c[j] - c[i]) / c[i]
        if abs(fwd) < deadband:
            labels.append(None)
        else:
            labels.append(1 if fwd > 0 else 0)
    return labels


def _ret(c: Sequence[float], i: int, lag: int) -> float:
    if i - lag < 0 or c[i - lag] == 0:
        return 0.0
    return (c[i] - c[i - lag]) / c[i - lag]
