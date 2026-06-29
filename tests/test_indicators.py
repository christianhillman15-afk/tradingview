from datetime import datetime, timezone

from ai_futures_bot import indicators as ind


def test_sma_basic():
    out = ind.sma([1, 2, 3, 4, 5], 3)
    assert out[:2] == [None, None]
    assert out[2] == 2.0 and out[3] == 3.0 and out[4] == 4.0


def test_ema_defined_after_period():
    out = ind.ema([1, 2, 3, 4, 5, 6], 3)
    assert out[0] is None and out[1] is None
    assert out[2] is not None
    assert out[-1] > out[2]  # rising series


def test_rsi_bounds_and_extremes():
    rising = list(range(1, 40))
    out = ind.rsi(rising, 14)
    last = out[-1]
    assert last is not None and 99.0 <= last <= 100.0  # all gains -> ~100
    for v in out:
        assert v is None or 0.0 <= v <= 100.0


def test_atr_positive():
    highs = [10, 11, 12, 13, 14, 15]
    lows = [9, 9.5, 10, 11, 12, 13]
    closes = [9.5, 10.5, 11, 12, 13, 14]
    out = ind.atr(highs, lows, closes, 3)
    assert any(v is not None for v in out)
    assert all(v is None or v >= 0 for v in out)


def test_donchian_channels():
    highs = [5, 6, 7, 4, 8]
    lows = [1, 2, 1, 0, 3]
    up, lo = ind.donchian_channels(highs, lows, 3)
    assert up[2] == 7 and lo[2] == 1
    assert up[4] == 8 and lo[4] == 0


def test_session_vwap_resets():
    h = [10, 10, 20, 20]
    l = [10, 10, 20, 20]
    c = [10, 10, 20, 20]
    v = [1, 1, 1, 1]
    sids = ["d1", "d1", "d2", "d2"]
    out = ind.session_vwap(h, l, c, v, sids)
    assert out[1] == 10.0          # day 1 vwap
    assert out[2] == 20.0          # day 2 resets to its own price
    assert out[3] == 20.0


def test_crossover_helpers():
    a = [1, 2, 3, 2, 1]
    b = [2, 2, 2, 2, 2]
    assert ind.crossed_above(a, b, 2) is True   # 2 -> 3 crosses up through 2
    assert ind.crossed_below(a, b, 3) is False  # 3 -> 2 only touches the level
    assert ind.crossed_below(a, b, 4) is True   # 2 -> 1 completes the cross down
    assert ind.crossed_above(a, b, 1) is False


def test_macd_shapes():
    values = [float(x) for x in range(1, 60)]
    macd_line, signal, hist = ind.macd(values)
    assert len(macd_line) == len(values) == len(signal) == len(hist)
    assert macd_line[-1] is not None and signal[-1] is not None
