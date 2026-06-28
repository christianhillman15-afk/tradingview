import pytest

from ai_futures_bot.realdata import parse_yahoo_chart, yahoo_ticker


def test_yahoo_ticker_mapping():
    assert yahoo_ticker("ES") == "ES=F"
    assert yahoo_ticker("mes") == "MES=F"
    assert yahoo_ticker("6E") == "6E=F"
    assert yahoo_ticker("ZC") == "ZC=F"
    assert yahoo_ticker("WTF") == "WTF=F"  # unknown -> default pattern


def _payload(opens):
    return {
        "chart": {
            "error": None,
            "result": [
                {
                    "timestamp": [1704153600, 1704240000, 1704326400],
                    "indicators": {
                        "quote": [
                            {
                                "open": opens,
                                "high": [4820.0, 4830.0, 4840.0],
                                "low": [4790.0, 4800.0, 4810.0],
                                "close": [4815.0, 4825.0, 4835.0],
                                "volume": [100000, 120000, 130000],
                            }
                        ]
                    },
                }
            ],
        }
    }


def test_parse_yahoo_chart_basic():
    bars = parse_yahoo_chart(_payload([4800.0, 4810.0, 4820.0]))
    assert len(bars) == 3
    assert bars[0].open == 4800.0 and bars[0].close == 4815.0
    assert bars[-1].high == 4840.0
    # sorted ascending by time
    assert bars[0].timestamp < bars[1].timestamp < bars[2].timestamp


def test_parse_skips_null_candles():
    bars = parse_yahoo_chart(_payload([4800.0, None, 4820.0]))
    assert len(bars) == 2  # the null-open candle is dropped


def test_parse_raises_on_error_or_empty():
    with pytest.raises(ValueError):
        parse_yahoo_chart({"chart": {"error": {"code": "Not Found"}, "result": None}})
    with pytest.raises(ValueError):
        parse_yahoo_chart({"chart": {"error": None, "result": []}})
