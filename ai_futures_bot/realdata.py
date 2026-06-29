"""Real market-data ingestion (free sources, stdlib only).

Fetches historical OHLCV bars from the Yahoo Finance chart API and converts them
into :class:`~ai_futures_bot.data.Bar` objects compatible with the rest of the
bot. Save to CSV with the ``fetch`` CLI command, then backtest with ``--csv``.

Notes:
  * Uses only the standard library (urllib/json) and respects the system HTTPS
    proxy and CA bundle via the default SSL context.
  * Free endpoints are rate-limited and occasionally block shared/cloud IPs; if
    a fetch fails, retry later or run it from your own machine. The parsing is
    fully unit-tested offline so the code is known-good regardless of network.
"""

from __future__ import annotations

import json
import ssl
import urllib.request
from datetime import datetime, timezone

from .data import Bar

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# CME/ICE root -> Yahoo continuous-future ticker. Most roots are ROOT=F; the
# exceptions / common ones are listed explicitly.
_YAHOO_OVERRIDES = {
    "ES": "ES=F", "MES": "MES=F", "NQ": "NQ=F", "MNQ": "MNQ=F", "YM": "YM=F",
    "MYM": "MYM=F", "RTY": "RTY=F", "M2K": "M2K=F", "CL": "CL=F", "MCL": "MCL=F",
    "BZ": "BZ=F", "NG": "NG=F", "RB": "RB=F", "HO": "HO=F", "GC": "GC=F",
    "MGC": "MGC=F", "SI": "SI=F", "HG": "HG=F", "PL": "PL=F", "PA": "PA=F",
    "ZN": "ZN=F", "ZB": "ZB=F", "ZF": "ZF=F", "ZT": "ZT=F", "ZC": "ZC=F",
    "ZS": "ZS=F", "ZW": "ZW=F", "ZL": "ZL=F", "ZM": "ZM=F", "LE": "LE=F",
    "GF": "GF=F", "HE": "HE=F", "KC": "KC=F", "SB": "SB=F", "CC": "CC=F",
    "CT": "CT=F", "OJ": "OJ=F", "6E": "6E=F", "6J": "6J=F", "6B": "6B=F",
    "6A": "6A=F", "6C": "6C=F", "6S": "6S=F", "BTC": "BTC=F", "MBT": "MBT=F",
}


def yahoo_ticker(symbol: str) -> str:
    """Map a contract root symbol to a Yahoo continuous-future ticker."""
    s = symbol.upper().strip()
    return _YAHOO_OVERRIDES.get(s, f"{s}=F")


def parse_yahoo_chart(payload: dict, *, tz: timezone = timezone.utc) -> list[Bar]:
    """Parse a Yahoo Finance chart-API JSON payload into bars.

    Tolerates the nulls Yahoo sprinkles into incomplete candles by skipping any
    bar with a missing OHLC value.
    """
    chart = payload.get("chart") or {}
    if chart.get("error"):
        raise ValueError(f"Yahoo error: {chart['error']}")
    results = chart.get("result") or []
    if not results:
        raise ValueError("Yahoo response had no result data.")
    result = results[0]
    timestamps = result.get("timestamp") or []
    quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
    opens = quote.get("open") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    closes = quote.get("close") or []
    volumes = quote.get("volume") or []

    bars: list[Bar] = []
    for i, ts in enumerate(timestamps):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        if None in (o, h, l, c) or ts is None:
            continue
        v = volumes[i] if i < len(volumes) and volumes[i] is not None else 0
        bars.append(
            Bar(
                timestamp=datetime.fromtimestamp(ts, tz),
                open=float(o), high=float(h), low=float(l), close=float(c), volume=float(v),
            )
        )
    bars.sort(key=lambda b: b.timestamp)
    return bars


def fetch_yahoo(
    symbol: str,
    *,
    range_: str = "2y",
    interval: str = "1d",
    timeout: float = 30.0,
) -> list[Bar]:
    """Fetch bars for a contract symbol from Yahoo Finance.

    ``range_``: 1d/5d/1mo/3mo/6mo/1y/2y/5y/10y/ytd/max.
    ``interval``: 1m/2m/5m/15m/30m/60m/1h/1d/1wk/1mo (intraday ranges are limited).
    """
    ticker = yahoo_ticker(symbol)
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?range={range_}&interval={interval}&includePrePost=false"
    )
    req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
    ctx = ssl.create_default_context()  # picks up SSL_CERT_FILE / system CAs
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return parse_yahoo_chart(payload)
