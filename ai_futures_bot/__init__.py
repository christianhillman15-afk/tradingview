"""AI Futures Trading Bot.

A modular, research-grounded framework for backtesting and (paper) trading
futures contracts such as the E-mini / Micro E-mini S&P 500 (ES/MES).

The core engine (indicators, strategies, risk, backtester, paper broker,
synthetic data) is implemented in **pure-Python stdlib** so it runs anywhere
with zero third-party dependencies. The optional machine-learning strategy
uses scikit-learn and degrades gracefully when it is not installed.

DISCLAIMER
----------
This software is for **education and research only**. It is not financial
advice. Trading futures involves substantial risk of loss and is not suitable
for every investor. Past/backtested performance does not guarantee future
results. Nothing here is a promise of profit. Use the paper broker until you
fully understand the code, and never risk money you cannot afford to lose.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
