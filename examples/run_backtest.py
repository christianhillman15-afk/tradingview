"""Minimal programmatic example: backtest a strategy and print the metrics.

Run:  python examples/run_backtest.py
"""

from __future__ import annotations

import os
import sys

# Allow running directly from a checkout (python examples/run_backtest.py)
# without installing the package first.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_futures_bot.backtester import Backtester
from ai_futures_bot.contracts import get_contract
from ai_futures_bot.data import SyntheticDataGenerator
from ai_futures_bot.risk import RiskConfig, RiskManager
from ai_futures_bot.strategies import get_strategy


def main() -> None:
    spec = get_contract("ES")                         # E-mini S&P 500
    bars = SyntheticDataGenerator(seed=7).generate(days=90)
    strategy = get_strategy("donchian_trend", entry_period=20, exit_period=10)
    risk = RiskManager(config=RiskConfig(risk_per_trade=0.01, daily_loss_limit=0.03))

    bt = Backtester(strategy, spec, risk, starting_cash=50_000.0)
    result = bt.run(bars)

    m = result.metrics
    print(f"Strategy      : {result.strategy_name}")
    print(f"Net profit    : ${m['net_profit']:,.2f} ({m['roi_pct']:+.2f}%)")
    print(f"Trades        : {m['num_trades']} (win rate {m['win_rate_pct']:.1f}%)")
    print(f"Profit factor : {m['profit_factor']}")
    print(f"Max drawdown  : {m['max_drawdown_pct']:.2f}%")
    print(f"Sharpe        : {m['sharpe']}")


if __name__ == "__main__":
    main()
