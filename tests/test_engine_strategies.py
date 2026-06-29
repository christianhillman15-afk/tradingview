from datetime import datetime, timedelta, timezone

import pytest

from ai_futures_bot.backtester import Backtester
from ai_futures_bot.contracts import get_contract
from ai_futures_bot.data import Bar, SyntheticDataGenerator
from ai_futures_bot.risk import RiskConfig, RiskManager
from ai_futures_bot.strategies import get_strategy, list_strategies

RULE_BASED = [
    "donchian_trend",
    "macd_momentum",
    "bollinger_reversion",
    "opening_range_breakout",
    "vwap_reversion",
]


@pytest.fixture(scope="module")
def bars():
    return SyntheticDataGenerator(seed=11).generate(days=30)


def _backtest(strategy_name, bars, **params):
    spec = get_contract("ES")
    risk = RiskManager(config=RiskConfig())
    strat = get_strategy(strategy_name, **params)
    bt = Backtester(strat, spec, risk, starting_cash=50_000)
    return bt.run(bars)


@pytest.mark.parametrize("name", RULE_BASED)
def test_strategy_runs_and_reports(name, bars):
    result = _backtest(name, bars)
    m = result.metrics
    assert m["starting_equity"] == 50_000
    assert m["num_trades"] >= 0
    assert 0 <= m["win_rate_pct"] <= 100
    # Equity curve has one point per bar (plus possible final flatten mark).
    assert len(result.equity_curve) >= len(bars)
    # Realized PnL reconciles with the trade blotter.
    assert result.portfolio.realized_pnl == pytest.approx(
        sum(t.pnl for t in result.trades), abs=1e-6
    )


def test_all_strategies_registered():
    assert set(RULE_BASED).issubset(set(list_strategies()))
    assert "ml_ensemble" in list_strategies()


def test_intraday_strategy_flat_overnight(bars):
    # ORB is intraday: it should never hold a position across a session boundary.
    spec = get_contract("ES")
    risk = RiskManager(config=RiskConfig())
    strat = get_strategy("opening_range_breakout")
    bt = Backtester(strat, spec, risk, starting_cash=50_000)
    result = bt.run(bars)
    # Every trade opens and closes on the same calendar day.
    for t in result.trades:
        assert t.entry_time.date() == t.exit_time.date()


def test_protective_stop_caps_loss():
    # Construct a long breakout then a gap down through the stop.
    spec = get_contract("ES")
    t0 = datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc)
    bars = []
    price = 5000.0
    for i in range(60):
        # gentle uptrend to trigger a Donchian long
        price += 1.0
        bars.append(Bar(t0 + timedelta(minutes=i), price, price + 1, price - 1, price, 1000))
    # Now a sharp drop that should hit the ATR stop.
    for i in range(60, 70):
        price -= 30.0
        bars.append(Bar(t0 + timedelta(minutes=i), price + 5, price + 6, price - 1, price, 5000))
    risk = RiskManager(config=RiskConfig(risk_per_trade=0.01))
    strat = get_strategy("donchian_trend", entry_period=20, exit_period=10)
    bt = Backtester(strat, spec, risk, starting_cash=50_000)
    result = bt.run(bars)
    # A stop or exit must have fired; we should not lose more than the risk budget
    # by a wide margin on the single position (allowing slippage).
    if result.trades:
        worst = min(t.pnl for t in result.trades)
        assert worst > -2_000  # ~1% risk of 50k plus slippage, comfortably bounded
