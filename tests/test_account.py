from datetime import datetime, timedelta, timezone

import pytest

from ai_futures_bot.contracts import get_contract
from ai_futures_bot.data import LiveFeed
from ai_futures_bot.live import LiveTrader
from ai_futures_bot.risk import RiskConfig, RiskManager
from ai_futures_bot.strategies import get_strategy


def test_live_feed_is_deterministic_and_resumable():
    a = LiveFeed(start_price=5000, seed=7)
    b = LiveFeed(start_price=5000, seed=7)
    t = datetime(2024, 1, 2, tzinfo=timezone.utc)
    seq_a = [a.next_bar(t + timedelta(minutes=i)).close for i in range(20)]
    seq_b = [b.next_bar(t + timedelta(minutes=i)).close for i in range(20)]
    assert seq_a == seq_b  # same seed -> same path
    # export/restore continues the exact same path
    state = a.export()
    c = LiveFeed()
    c.restore(state)
    assert c.next_bar(t).close == a.next_bar(t).close


def _trader():
    return LiveTrader(
        get_strategy("donchian_trend"), get_contract("MES"),
        RiskManager(config=RiskConfig()), starting_cash=50_000, history_window=200,
    )


def test_paper_account_persist_and_resume():
    trader = _trader()
    feed = LiveFeed(start_price=5000, seed=3)
    now = datetime(2024, 1, 2, tzinfo=timezone.utc)
    trader.seed_history([feed.next_bar(now - timedelta(minutes=200 - k)) for k in range(200)])
    for k in range(400):
        trader.on_bar(feed.next_bar(now + timedelta(minutes=k)))

    acct = trader.export_account(feed)
    cash_before = trader.portfolio.cash
    trades_before = len(trader.portfolio.trades)
    bars_before = trader._bars_processed

    # Restore into a fresh trader.
    resumed = _trader()
    resumed.restore_account(acct)
    assert resumed.portfolio.cash == pytest.approx(cash_before)
    assert len(resumed.portfolio.trades) == trades_before
    assert resumed._bars_processed == bars_before
    assert resumed.portfolio.realized_pnl == pytest.approx(cash_before - 50_000)

    # And it can keep trading without error, equity stays finite/positive.
    f2 = LiveFeed()
    f2.restore(acct["feed"])
    resumed.on_bar(f2.next_bar(now + timedelta(minutes=401)))
    assert resumed.portfolio.equity(f2.price) > 0


def test_open_position_survives_round_trip():
    trader = _trader()
    feed = LiveFeed(start_price=5000, seed=11)
    now = datetime(2024, 1, 2, tzinfo=timezone.utc)
    trader.seed_history([feed.next_bar(now - timedelta(minutes=200 - k)) for k in range(200)])
    # Run until a position happens to be open (or give up after many bars).
    for k in range(2000):
        trader.on_bar(feed.next_bar(now + timedelta(minutes=k)))
        if trader.portfolio.position is not None:
            break
    acct = trader.export_account(feed)
    resumed = _trader()
    resumed.restore_account(acct)
    if trader.portfolio.position is not None:
        assert resumed.portfolio.position is not None
        assert resumed.portfolio.position.side == trader.portfolio.position.side
        assert resumed.portfolio.position.quantity == trader.portfolio.position.quantity
