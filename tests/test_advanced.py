from datetime import datetime, timedelta, timezone

import pytest

from ai_futures_bot import indicators as ind
from ai_futures_bot.backtester import Backtester
from ai_futures_bot.contracts import get_contract
from ai_futures_bot.data import Bar, SyntheticDataGenerator, resample
from ai_futures_bot.engine import ExecutionEngine
from ai_futures_bot.metrics import compute_metrics
from ai_futures_bot.montecarlo import monte_carlo
from ai_futures_bot.optimize import grid_search
from ai_futures_bot.portfolio import Portfolio, Trade
from ai_futures_bot.risk import RiskConfig, RiskManager
from ai_futures_bot.strategies import get_strategy
from ai_futures_bot.strategies.base import Signal, Strategy
from ai_futures_bot.walkforward import walk_forward

SWING = ["supertrend", "bollinger_squeeze", "zscore_reversion", "ensemble"]


@pytest.fixture(scope="module")
def hourly_bars():
    return resample(SyntheticDataGenerator(seed=5).generate(days=200), 60)


# --- indicators ----------------------------------------------------------
def test_new_indicators_shapes(hourly_bars):
    h = [b.high for b in hourly_bars]
    l = [b.low for b in hourly_bars]
    c = [b.close for b in hourly_bars]
    st, d = ind.supertrend(h, l, c)
    a, pdi, mdi = ind.adx(h, l, c)
    ku, km, kl = ind.keltner_channels(h, l, c)
    z = ind.zscore(c)
    er = ind.efficiency_ratio(c)
    assert len(st) == len(d) == len(a) == len(hourly_bars)
    assert set(x for x in d if x is not None) <= {1, -1}
    assert all(v is None or 0 <= v <= 100 for v in a)
    # Keltner ordering where defined.
    for i in range(len(hourly_bars)):
        if ku[i] is not None:
            assert kl[i] <= km[i] <= ku[i]
    assert all(v is None or 0 <= v <= 1.0001 for v in er)


# --- resampling ----------------------------------------------------------
def test_resample_aggregates_ohlc():
    t0 = datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)
    bars = [
        Bar(t0 + timedelta(minutes=i), open=100 + i, high=105 + i, low=95 + i, close=100 + i, volume=10)
        for i in range(6)
    ]
    out = resample(bars, 3)
    assert len(out) == 2
    first = out[0]
    assert first.open == bars[0].open
    assert first.close == bars[2].close
    assert first.high == max(b.high for b in bars[:3])
    assert first.low == min(b.low for b in bars[:3])
    assert first.volume == 30


def test_resample_does_not_cross_sessions():
    t0 = datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)
    day2 = datetime(2024, 1, 3, 14, 30, tzinfo=timezone.utc)
    bars = [Bar(t0 + timedelta(minutes=i), 100, 101, 99, 100, 1) for i in range(2)]
    bars += [Bar(day2 + timedelta(minutes=i), 200, 201, 199, 200, 1) for i in range(2)]
    out = resample(bars, 10)  # bucket bigger than each session
    assert len(out) == 2  # one bar per session, never merged across days


# --- new strategies run --------------------------------------------------
@pytest.mark.parametrize("name", SWING)
def test_swing_strategies_run(name, hourly_bars):
    spec = get_contract("MES")
    risk = RiskManager(config=RiskConfig())
    result = Backtester(get_strategy(name), spec, risk, starting_cash=50_000).run(hourly_bars)
    assert result.metrics["num_trades"] >= 0
    assert result.portfolio.realized_pnl == pytest.approx(
        sum(t.pnl for t in result.trades), abs=1e-6
    )


def test_ensemble_uses_members(hourly_bars):
    strat = get_strategy("ensemble")
    strat.prepare(hourly_bars)
    assert len(strat._members) == len(strat.member_specs)


# --- trailing stop -------------------------------------------------------
class _OneShotLong(Strategy):
    name = "oneshot"

    def __init__(self, fire_at: int, trail: float):
        super().__init__()
        self.fire_at = fire_at
        self.trail = trail

    def warmup(self):
        return 0

    def on_bar(self, i):
        if i == self.fire_at:
            return Signal("long", stop=self._bars[i].close - 50, trail_atr_mult=self.trail)
        return None


def test_trailing_stop_ratchets_and_exits():
    spec = get_contract("MES")
    t0 = datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)
    bars = []
    price = 5000.0
    for i in range(40):                       # steady rise
        price += 5
        bars.append(Bar(t0 + timedelta(minutes=i), price, price + 2, price - 2, price, 1000))
    for i in range(40, 60):                    # then a drop to trigger the trail
        price -= 20
        bars.append(Bar(t0 + timedelta(minutes=i), price + 2, price + 3, price - 3, price, 2000))

    pf = Portfolio(spec, starting_cash=50_000)
    risk = RiskManager(config=RiskConfig(risk_per_trade=0.02))
    eng = ExecutionEngine(_OneShotLong(fire_at=20, trail=2.0), spec, risk, pf)
    eng.prepare(bars)

    stops = []
    for i in range(len(bars)):
        eng.step(bars, i, is_session_end=(i == len(bars) - 1))
        if pf.position is not None and pf.position.stop is not None:
            stops.append(pf.position.stop)
    # Stop should have ratcheted strictly upward during the rise.
    rising = [b for a, b in zip(stops, stops[1:]) if b >= a]
    assert len(stops) > 3
    assert len(rising) == len(stops) - 1  # never loosened
    # And the trade closed (trailing stop hit on the drop) with a recorded trade.
    assert len(pf.trades) == 1
    assert pf.trades[0].bars_held > 0


# --- extended metrics ----------------------------------------------------
def test_extended_metrics_present():
    t0 = datetime(2024, 1, 2, tzinfo=timezone.utc)
    curve = [(t0 + timedelta(days=i), 50_000 + 200 * i) for i in range(20)]
    trades = [
        Trade("x", "long", 1, t0, 5000, t0 + timedelta(hours=2), 5010, 400, 5, bars_held=3, mfe=600, mae=-100),
        Trade("x", "short", 1, t0, 5000, t0 + timedelta(hours=1), 5005, -150, 5, bars_held=2, mfe=50, mae=-300),
    ]
    m = compute_metrics(curve, trades, 50_000)
    for key in ("sortino", "calmar", "exposure_pct", "avg_bars_held", "avg_mfe", "avg_mae",
                "largest_win", "largest_loss", "gross_profit", "gross_loss"):
        assert key in m
    assert m["avg_bars_held"] == pytest.approx(2.5)


# --- Monte Carlo ---------------------------------------------------------
def test_monte_carlo_distribution():
    t0 = datetime(2024, 1, 2, tzinfo=timezone.utc)
    trades = [
        Trade("x", "long", 1, t0, 5000, t0, 5000 + p / 5, p, 5) for p in [300, -100, 250, -120, 180, -90, 400, -150]
    ]
    res = monte_carlo(trades, 50_000, simulations=500, seed=1)
    d = res.to_dict()
    assert 0.0 <= d["prob_profit"] <= 1.0
    assert 0.0 <= d["risk_of_ruin"] <= 1.0
    assert d["final_equity_pctiles"]["p5"] <= d["final_equity_pctiles"]["p95"]
    assert d["simulations"] == 500


def test_monte_carlo_empty_trades():
    res = monte_carlo([], 50_000, simulations=100)
    assert res.prob_profit == 0.0
    assert res.mean_final_equity == 50_000


# --- optimizer & walk-forward -------------------------------------------
def test_grid_search_ranks(hourly_bars):
    spec = get_contract("MES")
    results = grid_search(
        "donchian_trend",
        {"entry_period": [10, 20, 40], "exit_period": [5, 10]},
        hourly_bars, spec, objective="sharpe", starting_cash=50_000, top_n=3,
    )
    assert 1 <= len(results) <= 3
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert "entry_period" in results[0].params


def test_walk_forward_oos(hourly_bars):
    spec = get_contract("MES")
    wf = walk_forward("donchian_trend", hourly_bars, spec, n_splits=3, train_frac=0.5)
    assert len(wf.windows) >= 1
    assert "roi_pct" in wf.aggregate
    assert wf.final_equity > 0


# --- expanded contract universe -----------------------------------------
def test_contract_universe_is_broad_and_consistent():
    from ai_futures_bot.contracts import list_contracts

    cs = list_contracts()
    assert len(cs) >= 50  # not just the S&P 500 — all asset classes
    syms = {c.symbol for c in cs}
    # representative contracts from each asset class
    for s in ["ES", "6E", "ZB", "CL", "GC", "ZC", "LE", "KC", "BTC"]:
        assert s in syms, f"missing {s}"
    # every contract's economics are internally consistent
    for c in cs:
        assert c.tick_size > 0 and c.tick_value > 0
        assert c.point_value == pytest.approx(c.tick_value / c.tick_size)


def test_tsmom_runs(hourly_bars):
    spec = get_contract("MES")
    risk = RiskManager(config=RiskConfig())
    res = Backtester(get_strategy("tsmom", lookback=150), spec, risk, starting_cash=50_000).run(hourly_bars)
    assert res.metrics["num_trades"] >= 0
    assert res.portfolio.realized_pnl == pytest.approx(sum(t.pnl for t in res.trades), abs=1e-6)
