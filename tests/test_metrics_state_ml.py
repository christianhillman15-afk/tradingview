import importlib
from datetime import datetime, timedelta, timezone

import pytest

from ai_futures_bot.contracts import get_contract
from ai_futures_bot.data import SyntheticDataGenerator
from ai_futures_bot.metrics import compute_metrics
from ai_futures_bot.portfolio import Portfolio, Trade
from ai_futures_bot.state import read_state, snapshot, write_state


def _trade(pnl, day=2):
    t = datetime(2024, 1, day, 15, 0, tzinfo=timezone.utc)
    side = "long" if pnl >= 0 else "short"
    return Trade(
        strategy="x", side=side, quantity=1,
        entry_time=t, entry_price=5000.0,
        exit_time=t + timedelta(minutes=5), exit_price=5000.0 + pnl / 50.0,
        pnl=pnl, commission=5.0,
    )


def test_metrics_win_rate_and_profit_factor():
    t0 = datetime(2024, 1, 2, tzinfo=timezone.utc)
    curve = [(t0 + timedelta(days=i), 50_000 + 1000 * i) for i in range(5)]
    trades = [_trade(300), _trade(-100), _trade(200), _trade(-50)]
    m = compute_metrics(curve, trades, 50_000)
    assert m["num_trades"] == 4
    assert m["wins"] == 2 and m["losses"] == 2
    assert m["win_rate_pct"] == 50.0
    # PF = (300+200) / (100+50) = 3.333
    assert m["profit_factor"] == pytest.approx(3.333, abs=0.01)


def test_metrics_max_drawdown():
    t0 = datetime(2024, 1, 2, tzinfo=timezone.utc)
    vals = [50_000, 55_000, 52_000, 60_000, 48_000]
    curve = [(t0 + timedelta(days=i), v) for i, v in enumerate(vals)]
    m = compute_metrics(curve, [], 50_000)
    # Worst peak->trough is 60_000 -> 48_000 = 20%.
    assert m["max_drawdown_pct"] == pytest.approx(20.0, abs=0.01)


def test_snapshot_roundtrip(tmp_path):
    spec = get_contract("ES")
    pf = Portfolio(spec, starting_cash=50_000)
    t0 = datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc)
    pf.open(1, 1, 5000.0, t0)
    pf.mark(t0, 5005.0)
    state = snapshot(
        portfolio=pf, spec=spec, strategy_name="donchian_trend",
        last_price=5005.0, last_time=t0, events=[{"time": t0.isoformat(), "kind": "open", "message": "x"}],
        mode="backtest", bars_processed=1,
    )
    assert state["symbol"] == "ES"
    assert state["tradingview_symbol"] == "CME_MINI:ES1!"
    assert state["position"]["side"] == "long"
    path = tmp_path / "state.json"
    write_state(str(path), state)
    loaded = read_state(str(path))
    assert loaded["symbol"] == "ES"
    assert loaded["account"]["starting_equity"] == 50_000


def test_read_state_missing_returns_none(tmp_path):
    assert read_state(str(tmp_path / "nope.json")) is None


@pytest.mark.skipif(
    importlib.util.find_spec("sklearn") is None,
    reason="scikit-learn not installed",
)
def test_ml_features_and_model_train():
    from ai_futures_bot.ml.features import FEATURE_NAMES, build_features, build_labels
    from ai_futures_bot.ml.model import EnsembleModel

    bars = SyntheticDataGenerator(seed=3).generate(days=40)
    rows, valid = build_features(bars)
    assert len(rows) == len(bars)
    assert all(len(r) == len(FEATURE_NAMES) for r in rows)
    labels = build_labels(bars, horizon=10)
    X, y = [], []
    for i in range(len(bars)):
        if valid[i] and labels[i] is not None:
            X.append(rows[i])
            y.append(labels[i])
    assert len(X) > 100 and len(set(y)) == 2
    model = EnsembleModel().fit(X, y)
    probs = model.predict_proba_up(X[:20])
    assert len(probs) == 20
    assert all(0.0 <= p <= 1.0 for p in probs)
