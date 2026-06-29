import importlib
from datetime import datetime, timedelta, timezone

import pytest

from ai_futures_bot.data import Bar, SyntheticDataGenerator, resample
from ai_futures_bot.ml.cv import PurgedKFold
from ai_futures_bot.ml.labeling import binary_labels, triple_barrier_labels


def _rising(n=120, step=2.0):
    t0 = datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)
    bars, p = [], 5000.0
    for i in range(n):
        p += step
        bars.append(Bar(t0 + timedelta(minutes=i), p, p + 1, p - 1, p, 1000))
    return bars


def test_triple_barrier_invariants():
    bars = resample(SyntheticDataGenerator(seed=3).generate(days=120), 60)
    labels, t1 = triple_barrier_labels(bars, max_horizon=20)
    assert len(labels) == len(bars) == len(t1)
    defined = [x for x in labels if x is not None]
    assert set(defined) <= {-1, 0, 1}
    for i in range(len(bars)):
        assert t1[i] >= i
        assert t1[i] <= i + 20


def test_triple_barrier_directional_sign():
    # A steadily rising series should hit the upper barrier first -> label +1.
    bars = _rising(step=3.0)
    labels, _ = triple_barrier_labels(bars, pt_mult=2.0, sl_mult=2.0, max_horizon=20)
    defined = [x for x in labels if x is not None]
    assert defined.count(1) > defined.count(-1)


def test_binary_labels_mapping():
    assert binary_labels([1, -1, 0, None]) == [1, 0, None, None]


def test_purged_kfold_no_leakage():
    n = 100
    t1 = [min(i + 5, n - 1) for i in range(n)]  # 5-bar label windows
    kf = PurgedKFold(n_splits=5, embargo=0.02)
    all_test = []
    for train_idx, test_idx in kf.split(n, t1):
        train_set, test_set = set(train_idx), set(test_idx)
        # train and test never overlap
        assert not (train_set & test_set)
        ts, te = min(test_idx), max(test_idx) + 1
        for j in train_idx:
            # no training label window may reach into the test fold (purged)
            assert not (j <= te - 1 and t1[j] >= ts)
            # embargo: nothing in the 2 bars right after the test fold
            assert not (te <= j < te + 2)
        all_test.extend(test_idx)
    assert sorted(all_test) == list(range(n))  # folds partition the data


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="sklearn not installed")
def test_meta_labeling_runs_and_filters():
    from ai_futures_bot.backtester import Backtester
    from ai_futures_bot.contracts import get_contract
    from ai_futures_bot.risk import RiskConfig, RiskManager
    from ai_futures_bot.strategies import get_strategy

    bars = resample(SyntheticDataGenerator(seed=5).generate(days=500), 60)
    spec = get_contract("MES")
    meta = Backtester(get_strategy("ml_meta", primary="supertrend"), spec,
                      RiskManager(config=RiskConfig()), starting_cash=50_000).run(bars)
    # Runs cleanly with invariants intact.
    assert meta.portfolio.position is None
    assert meta.portfolio.realized_pnl == pytest.approx(sum(t.pnl for t in meta.trades), abs=1e-6)
    # Meta only trades the held-out region, so it should not exceed the primary's
    # total signal count run over the same data.
    primary = Backtester(get_strategy("supertrend"), spec,
                         RiskManager(config=RiskConfig()), starting_cash=50_000).run(bars)
    assert meta.metrics["num_trades"] <= primary.metrics["num_trades"] + 2


@pytest.mark.skipif(importlib.util.find_spec("sklearn") is None, reason="sklearn not installed")
def test_purged_cv_score_returns_probability():
    from ai_futures_bot.ml.cv import purged_cv_score
    from ai_futures_bot.ml.features import build_features

    bars = resample(SyntheticDataGenerator(seed=4).generate(days=150), 60)
    rows, valid = build_features(bars)
    labels, t1 = triple_barrier_labels(bars, max_horizon=20)
    binL = binary_labels(labels)
    acc = purged_cv_score(rows, binL, valid, t1, n_splits=5, embargo=0.01)
    # synthetic data has no real edge -> accuracy should be near a coin flip
    assert acc is None or (0.0 <= acc <= 1.0)
    if acc is not None:
        assert 0.30 <= acc <= 0.70
