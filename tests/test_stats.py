import math

import pytest

from ai_futures_bot import stats


def test_norm_cdf_known_values():
    assert stats.norm_cdf(0.0) == pytest.approx(0.5, abs=1e-9)
    assert stats.norm_cdf(1.96) == pytest.approx(0.975, abs=1e-3)
    assert stats.norm_cdf(-1.96) == pytest.approx(0.025, abs=1e-3)


def test_norm_ppf_inverts_cdf():
    for p in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        x = stats.norm_ppf(p)
        assert stats.norm_cdf(x) == pytest.approx(p, abs=1e-4)
    assert stats.norm_ppf(0.975) == pytest.approx(1.96, abs=1e-2)


def test_returns_moments_normalish():
    # Symmetric returns -> skew ~ 0, kurtosis ~ 3 region (small sample).
    rets = [0.01, -0.01, 0.02, -0.02, 0.015, -0.015]
    sr, skew, kurt, n = stats.returns_moments(rets)
    assert n == 6
    assert abs(skew) < 0.5
    assert sr == pytest.approx(0.0, abs=1e-9)  # zero mean


def test_psr_increases_with_track_length():
    # Same Sharpe, more observations -> higher confidence it's > 0.
    short = stats.probabilistic_sharpe_ratio(0.1, n=20, skew=0.0, kurtosis=3.0)
    long = stats.probabilistic_sharpe_ratio(0.1, n=500, skew=0.0, kurtosis=3.0)
    assert 0.0 <= short <= long <= 1.0
    assert long > short


def test_psr_zero_sharpe_is_half():
    assert stats.probabilistic_sharpe_ratio(0.0, n=100) == pytest.approx(0.5, abs=1e-6)


def test_psr_negative_skew_and_fat_tails_reduce_confidence():
    base = stats.probabilistic_sharpe_ratio(0.1, n=200, skew=0.0, kurtosis=3.0)
    worse = stats.probabilistic_sharpe_ratio(0.1, n=200, skew=-1.0, kurtosis=8.0)
    assert worse < base


def test_deflated_sharpe_drops_with_more_trials():
    # The same best Sharpe is less impressive if many variants were tried.
    trials = [0.05 + 0.001 * i for i in range(50)]
    best = max(trials)
    dsr_few = stats.deflated_sharpe_ratio(best, n=500, trial_sharpes=trials[:3], n_trials=3)
    dsr_many = stats.deflated_sharpe_ratio(best, n=500, trial_sharpes=trials, n_trials=200)
    assert 0.0 <= dsr_many <= dsr_few <= 1.0


def test_expected_max_sharpe_nonnegative_and_grows_with_trials():
    trials = [0.02 * i - 0.5 for i in range(40)]
    sr10 = stats.expected_max_sharpe(trials, n_trials=10)
    sr1000 = stats.expected_max_sharpe(trials, n_trials=1000)
    assert sr1000 > sr10 >= 0.0
