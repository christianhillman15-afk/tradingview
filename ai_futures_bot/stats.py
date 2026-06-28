"""Statistical tools for deciding whether a result is skill or luck.

Implements the Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR)
from Bailey & López de Prado. These answer the question a backtest cannot on its
own: *given the track-record length, the non-normality of returns, and how many
strategy variants were tried, how confident can I be that the true Sharpe is
above a benchmark?*

  * PSR(SR*) = probability the true Sharpe exceeds SR*, accounting for sample
    length n, skewness, and kurtosis of the returns.
  * DSR = PSR evaluated at the *deflated* benchmark SR*₀ — the Sharpe you would
    expect from the **best of N independent trials** purely by chance. A high
    in-sample Sharpe from a big parameter sweep can have a low DSR, which is the
    statistical fingerprint of overfitting.

Pure stdlib (math.erf / a rational inverse-normal), so no scipy dependency.
"""

from __future__ import annotations

import math
from typing import Sequence

_EULER_MASCHERONI = 0.5772156649015329


def norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if p <= 0.0:
        return -math.inf
    if p >= 1.0:
        return math.inf
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]
    p_low, p_high = 0.02425, 1.0 - 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def returns_moments(returns: Sequence[float]) -> tuple[float, float, float, int]:
    """Return ``(per_period_sharpe, skew, kurtosis_nonexcess, n)`` for a series.

    Sharpe here is the *non-annualised* per-observation ratio mean/std, which is
    the scale PSR/DSR operate on.
    """
    n = len(returns)
    if n < 2:
        return 0.0, 0.0, 3.0, n
    mean = sum(returns) / n
    m2 = sum((r - mean) ** 2 for r in returns) / n
    if m2 <= 0:
        return 0.0, 0.0, 3.0, n
    std = math.sqrt(m2)
    m3 = sum((r - mean) ** 3 for r in returns) / n
    m4 = sum((r - mean) ** 4 for r in returns) / n
    skew = m3 / (std ** 3)
    kurt = m4 / (m2 ** 2)          # non-excess (normal == 3)
    return mean / std, skew, kurt, n


def probabilistic_sharpe_ratio(
    sharpe: float,
    n: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    benchmark: float = 0.0,
) -> float:
    """PSR: P(true Sharpe > ``benchmark``). ``sharpe``/``benchmark`` are on the
    same per-observation scale. Returns a probability in [0, 1]."""
    if n < 2:
        return 0.0
    denom = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe * sharpe
    if denom <= 0:
        return 0.0
    z = (sharpe - benchmark) * math.sqrt(n - 1) / math.sqrt(denom)
    return norm_cdf(z)


def expected_max_sharpe(trial_sharpes: Sequence[float], n_trials: int | None = None) -> float:
    """Expected maximum Sharpe from ``N`` independent trials under the null of no
    skill — the deflation benchmark SR*₀."""
    trials = [s for s in trial_sharpes if s is not None]
    n = n_trials or len(trials)
    if n < 2 or len(trials) < 2:
        return 0.0
    mean = sum(trials) / len(trials)
    var = sum((s - mean) ** 2 for s in trials) / (len(trials) - 1)
    sd = math.sqrt(var)
    if sd <= 0:
        return 0.0
    g = _EULER_MASCHERONI
    return sd * ((1.0 - g) * norm_ppf(1.0 - 1.0 / n) + g * norm_ppf(1.0 - 1.0 / (n * math.e)))


def deflated_sharpe_ratio(
    best_sharpe: float,
    n: int,
    trial_sharpes: Sequence[float],
    skew: float = 0.0,
    kurtosis: float = 3.0,
    n_trials: int | None = None,
) -> float:
    """DSR: PSR of the best strategy against the deflated benchmark expected from
    the number of trials. A low DSR means the result is likely overfit."""
    sr_star = expected_max_sharpe(trial_sharpes, n_trials)
    return probabilistic_sharpe_ratio(best_sharpe, n, skew, kurtosis, benchmark=sr_star)
