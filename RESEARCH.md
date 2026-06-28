# Research notes — what actually makes a futures bot robust

This document distills a multi-source research pass (academic papers, López de
Prado, AQR, NY Fed, CME) into concrete, prioritised guidance for this bot. It
favours reputable primary sources over marketing blogs.

**Confidence labels.** ✅ = independently verified by adversarial cross-check in
our research run. 📄 = drawn from a reputable primary source but not
independently re-verified in this run (treat as well-supported, not gospel). The
verification pass was cut short by an API rate limit, so several 📄 items would
ordinarily be ✅.

---

## 1. Strategy edges that actually persist

- ✅ **Time-series momentum (trend) is the most robust documented edge.** A
  diversified TSMOM portfolio across equity-index, FX, commodity and bond futures
  earns a **gross Sharpe > 1.0 (~2.5× the equity market)** with little factor
  correlation (Moskowitz, Ooi & Pedersen 2012).
- ✅ **The canonical signal is the past 12-month return.** Continuation is
  strong for ~1 year, then **partially reverses over the next ~4 years** — so
  very long lookbacks invert sign.
- ✅ **It's pervasive, not data-mined.** All 58 liquid futures studied (1985–2009)
  had positive 12-month TSMOM; **52 of 58 significant at 5%**.
- ✅ **But single-market expectations are modest:** per-asset trend Sharpe ranges
  0–1, **averaging ~0.4 gross of costs** (AQR, *Understanding Managed Futures*).
  The portfolio Sharpe comes from **diversification** (avg pairwise correlation
  ~0.08), not any one market.
- 📄 **Intraday momentum is real but conditional:** on SPY/ES the first half-hour
  return predicts the last half-hour return, stronger on high-volatility,
  high-volume, and macro-news days (Gao, Han, Li & Zhou 2018).
- 📄 **Overnight drift exists but dies after costs:** the 2:00–3:00 ET hour earns
  ~1.1 Sharpe **gross** on ES, but **−0.5 after bid-ask spreads** (NY Fed staff
  report). A cautionary tale: a clean in-sample edge can be uneconomic net of
  execution.
- 📄 **Variance risk premium (VIX² − realised var)** predicts S&P 500 excess
  returns, peaking at the quarterly horizon (Bollerslev, Tauchen & Zhou 2009).

**→ Bot changes:** (1) add a **12-month time-series-momentum** strategy — done
(`tsmom`); (2) treat single-market Sharpe expectations as ~0.4 gross, not the
inflated synthetic numbers; (3) the biggest available upgrade is **multi-market
diversification** with volatility-parity sizing (roadmap: portfolio backtester);
(4) never trust an edge before modelling realistic spreads/commissions.

## 2. Execution & microstructure

- 📄 Intraday edges are **extremely cost-sensitive** — the overnight-drift result
  above (1.1 → −0.5 Sharpe after spreads) is the canonical example.
- Realistic ES/MES assumptions: ~1 tick bid-ask in liquid hours; model **1–2
  ticks of slippage round-turn** for market orders plus commission. MES round-turn
  commission is far cheaper than ES.
- Bar resolution matters: intraday strategies need tick/minute data; swing
  strategies should run on higher timeframes to avoid being dominated by costs.

**→ Bot changes:** already models commission + slippage and resamples swing
strategies to higher timeframes. Keep slippage assumptions conservative; never
backtest intraday logic on coarse bars.

## 3. Risk management & position sizing

- 📄 **Volatility targeting / parity is the professional default:** size each
  position to a fixed risk (e.g. target annualised vol per instrument), which
  prevents the most volatile market from dominating portfolio risk. This is what
  lifts diversified Sharpe.
- 📄 **Kelly:** full Kelly maximises long-run log-growth but is far too volatile
  in practice; **fractional (½ or ¼) Kelly** is the standard compromise — much
  lower drawdowns for a small growth give-up.
- **Risk of ruin** rises sharply with leverage even for positive-expectancy
  systems; drawdown control beats return chasing.

**→ Bot changes:** keep ATR/1%-risk sizing (already vol-aware); **add an optional
volatility-target sizing mode** and document fractional-Kelly as an alternative;
the daily-loss kill switch and drawdown halt already encode "survive first".

## 4. Robustness & validation (most important)

- 📄 **Deflated Sharpe Ratio** corrects an observed Sharpe for (a) multiple-testing
  selection bias and (b) non-normal returns — exactly the PSR/DSR this bot now
  implements (Bailey & López de Prado 2014). The deflation threshold SR₀ is a
  closed-form function of the number of trials N and the variance of trial
  Sharpes (using the Euler–Mascheroni constant) — **which is what `stats.py`
  computes.**
- 📄 **Minimum Backtest Length (MinBTL):** with only ~5 years of data, trying
  **more than ~45 independent configurations** almost guarantees a skill-less
  strategy will show an in-sample Sharpe of 1. Bound: `MinBTL (yrs) <
  2·ln(N) / E[max]²`.
- 📄 **Purged & embargoed / combinatorial cross-validation** (López de Prado)
  prevents leakage from overlapping labels in ML backtests.
- Walk-forward + out-of-sample degradation is the practical acceptance test: a
  strategy that only shines in-sample is rejected.

**→ Bot changes:** PSR on every backtest and **DSR in the optimizer** — done; add
a **MinBTL warning** when the trial count is large relative to the data length —
done (`optimize` prints it); ML strategy should move to purged CV (roadmap).

## 5. Machine learning for futures

- 📄 **Triple-barrier labeling** (profit-take / stop-loss / time barriers) and
  **meta-labeling** (a second model that sizes/【filters】 a primary model's bets)
  are the López de Prado standards; they beat naive fixed-horizon labels.
- 📄 **Leakage is the #1 killer:** features must not peek across the label window;
  use purged CV with an embargo. Sample weights should down-weight overlapping
  labels.
- Naive ML overfits because financial data has a low signal-to-noise ratio and
  non-stationarity; ensembles + heavy regularisation + realistic costs are
  mandatory, and expectations should be modest.

**→ Bot changes:** the ML strategy already uses a forward-return dead-band and a
strict train/test split; roadmap: triple-barrier labels, purged CV, and
meta-labeling on top of the rule-based signals.

## 6. Regime detection

- ADX, the Kaufman efficiency ratio, and choppiness index are cheap, transparent
  trend-vs-range classifiers; the bot's `ensemble` already gates members by
  **ADX + efficiency ratio**.
- 📄 More advanced: **Hidden Markov Models / volatility regimes** (e.g. high vs
  low VIX) to switch or weight strategies; vol regimes are persistent and
  forecastable.

**→ Bot changes:** ensemble regime gating is in place; roadmap: a volatility-state
overlay (e.g. scale exposure down in high-vol regimes) and optional HMM regime
labels.

---

## Prioritised roadmap (highest leverage first)

1. **Multi-market portfolio backtesting** — done (`portfolio` command /
   `portfolio_backtest.py`): runs a strategy across a diversified basket as
   risk-balanced sleeves, with equal or inverse-volatility weighting, and reports
   average pairwise correlation and the diversification ratio. On the synthetic
   demo, individual markets swing −20%…+33% but the basket is smooth with ~3%
   drawdown — the diversification benefit the research predicts.
2. **12-month time-series momentum** strategy — done (`tsmom`).
3. **Volatility-target sizing mode** in the risk manager (inverse-vol capital
   weighting is available at the portfolio level; per-trade vol-target sizing is
   the next step).
4. **MinBTL / DSR gating** surfaced everywhere a search happens — done in
   `optimize`.
5. **Purged/combinatorial CV + triple-barrier labeling** for the ML strategy.
6. **Realistic cost modelling** stays front-and-centre — the overnight-drift
   result is the reminder that gross edges routinely vanish net of spreads.

## Sources

- Moskowitz, Ooi, Pedersen, "Time Series Momentum" (2012) — https://w4.stern.nyu.edu/facdir/lpederse/papers/TimeSeriesMomentum.pdf
- AQR, "Understanding Managed Futures" — https://www.aqr.com/-/media/AQR/Documents/Insights/White-Papers/Understanding-Managed-Futures.pdf
- Hurst, Ooi, Pedersen, "A Century of Evidence on Trend-Following" — https://fairmodel.econ.yale.edu/ec439/hurst.pdf
- Gao, Han, Li, Zhou, "Market Intraday Momentum" (2018) — https://www.sciencedirect.com/science/article/abs/pii/S0304405X18301351
- NY Fed, "The Overnight Drift" — https://www.newyorkfed.org/research/staff_reports
- Bollerslev, Tauchen, Zhou, "Expected Stock Returns and Variance Risk Premia" (2009) — https://public.econ.duke.edu/~boller/Published_Papers/rfs_09.pdf
- Bailey & López de Prado, "The Deflated Sharpe Ratio" — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- Bailey, Borwein, López de Prado, Zhu, "Pseudo-Mathematics and Financial Charlatanism" (MinBTL) — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308659
- MacLean, Thorp, Ziemba, "The Good and Bad Properties of the Kelly Criterion" — https://www.stat.berkeley.edu/~aldous/157/Papers/Good_Bad_Kelly.pdf
- López de Prado, *Advances in Financial Machine Learning* (purged CV, triple-barrier, meta-labeling)

> **Caveat:** the synthesis/verification pass was truncated by an API rate limit.
> The ✅ items were adversarially cross-checked; 📄 items come from the cited
> primary sources but were not independently re-verified in this run. Verify
> before risking capital.
