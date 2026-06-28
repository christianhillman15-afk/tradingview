# Strategies — what the bot trades and why

This document summarises the professional, well-documented futures strategies
the bot implements, the rules each one follows, and the sources the rules were
drawn from. Every strategy here is a *published, widely-used* approach — none of
it is secret sauce, and none of it guarantees profit. The value is in a clean,
risk-managed, testable implementation you can study and extend.

> **Reality check.** Markets are adversarial and (largely) efficient. Edges are
> small, regime-dependent, and erode after costs. Treat every backtest number as
> a hypothesis to be stress-tested, not a promise. The synthetic data shipped
> with this repo is a *pipeline test*, not an edge — most strategies lose on it
> after commissions and slippage, which is the honest, expected outcome for a
> random-ish process.

---

## 1. Opening Range Breakout (ORB) — `opening_range_breakout`

A staple index-futures day-trading setup. Define the high/low of the first few
minutes after the open, then trade a break of that range in the direction of the
intraday trend.

**Rules implemented**
- Opening range = high/low of the first *N* minutes (default 5).
- Long on a break above the range high while price is **above session VWAP**;
  short below the range low while below VWAP.
- Stop at the opposite side of the opening range; target a 1.5× extension of the
  range height.
- One trade per session per direction; flatten by a session cutoff (~90 min in)
  and never hold overnight.

**Why these rules:** the first 5-minute range is a statistically robust window
for ES, breakouts are filtered by VWAP and trend alignment, and stops/targets
are anchored to the range itself. Sources: Fazen Capital, Trade That Swing,
edgeful, Warrior Trading (see Sources).

---

## 2. VWAP Mean-Reversion — `vwap_reversion`

VWAP (Volume-Weighted Average Price) is the intraday "fair value" anchor that
institutions benchmark against. Price tends to revert to it.

**Rules implemented**
- VWAP resets each session.
- Long when price is stretched ≥ *k*×ATR **below** VWAP **and** RSI < 25; short
  when stretched above VWAP with RSI > 75. Target is VWAP itself.
- Most reliable in the first 90 and last 60 minutes; an optional filter skips the
  11:00–14:00 ET lull.
- ATR-based protective stop.

**Why these rules:** VWAP should never be a standalone signal — it is combined
with an RSI extreme and a volatility-scaled stretch, and confined to the hours
when institutional flow makes reversion reliable. Sources: MetroTrade, Bulls on
Wall Street, LuxAlgo, OnlyPropFirms.

---

## 3. Donchian / "Turtle" Trend Following — `donchian_trend`

The classic Dennis/Eckhardt Turtle breakout system, built on Donchian channels.

**Rules implemented**
- Enter long on a break of the *N*-bar high (default 20), short on the *N*-bar
  low.
- Exit on a break of a shorter opposite channel (default 10).
- Volatility-based position sizing and a 2×ATR protective stop.

**Why these rules:** trend systems make money by cutting losers fast and riding
winners far; the Donchian breakout is the canonical, fully-mechanical entry, and
ATR sizing keeps risk constant across instruments and regimes. Sources: Altrady,
Tradeciety, Alchemy Markets, FundedTradingPlus.

---

## 4. MACD Momentum with Trend Filter — `macd_momentum`

**Rules implemented**
- MACD(12,26,9). Long when the MACD line crosses above its signal line **and**
  price is above the 200-period EMA; short on the opposite cross below the EMA.
- Exit when MACD crosses back through the signal line. ATR stop.

**Why these rules:** raw MACD crosses whipsaw badly in ranges, so crosses are
only taken in the direction of the higher-timeframe trend (the EMA filter), which
is the standard refinement in the literature. Sources: QuantifiedStrategies,
QuantInsti, the arXiv MACD comparative study.

---

## 5. Bollinger Band + RSI Mean-Reversion — `bollinger_reversion`

**Rules implemented**
- Long when price closes below the lower band and RSI < 30; short above the upper
  band with RSI > 70. Target the middle band (the mean).
- Stop 0.5×ATR beyond the pierced band; time-stop if the trade hasn't begun
  reverting within *max_hold* bars.

**Why these rules:** fading band extremes confirmed by an RSI extreme is a high
win-rate, modest-payoff approach that complements the breakout/momentum systems —
they win in different regimes. Sources: MetroTrade, Alchemy Markets, Tradewink,
CrossTrade.

---

## 6. AI / ML Ensemble — `ml_ensemble`

A soft-voting ensemble (RandomForest + GradientBoosting) over technical features
predicts the probability that price is higher *H* bars ahead.

**Rules implemented**
- Features: multi-horizon returns, RSI, MACD histogram, Bollinger position,
  ATR%, momentum, distance from EMAs, and a volume ratio.
- Label: sign of the forward return over the horizon, with a dead-band so tiny
  moves are excluded.
- To avoid look-ahead bias the model trains **only** on the first `train_frac`
  of the data and trades **only** the held-out remainder (a single split). A
  pre-trained model can be loaded via `model_path`.
- Long when P(up) > 0.58, short when < 0.42; ATR stop; size scaled by confidence.

**Rigor:** the model trains on **path-aware triple-barrier labels** (López de
Prado) rather than naive fixed-horizon return signs, and the `train` command
reports a **purged + embargoed 5-fold CV** accuracy that removes label-overlap
leakage — the honest out-of-sample read. On random synthetic data this correctly
reports ~50% (no edge), which is exactly the point: it refuses to be fooled by an
in-sample backtest that looks profitable.

**Caveats (important):** the in-strategy split is still a *single* train/test
split — real use needs rolling retraining, meta-labeling, realistic transaction
costs, and out-of-sample validation on multiple regimes.
Ensembles reduce variance but cannot manufacture an edge that isn't in the data.
Sources: arXiv 2412.15448, the CFA Institute ML-in-commodities chapter,
ScienceDirect deep-ensemble HFT paper.

---

## 7. Supertrend (ADX-filtered) — `supertrend`

A volatility-based trend follower. The Supertrend line (ATR×multiplier around
the HL2 midline) flips sides of price when the trend changes.

**Rules implemented**
- Enter on the Supertrend flip; only when ADX ≥ `adx_min` (skip the chop).
- The Supertrend line is the stop, and an ATR trailing stop ratchets the exit —
  this is the professional "Supertrend as a trailing stop" usage.

Defaults: ATR period 10, multiplier 3.0, ADX filter 20. Sources: TrendSpider,
CrossTrade, LuxAlgo.

## 8. Squeeze Breakout (Bollinger/Keltner) — `bollinger_squeeze`

John Carter's TTM Squeeze: when the Bollinger Bands contract *inside* the
Keltner Channels, volatility is coiled; the "squeeze" firing (BB expanding back
outside KC) precedes a directional move.

**Rules implemented**
- Detect squeeze ON (BB inside KC); on release, enter in the direction of
  momentum (ROC sign); stop at the opposite squeeze-range extreme; ATR trail.

Defaults: BB(20, 2.0), KC(20, 1.5×ATR), ROC(12). Sources: TrendSpider, StockCharts
ChartSchool, Deepvue.

## 9. Z-Score Mean-Reversion — `zscore_reversion`

Fades extremes measured in standard deviations of price from its rolling mean.

**Rules implemented**
- Long when z ≤ −`entry_z` (cheap), short when z ≥ +`entry_z` (rich); target the
  mean (z→0); ATR stop; time-stop if it doesn't revert. Defaults: period 20,
  entry_z 2.0. Source: standard statistical-arbitrage / Ornstein-Uhlenbeck logic.

## 10. Regime-Aware Ensemble — `ensemble` (the flagship)

Markets only trend ~30% of the time, so no single strategy wins in every regime.
This meta-strategy runs several members as parallel opinion generators and gates
them by an **ADX + efficiency-ratio regime filter**: in a trending regime only
trend/breakout/ml members vote; in a ranging regime only reversion members vote.
Weighted votes net to a single position; entries fire when conviction clears a
threshold and flatten when it collapses.

This is the "diversify across uncorrelated strategies and only deploy each in its
favourable regime" principle that systematic shops use. Sources: FMZQuant ADX
filter study, the regime-classifier (ADX + efficiency ratio + choppiness) approach.

---

## Robustness & validation (what makes the bot trustworthy)

Implementing strategies is the easy part; *trusting* them requires defending
against overfitting. The bot includes:

- **Probabilistic & Deflated Sharpe Ratio** (`stats.py`): the Bailey & López de
  Prado test for *skill vs luck*. PSR = P(true Sharpe > 0) given track-record
  length, skew, and kurtosis (reported on every backtest and the dashboard). The
  optimiser reports the **Deflated** Sharpe — the PSR after correcting for how
  many parameter variants were tried — and flags a result as "likely overfit"
  when it is low. A high in-sample Sharpe from a big sweep no longer fools you.
- **Walk-forward analysis** (`walkforward.py`): optimise parameters on in-sample
  history, evaluate on the next unseen segment, compound out-of-sample. The
  honest test that exposes curve-fitting — a strategy that only shines in-sample
  is rejected.
- **Parameter optimisation** (`optimize.py`): grid/random search over risk-adjusted
  objectives (Sharpe, Calmar, profit factor), with a minimum-trades guard so noise
  can't win. Always paired with walk-forward confirmation.
- **Monte Carlo stress testing** (`montecarlo.py`): bootstrap/reshuffle the trade
  sequence thousands of times to estimate probability of profit, **risk of ruin**,
  and the *likely-worst* drawdown that a single equity curve hides.
- **Higher-timeframe resampling** so swing strategies don't overtrade minute noise.
- **Volatility-based sizing** + **trailing stops** + **daily-loss & drawdown
  kill switches** for capital preservation.

Sources: López de Prado *Advances in Financial Machine Learning* (walk-forward,
purged CV, deflated Sharpe), QuantInsti & Interactive Brokers on walk-forward,
QuantifiedStrategies on Monte Carlo, the 7 Circles / Concretum notes on
volatility targeting.

## Risk management (applies to every strategy)

The professional rules, enforced centrally in `risk.py`:

- **Risk a small fixed fraction per trade** (default 1% of equity), sized from
  the **stop distance**, not from available margin:
  `contracts = floor(risk_$ / (stop_points × point_value))`.
- **Daily-loss kill switch:** stop trading for the session once realised losses
  exceed a threshold (default 3% of start-of-day equity).
- **Global drawdown halt:** stop entirely past a max drawdown (default 20%).
- **Caps** on contract count and margin utilisation; volatility (ATR) scales
  both stops and size.

Sources: Optimus Futures, Insignia Futures, MyFundedFutures, Trade That Swing,
QuantVPS.

---

## Sources

Opening Range Breakout
- https://fazencapital.com/learn/en/opening-range-breakout-orb-strategy-indices-guide
- https://tradethatswing.com/opening-range-breakout-strategy-up-400-this-year/
- https://www.edgeful.com/blog/posts/es-futures-trading-strategies
- https://www.warriortrading.com/opening-range-breakout/

VWAP
- https://www.metrotrade.com/understanding-vwap-for-futures-trading/
- https://www.bullsonwallstreet.com/post/what-is-the-vwap-trading-indicator-and-how-to-use-it-as-a-day-trader
- https://www.luxalgo.com/blog/vwap-entry-strategies-for-day-traders/
- https://onlypropfirms.com/lessons/mastering-the-vwap

Turtle / Donchian trend following
- https://www.altrady.com/blog/crypto-trading-strategies/turtle-trading-strategy-rules
- https://tradeciety.com/donchian-channel-trading-indicator-tips
- https://alchemymarkets.com/education/strategies/turtle-trading-guide/
- https://www.fundedtradingplus.com/propiq/turtle-trading-strategy-the-classic-breakout-system-made-simple-donchian-channels-trend-filter/

MACD momentum
- https://www.quantifiedstrategies.com/macd-trading-strategy/
- https://blog.quantinsti.com/moving-average-trading-strategies/
- https://arxiv.org/pdf/2206.12282

Bollinger / RSI mean-reversion
- https://www.metrotrade.com/mean-reversion-trading-strategy/
- https://alchemymarkets.com/education/strategies/mean-reversion/
- https://www.tradewink.com/learn/mean-reversion-strategy

Machine learning
- https://arxiv.org/html/2412.15448v1
- https://rpc.cfainstitute.org/research/foundation/2025/chapter-8-machine-learning-commodity-futures
- https://www.sciencedirect.com/science/article/abs/pii/S1051200422001841

Risk management
- https://insigniafutures.com/futures-risk-management-6-tactical-rules/
- https://optimusfutures.com/blog/day-trading-risk-management/
- https://myfundedfutures.com/blog/daily-loss-limits-explained-protect-your-account-and-sanity
- https://tradethatswing.com/the-1-risk-rule-for-day-trading-and-swing-trading/

Supertrend
- https://trendspider.com/learning-center/supertrend-indicator-a-comprehensive-guide/
- https://crosstrade.io/learn/technical-indicators/supertrend
- https://www.luxalgo.com/blog/how-to-use-the-supertrend-indicator-effectively/

Squeeze breakout (TTM / Bollinger-Keltner)
- https://trendspider.com/learning-center/introduction-to-ttm-squeeze/
- https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/ttm-squeeze
- https://deepvue.com/indicators/ttm-squeeze-indicator-for-breakout-trades/

ADX / regime detection
- https://medium.com/@FMZQuant/strong-trend-adx-momentum-filtered-entry-quantitative-trading-strategy-0ae42cffd566
- https://www.liberatedstocktrader.com/adx-indicator/

Robustness & validation
- López de Prado, "Advances in Financial Machine Learning" (purged/combinatorial CV, deflated Sharpe)
- https://blog.quantinsti.com/walk-forward-optimization-introduction/
- https://www.interactivebrokers.com/campus/ibkr-quant-news/the-future-of-backtesting-a-deep-dive-into-walk-forward-analysis/
- https://www.quantifiedstrategies.com/monte-carlo-simulation-in-trading/
- https://the7circles.uk/systematic-trading-4-volatility-targeting-and-position-sizing/
- https://concretumgroup.com/position-sizing-in-trend-following-comparing-volatility-targeting-volatility-parity-and-pyramiding/

Interactive Brokers API
- https://github.com/erdewit/ib_insync
- https://www.interactivebrokers.com/campus/ibkr-quant-news/interactive-brokers-python-api-native-a-step-by-step-guide/
