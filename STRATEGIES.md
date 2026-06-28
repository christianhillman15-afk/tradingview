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

**Caveats (important):** this is a *single* train/test split, not full
walk-forward — real use needs rolling retraining, purged cross-validation,
realistic transaction costs, and out-of-sample validation on multiple regimes.
Ensembles reduce variance but cannot manufacture an edge that isn't in the data.
Sources: arXiv 2412.15448, the CFA Institute ML-in-commodities chapter,
ScienceDirect deep-ensemble HFT paper.

---

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

Interactive Brokers API
- https://github.com/erdewit/ib_insync
- https://www.interactivebrokers.com/campus/ibkr-quant-news/interactive-brokers-python-api-native-a-step-by-step-guide/
