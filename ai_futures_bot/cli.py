"""Command-line interface.

Subcommands
-----------
  list-strategies / list-contracts   Discoverability.
  backtest    Run a strategy over historical/synthetic data; print metrics and
              write a dashboard state file (optionally serve the dashboard).
  live        Paper trade a (synthetic or CSV) feed, updating the dashboard
              state live; optionally serve the dashboard in the same process.
  train       Train and save the ML ensemble model.
  dashboard   Serve the dashboard from an existing state file.
  demo        Backtest, then serve the dashboard — one command to see it all.

Everything defaults to a $50,000 paper account on the E-mini S&P 500 (ES).
"""

from __future__ import annotations

import argparse
import sys
import threading
from typing import Sequence

from .backtester import Backtester
from .config import Config
from .contracts import get_contract, list_contracts
from .data import Bar, SyntheticDataGenerator, load_csv
from .live import LiveTrader
from .risk import RiskManager
from .state import snapshot, write_state
from .strategies import get_strategy, list_strategies


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------
def _load_bars(cfg: Config) -> list[Bar]:
    d = cfg.data
    if d.source == "csv":
        if not d.csv_path:
            raise SystemExit("data.source=csv requires data.csv_path")
        return load_csv(d.csv_path)
    gen = SyntheticDataGenerator(
        seed=d.seed,
        start_price=d.start_price,
        annual_drift=d.annual_drift,
        annual_vol=d.annual_vol,
    )
    return gen.generate(days=d.days, bars_per_day=d.bars_per_day)


def _build_risk(cfg: Config) -> RiskManager:
    return RiskManager(config=cfg.risk)


def _parse_overrides(pairs: Sequence[str] | None) -> dict:
    out: dict = {}
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        out[key.strip()] = _coerce(raw.strip())
    return out


def _coerce(s: str):
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def _apply_common(cfg: Config, args) -> Config:
    if getattr(args, "symbol", None):
        cfg.symbol = args.symbol
    if getattr(args, "strategy", None):
        cfg.strategy = args.strategy
    if getattr(args, "balance", None) is not None:
        cfg.starting_cash = args.balance
    if getattr(args, "days", None) is not None:
        cfg.data.days = args.days
    if getattr(args, "seed", None) is not None:
        cfg.data.seed = args.seed
    if getattr(args, "csv", None):
        cfg.data.source = "csv"
        cfg.data.csv_path = args.csv
    if getattr(args, "state", None):
        cfg.state_path = args.state
    if getattr(args, "set", None):
        cfg.strategy_params.update(_parse_overrides(args.set))
    return cfg


def _load_config(args) -> Config:
    cfg = Config.from_yaml(args.config) if getattr(args, "config", None) else Config()
    return _apply_common(cfg, args)


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def _print_metrics(title: str, cfg: Config, metrics: dict) -> None:
    print(f"\n=== {title} ===")
    print(f"Contract        : {cfg.symbol}")
    print(f"Strategy        : {cfg.strategy}")
    print(f"Starting equity : ${metrics['starting_equity']:,.2f}")
    print(f"Final equity    : ${metrics['final_equity']:,.2f}")
    print(f"Net profit      : ${metrics['net_profit']:,.2f}  ({metrics['roi_pct']:+.2f}% ROI)")
    print(f"CAGR            : {metrics['cagr_pct']:+.2f}%")
    print(f"Trades          : {metrics['num_trades']}  "
          f"(W {metrics['wins']} / L {metrics['losses']}, win rate {metrics['win_rate_pct']:.1f}%)")
    print(f"Profit factor   : {metrics['profit_factor']}")
    print(f"Expectancy      : ${metrics['expectancy']:,.2f} / trade")
    print(f"Avg win / loss  : ${metrics['avg_win']:,.2f} / ${metrics['avg_loss']:,.2f} "
          f"(payoff {metrics['payoff_ratio']})")
    print(f"Max drawdown    : {metrics['max_drawdown_pct']:.2f}%  (${metrics['max_drawdown_dollars']:,.2f})")
    print(f"Sharpe          : {metrics['sharpe']}")
    print(f"Max consec. loss: {metrics['max_consecutive_losses']}")


def _risk_status(risk: RiskManager) -> dict:
    return {
        "can_trade": risk.can_trade(),
        "halted": risk.halted,
        "day_realized": round(risk.day_realized, 2),
        "equity_peak": round(risk.equity_peak, 2),
    }


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def cmd_list_strategies(args) -> int:
    print("Available strategies:")
    for name in list_strategies():
        print(f"  - {name}")
    return 0


def cmd_list_contracts(args) -> int:
    print(f"{'SYM':<6}{'NAME':<28}{'EXCH':<8}{'TICK':>8}{'TICK$':>9}{'POINT$':>9}")
    for s in list_contracts():
        print(f"{s.symbol:<6}{s.name:<28}{s.exchange:<8}{s.tick_size:>8}{s.tick_value:>9}{s.point_value:>9.2f}")
    return 0


def cmd_backtest(args) -> int:
    cfg = _load_config(args)
    spec = get_contract(cfg.symbol)
    bars = _load_bars(cfg)
    strategy = get_strategy(cfg.strategy, **cfg.strategy_params)
    risk = _build_risk(cfg)
    bt = Backtester(
        strategy,
        spec,
        risk,
        starting_cash=cfg.starting_cash,
        commission_per_contract=cfg.commission_per_contract,
        slippage_ticks=cfg.slippage_ticks,
    )
    result = bt.run(bars)
    _print_metrics("Backtest", cfg, result.metrics)

    last = bars[-1]
    state = snapshot(
        portfolio=result.portfolio,
        spec=spec,
        strategy_name=result.strategy_name,
        last_price=last.close,
        last_time=last.timestamp,
        events=result.events,
        mode="backtest",
        bars_processed=len(bars),
        risk_status=_risk_status(risk),
    )
    write_state(cfg.state_path, state)
    print(f"\nDashboard state written to {cfg.state_path}")
    if args.serve:
        _serve(cfg)
    else:
        print(f"View it with:  python -m ai_futures_bot.cli dashboard --state {cfg.state_path}")
    return 0


def cmd_live(args) -> int:
    cfg = _load_config(args)
    spec = get_contract(cfg.symbol)
    bars = _load_bars(cfg)
    strategy = get_strategy(cfg.strategy, **cfg.strategy_params)
    risk = _build_risk(cfg)
    trader = LiveTrader(
        strategy,
        spec,
        risk,
        starting_cash=cfg.starting_cash,
        commission_per_contract=cfg.commission_per_contract,
        slippage_ticks=cfg.slippage_ticks,
        state_path=cfg.state_path,
        update_every=args.update_every,
    )
    if args.serve:
        _serve(cfg, background=True)

    print(f"Paper trading {cfg.symbol} with {cfg.strategy} on ${cfg.starting_cash:,.0f} "
          f"(speed={args.speed}s/bar). Ctrl-C to stop.")

    def _on_update(state: dict) -> None:
        a = state["account"]
        sys.stdout.write(
            f"\r bars={state['bars_processed']:>6}  equity=${a['equity']:,.0f}  "
            f"ROI={a['roi_pct']:+.2f}%  trades={state['metrics']['num_trades']}   "
        )
        sys.stdout.flush()

    try:
        trader.run(bars, speed=args.speed, on_update=_on_update)
    except KeyboardInterrupt:
        print("\nStopped.")
    print()
    _print_metrics("Live (paper) session", cfg, trader.build_state(bars[-1])["metrics"])
    if args.serve:
        print("\nDashboard still serving. Ctrl-C to exit.")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            pass
    return 0


def cmd_train(args) -> int:
    cfg = _load_config(args)
    bars = _load_bars(cfg)
    from .ml.features import build_features, build_labels
    from .ml.model import EnsembleModel

    rows, valid = build_features(bars)
    labels = build_labels(bars, horizon=args.horizon)
    X, y = [], []
    for i in range(len(bars)):
        if valid[i] and labels[i] is not None:
            X.append(rows[i])
            y.append(labels[i])
    if len(X) < 100 or len(set(y)) < 2:
        raise SystemExit("Not enough labelled data to train. Use more --days.")
    print(f"Training ensemble on {len(X)} samples ({sum(y)} up / {len(y) - sum(y)} down)…")
    model = EnsembleModel().fit(X, y)
    model.save(args.out)
    print(f"Saved model to {args.out}")
    print(f"Use it with:  --strategy ml_ensemble --set model_path={args.out}")
    return 0


def cmd_dashboard(args) -> int:
    cfg = _load_config(args)
    _serve(cfg)
    return 0


def cmd_demo(args) -> int:
    args.serve = True
    return cmd_backtest(args)


def _serve(cfg: Config, background: bool = False) -> None:
    from .dashboard.server import DashboardServer

    server = DashboardServer(cfg.state_path, cfg.dashboard.host, cfg.dashboard.port)
    if background:
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
    else:
        server.serve_forever()


# --------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------
def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", help="Path to a YAML config file")
    p.add_argument("--symbol", help="Contract root symbol (e.g. ES, MES, NQ)")
    p.add_argument("--strategy", help="Strategy name (see list-strategies)")
    p.add_argument("--balance", type=float, help="Starting paper balance (default 50000)")
    p.add_argument("--days", type=int, help="Synthetic days to generate")
    p.add_argument("--seed", type=int, help="Synthetic RNG seed")
    p.add_argument("--csv", help="Load bars from a CSV instead of synthetic data")
    p.add_argument("--state", help="Path to the dashboard state JSON")
    p.add_argument("--set", action="append", metavar="key=value",
                   help="Override a strategy parameter (repeatable)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai_futures_bot",
        description="AI futures trading bot — backtest, paper trade, and dashboard.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list-strategies").set_defaults(func=cmd_list_strategies)
    sub.add_parser("list-contracts").set_defaults(func=cmd_list_contracts)

    p_bt = sub.add_parser("backtest", help="Run a backtest")
    _add_common(p_bt)
    p_bt.add_argument("--serve", action="store_true", help="Serve the dashboard afterwards")
    p_bt.set_defaults(func=cmd_backtest)

    p_live = sub.add_parser("live", help="Paper trade a feed live")
    _add_common(p_live)
    p_live.add_argument("--speed", type=float, default=0.0, help="Seconds to sleep per bar")
    p_live.add_argument("--update-every", type=int, default=5, help="Write state every N bars")
    p_live.add_argument("--serve", action="store_true", help="Serve the dashboard concurrently")
    p_live.set_defaults(func=cmd_live)

    p_tr = sub.add_parser("train", help="Train and save the ML ensemble model")
    _add_common(p_tr)
    p_tr.add_argument("--out", default="runtime/model.pkl", help="Output model path")
    p_tr.add_argument("--horizon", type=int, default=10, help="Label horizon in bars")
    p_tr.set_defaults(func=cmd_train)

    p_dash = sub.add_parser("dashboard", help="Serve the dashboard from a state file")
    _add_common(p_dash)
    p_dash.set_defaults(func=cmd_dashboard)

    p_demo = sub.add_parser("demo", help="Backtest then serve the dashboard")
    _add_common(p_demo)
    p_demo.set_defaults(func=cmd_demo)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
