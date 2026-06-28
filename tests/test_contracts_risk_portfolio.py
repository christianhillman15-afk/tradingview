from datetime import datetime, timezone

import pytest

from ai_futures_bot.contracts import ContractSpec, get_contract
from ai_futures_bot.portfolio import Portfolio
from ai_futures_bot.risk import RiskConfig, RiskManager

# A zero-margin synthetic contract isolates stop-distance sizing from the
# margin cap (point value $50, like ES).
_SIZING_SPEC = ContractSpec("TST", "Test", "X", tick_size=0.25, tick_value=12.5, initial_margin=0)


def test_contract_point_value_and_pnl():
    es = get_contract("es")  # case-insensitive
    assert es.point_value == 50.0
    # 4-point move on 2 ES contracts = 4 * 50 * 2 = $400
    assert es.pnl(5000, 5004, 2) == pytest.approx(400.0)
    assert es.round_to_tick(5000.1) == 5000.0
    assert es.round_to_tick(5000.13) == 5000.25


def test_unknown_contract_raises():
    with pytest.raises(KeyError):
        get_contract("NOPE")


def test_position_sizing_from_stop_distance():
    rm = RiskManager(config=RiskConfig(risk_per_trade=0.01, max_contracts=100))
    # Risk budget = 50_000 * 1% = $500. Stop distance 10 pts * $50 = $500/contract -> 1 contract.
    qty = rm.position_size(50_000, entry_price=5000, stop_price=4990, spec=_SIZING_SPEC)
    assert qty == 1
    # Tighter 2-pt stop ($100/contract) -> 5 contracts.
    qty2 = rm.position_size(50_000, entry_price=5000, stop_price=4998, spec=_SIZING_SPEC)
    assert qty2 == 5


def test_position_sizing_respects_margin_cap():
    es = get_contract("ES")  # initial margin $13,200
    rm = RiskManager(config=RiskConfig(risk_per_trade=0.05, max_margin_fraction=0.5, max_contracts=100))
    # Risk budget is large, but margin cap = floor(50_000 * 0.5 / 13_200) = 1.
    qty = rm.position_size(50_000, entry_price=5000, stop_price=4999, spec=es)
    assert qty == 1


def test_position_size_zero_when_stop_too_wide():
    es = get_contract("ES")
    rm = RiskManager(config=RiskConfig(risk_per_trade=0.001))
    qty = rm.position_size(50_000, 5000, 4000, es)  # huge stop, tiny risk budget
    assert qty == 0


def test_daily_loss_limit_blocks_trading():
    rm = RiskManager(config=RiskConfig(daily_loss_limit=0.03))
    rm.start_bar("2024-01-02", 50_000)
    assert rm.can_trade() is True
    rm.register_trade(-1600)  # 3.2% of 50k
    assert rm.can_trade() is False
    # New day resets the counter.
    rm.start_bar("2024-01-03", 48_400)
    assert rm.can_trade() is True


def test_drawdown_kill_switch():
    rm = RiskManager(config=RiskConfig(max_drawdown_stop=0.20))
    rm.start_bar("d1", 50_000)
    rm.start_bar("d1", 39_000)  # 22% below peak
    assert rm.halted is True
    assert rm.can_trade() is False


def test_portfolio_roundtrip_pnl_and_costs():
    es = get_contract("ES")
    pf = Portfolio(es, starting_cash=50_000, commission_per_contract=2.5, slippage_ticks=0)
    t0 = datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc)
    pf.open(1, 2, 5000.0, t0)
    trade = pf.close(5010.0, t0, reason="target")
    # gross = 10 pts * $50 * 2 = $1000; round-turn commission = 2.5 * 2 * 2 = $10
    assert trade.pnl == pytest.approx(990.0)
    assert trade.commission == pytest.approx(10.0)
    assert pf.cash == pytest.approx(50_990.0)
    # realized_pnl reconciles with the blotter.
    assert pf.realized_pnl == pytest.approx(sum(t.pnl for t in pf.trades))


def test_portfolio_short_pnl_and_slippage():
    es = get_contract("ES")
    pf = Portfolio(es, starting_cash=50_000, commission_per_contract=0, slippage_ticks=1)
    t0 = datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc)
    pf.open(-1, 1, 5000.0, t0)              # fills 1 tick worse: 4999.75
    assert pf.position.entry_price == pytest.approx(4999.75)
    trade = pf.close(4990.0, t0)           # buy back 1 tick worse: 4990.25
    # short gross = (4990.25 - 4999.75) * 50 * (-1) = $475
    assert trade.pnl == pytest.approx(475.0)
