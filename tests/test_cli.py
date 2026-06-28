"""Smoke tests for the CLI entry points (fast subsets only)."""

import pytest

from ai_futures_bot.cli import main


def test_list_strategies(capsys):
    assert main(["list-strategies"]) == 0
    out = capsys.readouterr().out
    assert "ensemble" in out and "supertrend" in out


def test_list_contracts(capsys):
    assert main(["list-contracts"]) == 0
    out = capsys.readouterr().out
    assert "MES" in out and "ES" in out


def test_backtest_runs_and_reports_psr(capsys):
    rc = main(["backtest", "--symbol", "MES", "--strategy", "donchian_trend",
               "--days", "60", "--timeframe", "60", "--seed", "5",
               "--state", "runtime/test_state.json"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Prob. Sharpe>0" in out and "Net profit" in out


def test_compare_leaderboard(capsys):
    rc = main(["compare", "--symbol", "MES", "--days", "60", "--seed", "5",
               "--strategies", "donchian_trend,supertrend,macd_momentum"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "leaderboard" in out.lower()
    assert "SHARPE" in out and "PSR%" in out


def test_compare_handles_unknown_strategy(capsys):
    rc = main(["compare", "--symbol", "MES", "--days", "40",
               "--strategies", "donchian_trend,not_a_strategy"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "skipping unknown" in out.lower()
