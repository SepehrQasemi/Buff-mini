"""Stage-4 execution policy tests."""

from __future__ import annotations

import pandas as pd

from buffmini.execution.policy import ExecutionMode, Signal, apply_execution_policy


def test_net_mode_nets_opposite_signals() -> None:
    ts = pd.Timestamp("2026-01-01T00:00:00Z")
    signals = [
        Signal(ts=ts, symbol="BTC/USDT", direction=1, strength=1.0, strategy_id="s1"),
        Signal(ts=ts, symbol="BTC/USDT", direction=-1, strength=0.4, strategy_id="s2"),
    ]
    targets = apply_execution_policy(signals, mode=ExecutionMode.NET, per_symbol_netting=True)

    assert len(targets) == 1
    assert round(targets[0].net_exposure, 6) == 0.6
    assert round(targets[0].gross_exposure, 6) == 1.4


def test_hedge_mode_keeps_both_sides() -> None:
    ts = pd.Timestamp("2026-01-01T00:00:00Z")
    signals = [
        Signal(ts=ts, symbol="ETH/USDT", direction=1, strength=0.5, strategy_id="a"),
        Signal(ts=ts, symbol="ETH/USDT", direction=-1, strength=0.7, strategy_id="b"),
    ]
    targets = apply_execution_policy(signals, mode=ExecutionMode.HEDGE, per_symbol_netting=True)

    assert len(targets) == 2
    signed = sorted(round(target.net_exposure, 6) for target in targets)
    assert signed == [-0.7, 0.5]


def test_isolated_groups_by_strategy_id() -> None:
    ts = pd.Timestamp("2026-01-01T00:00:00Z")
    signals = [
        Signal(ts=ts, symbol="BTC/USDT", direction=1, strength=0.3, strategy_id="x1"),
        Signal(ts=ts, symbol="BTC/USDT", direction=1, strength=0.2, strategy_id="x2"),
    ]
    targets = apply_execution_policy(signals, mode=ExecutionMode.ISOLATED, per_symbol_netting=True)

    assert len(targets) == 2
    strategy_groups = sorted(target.components[0]["strategy_group"] for target in targets)
    assert strategy_groups == ["x1", "x2"]
    assert all(target.symbol == "BTC/USDT" for target in targets)

