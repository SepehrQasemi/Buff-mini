"""Stage-10.3 exit library tests."""

from __future__ import annotations

import numpy as np

from buffmini.stage10.exits import (
    apply_breakeven_after_1r,
    decide_exit_reason,
    normalize_exit_mode,
    partial_take_profit,
    trailing_stop_path,
)


def test_exit_priority_stop_first_deterministic() -> None:
    assert decide_exit_reason(stop_hit=True, take_profit_hit=True, time_stop_hit=True) == "stop_loss"
    assert decide_exit_reason(stop_hit=False, take_profit_hit=True, time_stop_hit=True) == "take_profit"
    assert decide_exit_reason(stop_hit=False, take_profit_hit=False, time_stop_hit=True) == "time_stop"
    assert decide_exit_reason(stop_hit=False, take_profit_hit=False, time_stop_hit=False) == ""


def test_trailing_stop_monotonic_for_long_side() -> None:
    path = trailing_stop_path(
        side="long",
        entry_stop=98.0,
        highs=[100.0, 101.0, 103.0, 104.0, 103.5],
        lows=[99.5, 99.0, 100.5, 102.0, 101.8],
        atr_values=[1.0, 1.0, 1.0, 1.0, 1.0],
        trailing_k=1.5,
    )
    diffs = np.diff(path.to_numpy(dtype=float))
    assert (diffs >= -1e-12).all()


def test_partial_take_profit_accounting_consistent() -> None:
    result = partial_take_profit(
        side="long",
        entry_price=100.0,
        target_price=103.0,
        qty_total=10.0,
        partial_fraction=0.5,
    )
    assert abs(result["partial_qty"] + result["remaining_qty"] - 10.0) <= 1e-12
    assert abs(result["realized_pnl"] - 15.0) <= 1e-12

    mode = normalize_exit_mode("partial_tp")
    assert mode == "partial_then_trail"

    moved = apply_breakeven_after_1r(side="long", entry_price=100.0, current_stop=97.0, one_r_reached=True)
    assert moved == 100.0
