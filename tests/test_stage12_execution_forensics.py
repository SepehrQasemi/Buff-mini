from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.stage12.forensics import (
    aggregate_execution_diagnostics,
    classify_invalid_reason,
    classify_stage12_1,
    metric_logic_validation,
)


def test_zero_trade_classified_separately() -> None:
    reason = classify_invalid_reason(
        trade_count=0.0,
        stability_classification="INSUFFICIENT_DATA",
        usable_windows=0,
        min_usable_windows_valid=3,
    )
    assert reason == "ZERO_TRADE"


def test_metric_logic_exp_lcb_recompute() -> None:
    pnl = np.asarray([1.0, -0.5, 0.25, 0.1], dtype=float)
    expected = float(np.mean(pnl) - np.std(pnl, ddof=0) / np.sqrt(float(pnl.size)))
    checks = metric_logic_validation(
        trade_count=4.0,
        profit_factor=1.2,
        pnl_values=pnl,
        exp_lcb_reported=expected,
        stability_classification="UNSTABLE",
        usable_windows=3,
        min_usable_windows_valid=3,
    )
    assert checks["exp_lcb_ok"] is True
    assert checks["all_ok"] is True


def test_execution_summary_and_classification() -> None:
    matrix = pd.DataFrame(
        [
            {
                "combo_key": "a",
                "raw_backtest_seconds": 0.001,
                "walkforward_executed": True,
                "MC_executed": False,
                "invalid_reason": "ZERO_TRADE",
                "metric_logic_all_ok": True,
                "walkforward_integrity_ok": True,
            },
            {
                "combo_key": "b",
                "raw_backtest_seconds": 0.002,
                "walkforward_executed": True,
                "MC_executed": True,
                "invalid_reason": None,
                "metric_logic_all_ok": True,
                "walkforward_integrity_ok": True,
            },
        ]
    )
    summary = aggregate_execution_diagnostics(matrix, suspicious_backtest_ms_threshold=5.0)
    assert summary["suspicious_execution"] is True
    leaderboard = pd.DataFrame([{"is_valid": False, "exp_lcb": 0.0, "stability_classification": "INVALID"}])
    classification = classify_stage12_1(diagnostics=summary, leaderboard=leaderboard)
    assert classification == "ENGINE_BUG"

