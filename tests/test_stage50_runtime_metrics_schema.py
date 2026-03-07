from __future__ import annotations

import pytest

from buffmini.stage50.reporting import validate_stage50_performance_summary


def _payload() -> dict:
    return {
        "stage": "50",
        "status": "SUCCESS",
        "baseline_runtime_seconds": 10.0,
        "upgraded_runtime_seconds": 11.0,
        "delta_runtime_seconds": 1.0,
        "slowest_phase": "replay_backtest",
        "baseline_raw_signals": 0,
        "upgraded_raw_signals": 1,
        "baseline_trade_count": 0.0,
        "upgraded_trade_count": 1.0,
        "research_best_exp_lcb_before": 0.0,
        "research_best_exp_lcb_after": 0.01,
        "live_best_exp_lcb_before": 0.0,
        "live_best_exp_lcb_after": 0.005,
        "promising": True,
        "summary_hash": "abc",
    }


def test_stage50_runtime_schema_accepts_valid_payload() -> None:
    validate_stage50_performance_summary(_payload())


def test_stage50_runtime_schema_rejects_missing_key() -> None:
    payload = _payload()
    payload.pop("slowest_phase")
    with pytest.raises(ValueError):
        validate_stage50_performance_summary(payload)

