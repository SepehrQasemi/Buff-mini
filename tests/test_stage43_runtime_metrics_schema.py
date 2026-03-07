from __future__ import annotations

import pytest

from buffmini.stage43.reporting import REQUIRED_PHASE_KEYS, validate_stage43_performance_summary


def _valid_payload() -> dict:
    return {
        "stage": "43.3",
        "seed": 42,
        "baseline": {
            "run_id": "baseline_run",
            "summary_hash": "abc",
            "raw_signal_count": 0,
            "activation_rate": 0.0,
            "trade_count": 0.0,
            "research_best_exp_lcb": 0.0,
            "live_best_exp_lcb": 0.0,
            "wf_executed_pct": 100.0,
            "mc_trigger_pct": 100.0,
            "runtime_seconds": 10.0,
            "next_bottleneck": "signal_quality",
        },
        "upgraded": {
            "run_id": "upgraded_run",
            "summary_hash": "def",
            "raw_signal_count": 3,
            "activation_rate": 0.1,
            "trade_count": 1.0,
            "research_best_exp_lcb": 0.01,
            "live_best_exp_lcb": 0.005,
            "wf_executed_pct": 100.0,
            "mc_trigger_pct": 100.0,
            "runtime_seconds": 12.0,
            "next_bottleneck": "cost_drag_vs_signal",
        },
        "delta": {
            "delta_raw_signal_count": 3.0,
            "delta_activation_rate": 0.1,
            "delta_trade_count": 1.0,
            "delta_research_best_exp_lcb": 0.01,
            "delta_live_best_exp_lcb": 0.005,
            "delta_runtime_seconds": 2.0,
        },
        "promising": True,
        "phase_runtime_seconds": {key: 0.1 for key in REQUIRED_PHASE_KEYS},
        "slowest_phase": "replay_backtest",
        "budget_mode": "small",
        "summary_hash": "xyz",
    }


def test_stage43_performance_schema_accepts_valid_payload() -> None:
    validate_stage43_performance_summary(_valid_payload())


def test_stage43_performance_schema_rejects_missing_phase_key() -> None:
    payload = _valid_payload()
    payload["phase_runtime_seconds"].pop("candidate_generation")
    with pytest.raises(ValueError):
        validate_stage43_performance_summary(payload)

