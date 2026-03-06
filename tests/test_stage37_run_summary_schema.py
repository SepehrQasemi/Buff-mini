from __future__ import annotations

import pytest

from buffmini.stage37.reporting import validate_stage37_engine_summary


def _valid_payload() -> dict:
    return {
        "stage": "37.4",
        "seed": 42,
        "baseline": {
            "run_id": "r1",
            "wf_executed_pct": 10.0,
            "mc_trigger_pct": 20.0,
            "raw_signal_count": 100,
            "activation_rate": 0.1,
            "trade_count": 10.0,
            "research_best_exp_lcb": 0.0,
            "live_best_exp_lcb": -0.01,
            "maxDD": 0.2,
        },
        "upgraded": {
            "run_id": "r2",
            "wf_executed_pct": 20.0,
            "mc_trigger_pct": 30.0,
            "raw_signal_count": 150,
            "activation_rate": 0.12,
            "trade_count": 15.0,
            "research_best_exp_lcb": 0.01,
            "live_best_exp_lcb": -0.005,
            "maxDD": 0.19,
        },
        "delta": {
            "delta_wf_executed_pct": 10.0,
            "delta_mc_trigger_pct": 10.0,
            "delta_raw_signal_count": 50.0,
            "delta_activation_rate": 0.02,
            "delta_trade_count": 5.0,
            "delta_research_best_exp_lcb": 0.01,
            "delta_live_best_exp_lcb": 0.005,
            "delta_maxDD": -0.01,
        },
        "promising": True,
    }


def test_stage37_run_summary_schema_accepts_valid_payload() -> None:
    payload = _valid_payload()
    validate_stage37_engine_summary(payload)


def test_stage37_run_summary_schema_rejects_missing_key() -> None:
    payload = _valid_payload()
    payload.pop("delta")
    with pytest.raises(ValueError):
        validate_stage37_engine_summary(payload)
