from __future__ import annotations

import pytest

from buffmini.stage38.reporting import validate_stage38_master_summary


def _valid_payload() -> dict:
    return {
        "stage": "38.6",
        "seed": 42,
        "stage28_run_id": "run_stage28",
        "trace_hash": "abc123",
        "lineage_table": {
            "raw_signal_count": 0,
            "legacy_raw_signal_count": 10,
            "post_threshold_count": 0,
            "post_cost_gate_count": 0,
            "post_feasibility_count": 0,
            "composer_signal_count": 0,
            "engine_raw_signal_count": 0,
            "final_trade_count": 0.0,
        },
        "collapse_reason": "no_raw_candidates",
        "contradiction_fixed": True,
        "oi_usage": {"short_only_enabled": True},
        "self_learning": {"registry_rows": 1},
        "verdict": "LOGIC_FIXED_NO_EDGE",
        "biggest_remaining_bottleneck": "signal_quality",
        "next_action": "Tune candidate families",
    }


def test_stage38_summary_schema_accepts_valid_payload() -> None:
    payload = _valid_payload()
    validate_stage38_master_summary(payload)


def test_stage38_summary_schema_rejects_missing_required_key() -> None:
    payload = _valid_payload()
    payload.pop("lineage_table")
    with pytest.raises(ValueError):
        validate_stage38_master_summary(payload)

