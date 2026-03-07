from __future__ import annotations

import pytest

from buffmini.stage50.reporting import validate_stage44_50_master_summary


def _payload() -> dict:
    return {
        "stage44_status": "SUCCESS",
        "stage45_status": "SUCCESS",
        "stage46_status": "SUCCESS",
        "stage47_status": "SUCCESS",
        "stage48_status": "SUCCESS",
        "stage49_status": "SUCCESS",
        "stage50_status": "PARTIAL",
        "stage47_raw_candidates_before": 24,
        "stage47_raw_candidates_after": 48,
        "stage48_stage_a_survivors": 10,
        "stage48_stage_b_survivors": 5,
        "stage49_registry_rows": 12,
        "stage49_elites_count": 5,
        "stage50_runtime_seconds_baseline": 100.0,
        "stage50_runtime_seconds_upgraded": 90.0,
        "stage50_promising": True,
        "stage50_5seed_executed": 5,
        "deterministic_summary_hash": "abc",
        "final_verdict": "TRADABILITY_IMPROVED_BUT_NO_ROBUST_EDGE",
        "biggest_remaining_bottleneck": "cost_drag_vs_signal",
        "next_cheapest_action": "Improve setup quality",
    }


def test_stage44_50_master_schema_accepts_valid_payload() -> None:
    validate_stage44_50_master_summary(_payload())


def test_stage44_50_master_schema_rejects_missing_key() -> None:
    payload = _payload()
    payload.pop("stage49_registry_rows")
    with pytest.raises(ValueError):
        validate_stage44_50_master_summary(payload)

