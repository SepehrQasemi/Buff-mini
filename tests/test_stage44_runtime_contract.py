from __future__ import annotations

import pytest

from buffmini.stage44.contracts import (
    build_runtime_event,
    validate_runtime_event,
    validate_stage44_summary,
)


def test_stage44_runtime_contract_elapsed_is_deterministic() -> None:
    event = build_runtime_event(
        module_name="mod",
        phase_name="phase",
        enter_ts=100.0,
        exit_ts=100.5,
        candidate_rows_in=10,
        candidate_rows_out=4,
    )
    validate_runtime_event(event)
    assert float(event["elapsed_seconds"]) == 0.5


def test_stage44_summary_schema_required_keys() -> None:
    payload = {
        "stage": "44",
        "status": "SUCCESS",
        "contribution_contract_defined": True,
        "failure_contract_defined": True,
        "runtime_contract_defined": True,
        "allocator_hooks_defined": True,
        "registry_compatibility_defined": True,
        "modules_covered": ["flow"],
        "remaining_gaps": [],
        "summary_hash": "abc",
    }
    validate_stage44_summary(payload)
    payload.pop("modules_covered")
    with pytest.raises(ValueError):
        validate_stage44_summary(payload)

