from __future__ import annotations

import pytest

from buffmini.stage44.contracts import (
    build_contribution_record,
    validate_contribution_record,
)


def test_stage44_contribution_contract_accepts_valid_record() -> None:
    record = build_contribution_record(
        module_name="mod",
        family_name="flow",
        setup_name="setup_a",
        raw_candidate_contribution=0.5,
        stage_a_survival_lift=0.2,
        stage_b_survival_lift=0.1,
        final_policy_contribution=0.05,
        runtime_seconds=0.01,
        registry_rows_added=1,
        cost_of_use_if_measurable=None,
        coverage_flags={"ok": True},
    )
    validate_contribution_record(record)


def test_stage44_contribution_contract_rejects_missing_key() -> None:
    record = {
        "module_name": "mod",
        "family_name": "flow",
        "setup_name": "setup_a",
    }
    with pytest.raises(ValueError):
        validate_contribution_record(record)

