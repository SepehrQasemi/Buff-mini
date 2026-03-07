from __future__ import annotations

import pytest

from buffmini.stage44.contracts import build_failure_record, validate_failure_record


def test_stage44_failure_contract_accepts_allowed_motif() -> None:
    record = build_failure_record(
        module_name="mod",
        family_name="flow",
        motif="REJECT::COST_DRAG",
        details={"count": 1},
    )
    validate_failure_record(record)


def test_stage44_failure_contract_rejects_unknown_motif() -> None:
    with pytest.raises(ValueError):
        validate_failure_record(
            {
                "module_name": "mod",
                "family_name": "flow",
                "motif": "REJECT::UNKNOWN",
                "details": {},
            }
        )

