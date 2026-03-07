from __future__ import annotations

import pytest

from buffmini.stage50.reporting import validate_stage50_5seed_summary


def _payload() -> dict:
    return {
        "stage": "50_5seed",
        "status": "PARTIAL",
        "skipped": True,
        "skip_reason_if_any": "dead",
        "executed_seeds": [],
        "activation_rate_distribution": {"median": 0.0, "worst": 0.0, "best": 0.0},
        "trade_count_distribution": {"median": 0.0, "worst": 0.0, "best": 0.0},
        "exp_lcb_distribution": {"median": 0.0, "worst": 0.0, "best": 0.0},
        "family_consistency": {"flow": 0.5},
        "summary_hash": "abc",
    }


def test_stage50_5seed_schema_accepts_valid_payload() -> None:
    validate_stage50_5seed_summary(_payload())


def test_stage50_5seed_schema_rejects_missing_key() -> None:
    payload = _payload()
    payload.pop("family_consistency")
    with pytest.raises(ValueError):
        validate_stage50_5seed_summary(payload)

