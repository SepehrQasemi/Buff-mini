from __future__ import annotations

import pytest

from buffmini.stage43.reporting import validate_stage43_5seed_summary


def _valid_payload() -> dict:
    return {
        "stage": "43.4",
        "seed": 42,
        "upgraded_reference_run_id": "upgraded_seed42",
        "executed_seed_count": 2,
        "skipped": False,
        "skip_reason": "",
        "note": "Executed all five validation seeds.",
        "rows": [
            {
                "seed": 43,
                "run_id": "run43",
                "raw_signal_count": 3,
                "activation_rate": 0.1,
                "trade_count": 1.0,
                "live_best_exp_lcb": 0.0,
            },
            {
                "seed": 44,
                "run_id": "run44",
                "raw_signal_count": 4,
                "activation_rate": 0.2,
                "trade_count": 2.0,
                "live_best_exp_lcb": 0.01,
            },
        ],
        "distribution": {
            "raw_candidate_count_median": 24.0,
            "activation_rate_median": 0.15,
            "trade_count_median": 1.5,
            "live_exp_lcb_median": 0.005,
            "live_exp_lcb_worst": 0.0,
            "live_exp_lcb_best": 0.01,
        },
        "summary_hash": "abc123",
    }


def test_stage43_5seed_schema_accepts_valid_payload() -> None:
    validate_stage43_5seed_summary(_valid_payload())


def test_stage43_5seed_schema_rejects_count_mismatch() -> None:
    payload = _valid_payload()
    payload["executed_seed_count"] = 1
    with pytest.raises(ValueError):
        validate_stage43_5seed_summary(payload)

