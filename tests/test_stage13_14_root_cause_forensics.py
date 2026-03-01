from __future__ import annotations

import pandas as pd

from buffmini.forensics.stage13_14_root_cause import (
    classify_zero_trade_cause,
    compute_invalid_pct_from_rows,
    rank_impact_drivers,
)
from buffmini.stage13.evaluate import _gate_metrics


def test_classify_zero_trade_cause_priority() -> None:
    assert (
        classify_zero_trade_cause(
            score_abs_max=0.9,
            threshold=0.3,
            nan_share_required=0.30,
            signal_nonzero_count=10,
            crossing_count=10,
        )
        == "MISSING_FEATURES_OR_NAN"
    )
    assert (
        classify_zero_trade_cause(
            score_abs_max=0.2,
            threshold=0.3,
            nan_share_required=0.0,
            signal_nonzero_count=0,
            crossing_count=0,
        )
        == "SCORE_BELOW_THRESHOLD"
    )
    assert (
        classify_zero_trade_cause(
            score_abs_max=0.8,
            threshold=0.3,
            nan_share_required=0.0,
            signal_nonzero_count=0,
            crossing_count=5,
        )
        == "SIGNAL_MAPPING_BUG"
    )


def test_compute_invalid_pct_from_rows_uses_invalid_reason() -> None:
    rows = pd.DataFrame(
        {
            "invalid_reason": ["VALID", "NO_WF", "VALID", "ZERO_TRADE"],
            "classification": ["NO_EDGE", "NO_EDGE", "WEAK_EDGE", "INSUFFICIENT_DATA"],
        }
    )
    pct = compute_invalid_pct_from_rows(rows)
    assert abs(pct - 50.0) < 1e-12


def test_rank_impact_drivers_orders_by_impact_score() -> None:
    rows = [
        {
            "driver": "baseline",
            "variant": "realistic",
            "best_exp_lcb": 1.0,
            "invalid_pct": 10.0,
            "walkforward_executed_true_pct": 90.0,
            "tpm": 12.0,
        },
        {
            "driver": "cost",
            "variant": "high",
            "best_exp_lcb": -4.0,
            "invalid_pct": 40.0,
            "walkforward_executed_true_pct": 60.0,
            "tpm": 8.0,
        },
        {
            "driver": "composer",
            "variant": "vote",
            "best_exp_lcb": 0.5,
            "invalid_pct": 15.0,
            "walkforward_executed_true_pct": 85.0,
            "tpm": 11.0,
        },
    ]
    ranked = rank_impact_drivers(rows)
    assert ranked[0]["driver"] == "cost"
    assert ranked[0]["impact_score"] >= ranked[1]["impact_score"]


def test_stage13_gate_metrics_invalid_pct_uses_invalid_reason() -> None:
    rows = pd.DataFrame(
        {
            "trade_count": [1.0, 0.0, 2.0, 1.0],
            "classification": ["NO_EDGE", "INSUFFICIENT_DATA", "WEAK_EDGE", "ROBUST_EDGE"],
            "invalid_reason": ["VALID", "ZERO_TRADE", "VALID", "LOW_USABLE_WINDOWS"],
            "walkforward_executed": [True, False, True, True],
            "mc_executed": [True, False, True, True],
            "tpm": [1.0, 0.0, 2.0, 3.0],
        }
    )
    metrics = _gate_metrics(rows)
    assert abs(float(metrics["invalid_pct"]) - 50.0) < 1e-12
