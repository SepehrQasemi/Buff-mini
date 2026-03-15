from __future__ import annotations

from buffmini.research.truth_table import (
    assign_regime_buckets,
    build_regime_truth_rows,
    classify_scope_truth,
)


def test_classify_scope_truth_detects_strict_kill_vs_relaxed_survivor() -> None:
    label = classify_scope_truth(
        row={
            "mode": "canonical_strict",
            "blocked": False,
            "promising_count": 0,
            "validated_count": 0,
            "robust_count": 0,
            "dominant_failure_reason": "exp_lcb",
            "death_stage_counts": {"replay": 1},
        },
        relaxed_row={"blocked": False, "promising_count": 1},
    )
    assert label == "strict_evaluation_kills_things"


def test_build_regime_truth_rows_projects_family_into_expected_buckets() -> None:
    rows = build_regime_truth_rows(
        {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "mode": "live_relaxed",
            "blocked": False,
            "truth_label": "weak_signal_exists",
            "dominant_failure_reason": "exp_lcb",
            "evaluations": [
                {
                    "candidate_id": "c1",
                    "family": "structure_pullback_continuation",
                    "expected_regime": "trend",
                    "final_class": "promising_but_unproven",
                    "candidate_hierarchy": "promising_but_unproven",
                }
            ],
        }
    )
    trend = next(row for row in rows if row["regime"] == "trend")
    low_vol = next(row for row in rows if row["regime"] == "low_vol")
    assert trend["candidate_count"] == 1
    assert trend["promising_count"] == 1
    assert low_vol["candidate_count"] == 1
    assert low_vol["truth_label"] == "weak_signal_exists"


def test_assign_regime_buckets_maps_range_to_chop() -> None:
    buckets = assign_regime_buckets(
        {
            "family": "liquidity_sweep_reversal",
            "expected_regime": "range",
        }
    )
    assert "chop" in buckets
