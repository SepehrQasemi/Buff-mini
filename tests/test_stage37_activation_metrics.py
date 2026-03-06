from __future__ import annotations

import pandas as pd

from buffmini.stage37.activation import (
    ActivationHuntConfig,
    calibrate_thresholds,
    compute_reject_chain_metrics,
    mode_settings,
)


def _trace_fixture() -> pd.DataFrame:
    ts = pd.date_range("2026-01-01T00:00:00Z", periods=6, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "context": ["RANGE", "RANGE", "TREND", "TREND", "RANGE", "TREND"],
            "net_score": [0.0, 0.04, 0.12, -0.08, 0.20, -0.30],
            "active_candidates": ["", "a1", "a1,a2", "a2", "a1", "a2"],
            "final_signal": [0, 1, 1, -1, 1, -1],
        }
    )


def _shadow_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["2026-01-01T01:00:00Z", "2026-01-01T03:00:00Z"],
            "context": ["RANGE", "TREND"],
            "reason": ["SIZE_TOO_SMALL", "SIZE_TOO_SMALL"],
        }
    )


def _finalists_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "candidate_id": ["a1", "a2"],
            "family": ["funding", "flow"],
            "context": ["RANGE", "TREND"],
            "exp_lcb": [0.03, -0.01],
        }
    )


def test_stage37_reject_chain_metrics_exist() -> None:
    out = compute_reject_chain_metrics(
        trace_df=_trace_fixture(),
        shadow_df=_shadow_fixture(),
        finalists_df=_finalists_fixture(),
        threshold=0.05,
        quality_floor=-0.02,
        final_trade_count=2.0,
    )
    overall = dict(out.get("overall", {}))
    for key in (
        "raw_signal_count",
        "post_threshold_count",
        "post_cost_gate_count",
        "post_feasibility_count",
        "final_trade_count",
        "activation_rate",
    ):
        assert key in overall
    assert "per_family" in out


def test_stage37_threshold_calibration_deterministic() -> None:
    cfg = ActivationHuntConfig(
        threshold_grid=(0.0, 0.05, 0.10, 0.20),
        quality_floor=0.0,
        min_quality_floor=-0.02,
        min_selected_rows=1,
    )
    context_quality = {"RANGE": 0.02, "TREND": -0.01}
    first = calibrate_thresholds(trace_df=_trace_fixture(), context_quality=context_quality, cfg=cfg)
    second = calibrate_thresholds(trace_df=_trace_fixture(), context_quality=context_quality, cfg=cfg)
    assert first == second


def test_stage37_activation_hunt_mode_does_not_leak_into_strict() -> None:
    strict = mode_settings("strict")
    hunt = mode_settings("hunt")
    assert strict["name"] == "strict"
    assert strict["cost_mode"] == "live_strict"
    assert strict["quality_floor"] == 0.0
    assert hunt["name"] == "hunt"
    assert hunt["cost_mode"] != strict["cost_mode"]
