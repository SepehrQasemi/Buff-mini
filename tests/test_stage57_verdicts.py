from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os

from buffmini.stage57 import derive_stage57_verdict, detect_stale_inputs


def test_stage57_verdict_passes_when_all_gates_pass() -> None:
    verdict = derive_stage57_verdict(
        replay_metrics={"trade_count": 45, "exp_lcb": 0.01, "maxDD": 0.15, "failure_reason_dominance": 0.4},
        walkforward_metrics={"usable_windows": 5, "median_forward_exp_lcb": 0.002},
        monte_carlo_metrics={"conservative_downside_bound": 0.001},
        cross_seed_metrics={"surviving_seeds": 3},
        validation_history=[],
    )
    assert verdict["verdict"] == "PASSING_EDGE"


def test_stage57_detects_stale_inputs(tmp_path) -> None:
    path = tmp_path / "old.json"
    path.write_text("{}", encoding="utf-8")
    old = (datetime.now(timezone.utc) - timedelta(days=10)).timestamp()
    os.utime(path, (old, old))
    check = detect_stale_inputs([path], max_age_hours=24.0)
    assert check["stale"] is True
    assert str(path) in check["stale_paths"]


def test_stage57_requires_three_consecutive_scope_failures_for_no_edge() -> None:
    verdict_non_consecutive = derive_stage57_verdict(
        replay_metrics={"trade_count": 0, "exp_lcb": -0.01, "maxDD": 0.30, "failure_reason_dominance": 0.9},
        walkforward_metrics={"usable_windows": 1, "median_forward_exp_lcb": -0.01},
        monte_carlo_metrics={"conservative_downside_bound": -0.02},
        cross_seed_metrics={"surviving_seeds": 0},
        validation_history=[
            {"scope_frozen": True, "verdict": "PARTIAL"},
            {"scope_frozen": True, "verdict": "PASSING_EDGE"},
            {"scope_frozen": True, "verdict": "PARTIAL"},
            {"scope_frozen": True, "verdict": "PARTIAL"},
        ],
    )
    assert verdict_non_consecutive["verdict"] == "PARTIAL"

    verdict_consecutive = derive_stage57_verdict(
        replay_metrics={"trade_count": 0, "exp_lcb": -0.01, "maxDD": 0.30, "failure_reason_dominance": 0.9},
        walkforward_metrics={"usable_windows": 1, "median_forward_exp_lcb": -0.01},
        monte_carlo_metrics={"conservative_downside_bound": -0.02},
        cross_seed_metrics={"surviving_seeds": 0},
        validation_history=[
            {"scope_frozen": True, "verdict": "PARTIAL"},
            {"scope_frozen": True, "verdict": "PARTIAL"},
            {"scope_frozen": True, "verdict": "PARTIAL"},
        ],
    )
    assert verdict_consecutive["verdict"] == "NO_EDGE_IN_SCOPE"
