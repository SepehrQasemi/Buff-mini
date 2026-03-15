from __future__ import annotations

from scripts.run_stage90 import _scenario_outcome


def test_stage90_campaign_outcome_classification() -> None:
    assert _scenario_outcome({"blocked": True, "blocked_count": 1}) == "blocked"
    assert _scenario_outcome({"robust_count": 1}) == "robust"
    assert _scenario_outcome({"validated_count": 1}) == "validated"
    assert _scenario_outcome({"promising_count": 1}) == "promising"
    assert _scenario_outcome({}) == "no_edge"
