from __future__ import annotations

from buffmini.research.campaign import classify_campaign_outcome, select_campaign_families


def test_stage84_campaign_helpers_select_and_classify() -> None:
    families = select_campaign_families({"family_priority_adjustments": {"failed_breakout_reversal": 0.2}})
    assert families[0] == "failed_breakout_reversal"
    outcome = classify_campaign_outcome(
        edge_inventory=[{"final_class": "promising_but_unproven"}],
        evaluated_assets=2,
        blocked_assets=0,
    )
    assert outcome == "weak_promising_signs_need_refinement"
