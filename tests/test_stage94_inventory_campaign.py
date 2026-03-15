from __future__ import annotations

from scripts.run_stage94 import _campaign_label


def test_stage94_campaign_labeling() -> None:
    assert _campaign_label([], [{"blocked_reason": "gap"}]) == "data_blocked_or_scope_blocked"
    assert _campaign_label([{"final_class": "robust_candidate"}], []) == "robust_candidate_found"
    assert _campaign_label([{"final_class": "promising_but_unproven"}], []) == "weak_promising_signs"
    assert _campaign_label([{"final_class": "rejected"}], []) == "no_edge_found"
