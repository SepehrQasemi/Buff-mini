from __future__ import annotations

from buffmini.research.bridge import classify_bridge_transition


def test_stage97_bridge_classification() -> None:
    relaxed = {"final_class": "promising_but_unproven", "first_death_stage": "walkforward"}
    assert classify_bridge_transition(relaxed_row=relaxed, strict_row={}) == "killed_by_data"
    assert (
        classify_bridge_transition(
            relaxed_row=relaxed,
            strict_row={"final_class": "promising_but_unproven", "first_death_stage": "survived"},
        )
        == "survive"
    )
    assert (
        classify_bridge_transition(
            relaxed_row=relaxed,
            strict_row={"final_class": "promising_but_unproven", "first_death_stage": "replay"},
        )
        == "become_weaker"
    )
    assert (
        classify_bridge_transition(
            relaxed_row=relaxed,
            strict_row={"final_class": "rejected", "candidate_hierarchy": "interesting_but_fragile"},
        )
        == "become_weaker"
    )
    assert (
        classify_bridge_transition(
            relaxed_row=relaxed,
            strict_row={"final_class": "rejected", "first_death_stage": "transfer"},
        )
        == "killed_by_transfer"
    )
