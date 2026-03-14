from __future__ import annotations

from buffmini.stage59 import build_conditional_expansion


def test_stage59_blocks_expansion_without_edge() -> None:
    result = build_conditional_expansion(
        stage58_verdict="NO_EDGE_IN_SCOPE",
        active_families=["structure_pullback_continuation"],
        next_families=["crowded_side_squeeze"],
    )
    assert result["expansion_allowed"] is False


def test_stage59_allows_expansion_for_positive_verdict() -> None:
    result = build_conditional_expansion(
        stage58_verdict="MEDIUM_EDGE",
        transfer_acceptable=True,
        active_families=["structure_pullback_continuation"],
        next_families=["structure_pullback_continuation", "crowded_side_squeeze"],
    )
    assert result["expansion_allowed"] is True
    assert result["next_families"] == ["crowded_side_squeeze"]


def test_stage59_blocks_expansion_without_transfer_acceptance() -> None:
    result = build_conditional_expansion(
        stage58_verdict="PASSING_EDGE",
        transfer_acceptable=False,
        active_families=["structure_pullback_continuation"],
        next_families=["crowded_side_squeeze"],
    )
    assert result["expansion_allowed"] is False
