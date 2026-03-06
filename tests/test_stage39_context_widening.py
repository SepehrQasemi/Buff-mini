from __future__ import annotations

from buffmini.stage39.signal_generation import widen_context_label


def test_stage39_context_widening_expected_mappings() -> None:
    assert widen_context_label("RANGE") == "range"
    assert widen_context_label("VOL_EXPANSION") == "squeeze"
    assert widen_context_label("VOLUME_SHOCK") == "shock"
    assert widen_context_label("TREND") == "trend"


def test_stage39_context_widening_is_deterministic_for_unknowns() -> None:
    left = widen_context_label("custom_unknown_state")
    right = widen_context_label("custom_unknown_state")
    assert left == right
    assert left == "flow-dominant"

