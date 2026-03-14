from __future__ import annotations

from buffmini.stage70 import EXPANDED_FAMILIES, generate_expanded_candidates


def test_stage70_generates_target_volume() -> None:
    frame = generate_expanded_candidates(discovery_timeframes=["15m", "1h"], budget_mode_selected="search")
    assert len(frame) >= 2500
    assert frame["family"].nunique() >= len(EXPANDED_FAMILIES)
    assert frame["timeframe"].nunique() == 2
    assert frame["economic_fingerprint"].nunique() == len(frame)
