from __future__ import annotations

from buffmini.research.mechanisms import generate_mechanism_source_candidates
from buffmini.stage52 import build_setup_candidate_v2
from buffmini.stage70.search_expansion import collapse_similarity_candidates


EXPECTED_FIELDS = {
    "subfamily",
    "participation",
    "risk_model",
    "exit_family",
    "time_stop_bars",
    "expected_regime",
    "expected_failure_modes",
    "trade_density_expectation",
    "transfer_expectation",
}


def test_stage87_refinement_schema_and_similarity_collapse() -> None:
    raw = generate_mechanism_source_candidates(
        discovery_timeframes=["1h"],
        budget_mode_selected="search",
        active_families=["structure_pullback_continuation", "failed_breakout_reversal"],
        target_min_candidates=200,
    )
    collapsed = collapse_similarity_candidates(raw, max_per_bucket=2)
    assert len(collapsed) < len(raw)
    upgraded = build_setup_candidate_v2(dict(raw.iloc[0].to_dict()), timeframe="1h")
    assert EXPECTED_FIELDS.issubset(upgraded.keys())
