from __future__ import annotations

from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.research.mechanisms import mechanism_registry
from buffmini.research.saturation import evaluate_mechanism_saturation
from buffmini.stage70.search_expansion import compress_subfamily_hypotheses, generate_expanded_candidates


def test_stage98_registry_contains_deeper_branches() -> None:
    registry = mechanism_registry()
    by_family = {row["family"]: row for row in registry}
    structure = by_family["structure_pullback_continuation"]
    assert "volatility_reset_reclaim" in structure["subfamilies"]
    assert len(structure["triggers"]) >= 3
    funding = by_family["funding_oi_imbalance_reversion"]
    assert "basis_dislocation_revert" in funding["triggers"]


def test_stage98_saturation_summary_is_produced() -> None:
    cfg = load_config(Path(DEFAULT_CONFIG_PATH))
    summary = evaluate_mechanism_saturation(cfg)
    assert summary["raw_candidate_count"] >= summary["post_dedup_candidate_count"] >= 1500
    assert isinstance(summary["stage98b_required"], bool)


def test_stage98b_compression_is_deterministic() -> None:
    frame = generate_expanded_candidates(
        discovery_timeframes=["30m", "1h"],
        budget_mode_selected="search",
        active_families=[
            "structure_pullback_continuation",
            "failed_breakout_reversal",
            "volatility_regime_transition",
        ],
    )
    compressed_a = compress_subfamily_hypotheses(frame, max_subfamilies_per_family=2, max_variants_per_subfamily=32)
    compressed_b = compress_subfamily_hypotheses(frame, max_subfamilies_per_family=2, max_variants_per_subfamily=32)
    assert compressed_a["candidate_id"].tolist() == compressed_b["candidate_id"].tolist()
