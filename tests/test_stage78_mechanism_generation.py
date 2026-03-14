from __future__ import annotations

from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.stage51 import resolve_research_scope
from buffmini.stage70 import generate_expanded_candidates


def test_stage78_generator_outputs_structured_mechanisms() -> None:
    frame = generate_expanded_candidates(
        discovery_timeframes=["1h", "4h"],
        budget_mode_selected="search",
        active_families=[
            "structure_pullback_continuation",
            "failed_breakout_reversal",
            "volatility_regime_transition",
        ],
    )
    assert len(frame) >= 2500
    assert {"risk_model", "exit_family", "time_stop_bars", "mechanism_signature", "modules"}.issubset(frame.columns)
    assert frame["family"].nunique() >= 3
    assert frame["mechanism_signature"].nunique() == len(frame)
    assert frame["risk_model"].astype(str).nunique() >= 3


def test_stage78_default_scope_exposes_full_mechanism_family_set() -> None:
    config = load_config(Path(DEFAULT_CONFIG_PATH))
    scope = resolve_research_scope(config)
    families = set(scope["active_setup_families"])
    assert {
        "structure_pullback_continuation",
        "liquidity_sweep_reversal",
        "squeeze_flow_breakout",
        "failed_breakout_reversal",
        "exhaustion_mean_reversion",
        "funding_oi_imbalance_reversion",
        "volatility_regime_transition",
        "multi_tf_disagreement_repair",
    }.issubset(families)
