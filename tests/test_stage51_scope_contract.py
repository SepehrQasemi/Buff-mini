from __future__ import annotations

from buffmini.stage51 import build_stage51_summary, resolve_budget_mode, resolve_research_scope


def test_stage51_scope_defaults_and_budget_selection() -> None:
    config = {
        "research_scope": {
            "primary_symbols": ["BTC/USDT"],
            "discovery_timeframes": ["15m", "30m", "1h", "2h", "4h"],
            "promotion_timeframes": 2,
            "final_validation_timeframes": 1,
            "active_setup_families": [
                "structure_pullback_continuation",
                "liquidity_sweep_reversal",
                "squeeze_flow_breakout",
            ],
            "expansion_rules": {"transfer_symbol": "ETH/USDT", "require_stage57_pass": True, "oi_core_enabled": False},
        },
        "budget_mode": {
            "selected": "search",
            "search": {
                "candidate_limit": 60,
                "stage_a_limit": 24,
                "stage_b_limit": 12,
                "micro_replay_limit": 20,
                "full_replay_limit": 8,
                "walkforward_limit": 3,
                "monte_carlo_limit": 2,
                "max_runtime_seconds": 1800,
            },
        },
        "data": {"futures_extras": {"open_interest": {"short_horizon_only": True, "short_horizon_max": "30m"}}},
    }
    scope = resolve_research_scope(config)
    budget = resolve_budget_mode(config)
    summary = build_stage51_summary(config)
    assert scope["promotion_timeframes"] == 2
    assert scope["final_validation_timeframes"] == 1
    assert scope["expansion_rules"]["oi_short_horizon_only"] is True
    assert budget["selected"] == "search"
    assert summary["stage"] == "51"
    assert summary["summary_hash"]
