from __future__ import annotations

from buffmini.research.reality import build_gate_sensitivity, infer_dominant_blockers


def test_stage85_reality_sensitivity_and_blockers() -> None:
    fake_results = {
        "synthetic_clean_easy": {
            "evaluations": [{"replay_trade_count": 50, "replay_exp_lcb": 0.01, "walkforward_usable_windows": 4, "transfer_classification": "transferable"}],
            "dominant_failure_reasons": {"ranking_filter": 1},
            "blocked_count": 0,
        },
        "live_strict": {
            "evaluations": [{"replay_trade_count": 10, "replay_exp_lcb": -0.01, "walkforward_usable_windows": 1, "transfer_classification": "not_transferable", "first_death_stage": "replay"}],
            "dominant_failure_reasons": {"gap_count=2,largest_gap_bars=2,max_gap_bars=0": 2, "transfer::regime_mismatch": 1},
            "blocked_count": 1,
        },
    }
    config = {
        "promotion_gates": {
            "replay": {"min_trade_count": 40, "min_exp_lcb": 0.0},
            "walkforward": {"min_usable_windows": 3},
        }
    }
    sensitivity = build_gate_sensitivity(results=fake_results, config=config)
    assert sensitivity["replay"]["survivor_counts"]["synthetic_clean_easy"] == 1
    assert sensitivity["continuity"]["blocked_counts"]["live_strict"] == 1
    blockers = infer_dominant_blockers(results=fake_results)
    assert blockers[0]["blocker"] in {"data_canonicalization", "transfer_limitation", "ranking_funnel_pressure"}
