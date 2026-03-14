from __future__ import annotations

from buffmini.stage57 import PromotionGates, derive_stage57_verdict


def test_stage57_blocks_passing_verdict_when_decision_evidence_is_proxy_only() -> None:
    verdict = derive_stage57_verdict(
        replay_metrics={"trade_count": 50, "exp_lcb": 0.01, "maxDD": 0.10, "failure_reason_dominance": 0.20},
        walkforward_metrics={"usable_windows": 6, "median_forward_exp_lcb": 0.002},
        monte_carlo_metrics={"conservative_downside_bound": 0.001},
        cross_seed_metrics={"surviving_seeds": 4},
        evidence_records=[
            {
                "candidate_id": "c1",
                "run_id": "r1",
                "config_hash": "cfg",
                "data_hash": "data",
                "seed": 42,
                "metric_name": "exp_lcb",
                "metric_value": 0.01,
                "metric_source_type": "proxy_only",
                "artifact_path": "docs/stage53_summary.json",
                "stage_origin": "stage53",
                "used_for_decision": True,
                "decision_use_allowed": False,
                "evidence_quality": "proxy_only",
                "execution_status": "EXECUTED",
                "validation_state": "PROXY_ONLY",
                "stage_role": "reporting_only",
            }
        ],
        gates=PromotionGates(required_real_sources=("real_replay",)),
    )
    assert verdict["verdict"] == "PARTIAL"
    assert verdict["decision_evidence"]["allowed"] is False
