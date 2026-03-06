from __future__ import annotations

from buffmini.stage42.self_learning2 import failure_aware_mutation_action


def test_stage42_failure_aware_mutation_zero_raw_signal() -> None:
    action = failure_aware_mutation_action(
        {
            "raw_candidate_count": 0,
            "activation_rate": 0.0,
            "final_trade_count": 0,
            "cost_gate_fail_rate": 0.0,
            "feasibility_fail_rate": 0.0,
            "exp_lcb": 0.0,
            "status": "dead_end",
        }
    )
    assert action == "widen_context_and_expand_grammar"


def test_stage42_failure_aware_mutation_cost_gate_kill() -> None:
    action = failure_aware_mutation_action(
        {
            "raw_candidate_count": 20,
            "activation_rate": 0.4,
            "final_trade_count": 1,
            "cost_gate_fail_rate": 0.7,
            "feasibility_fail_rate": 0.1,
            "exp_lcb": -0.01,
            "status": "active",
        }
    )
    assert action == "mutate_threshold_exit_and_cost_sensitivity"

