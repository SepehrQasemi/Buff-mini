from __future__ import annotations

from buffmini.research.robustness import summarize_layered_robustness


def test_stage80_layered_robustness_reaches_forward_plausibility_when_mc_not_ready() -> None:
    summary = summarize_layered_robustness(
        replay_metrics={"trade_count": 24, "exp_lcb": 0.002, "maxDD": 0.08},
        walkforward_summary={
            "usable_windows": 4,
            "p10_forward_expectancy": 0.0005,
            "positive_window_fraction": 0.50,
            "degradation_vs_holdout": -0.002,
            "worst_window_expectancy": -0.001,
            "p10_forward_profit_factor": 1.02,
            "dispersion": {"forward_expectancy_iqr": 0.002},
        },
        monte_carlo={
            "execution_status": "EXECUTED",
            "conservative_downside_bound": -0.04,
            "worst_case_drawdown_p95": 0.45,
            "scenario_rows": [{"scenario": "cost_stress", "passed": False}],
        },
        perturbation={"execution_status": "EXECUTED", "surviving_seeds": 4, "rows": []},
        split_perturbation={"execution_status": "EXECUTED", "summary": {"usable_windows": 3, "p10_forward_expectancy": 0.0001}},
        config={"promotion_gates": {"replay": {"min_exp_lcb": 0.0, "max_drawdown": 0.2}, "walkforward": {"min_usable_windows": 3}, "cross_seed": {"min_passing_seeds": 3}}},
    )
    assert summary["level_reached"] == 2
    assert summary["level_name"] == "forward_plausibility"
    assert summary["stop_reason"] == "full_robustness_not_met"
