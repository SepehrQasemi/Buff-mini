from __future__ import annotations

import pandas as pd

from buffmini.stage28.budget_funnel import BudgetFunnelConfig, run_budget_funnel


def test_stage28_diversity_constraint_filters_near_duplicates() -> None:
    candidates = pd.DataFrame(
        [
            {"candidate_id": "A", "exp_lcb": 1.0, "expectancy": 0.4, "trades_in_context": 60, "context_occurrences": 80, "cost_sensitivity": 0.01, "max_drawdown": 0.10},
            {"candidate_id": "B", "exp_lcb": 0.99, "expectancy": 0.39, "trades_in_context": 59, "context_occurrences": 79, "cost_sensitivity": 0.01, "max_drawdown": 0.11},
            {"candidate_id": "C", "exp_lcb": 0.98, "expectancy": 0.38, "trades_in_context": 58, "context_occurrences": 78, "cost_sensitivity": 0.02, "max_drawdown": 0.12},
            {"candidate_id": "D", "exp_lcb": 0.97, "expectancy": 0.37, "trades_in_context": 57, "context_occurrences": 77, "cost_sensitivity": 0.02, "max_drawdown": 0.13},
            {"candidate_id": "E", "exp_lcb": 0.50, "expectancy": 0.20, "trades_in_context": 40, "context_occurrences": 55, "cost_sensitivity": 0.05, "max_drawdown": 0.16},
        ]
    )
    signal_map = {
        "A": pd.Series([1, 1, 0, 0, 1, 1, 0, 0], dtype=float),
        "B": pd.Series([1, 1, 0, 0, 1, 1, 0, 0], dtype=float),  # near duplicate of A
        "C": pd.Series([0, 0, 1, 1, 0, 0, 1, 1], dtype=float),
        "D": pd.Series([0, 0, 1, 1, 0, 0, 1, 1], dtype=float),  # near duplicate of C
        "E": pd.Series([1, 0, -1, 0, 1, 0, -1, 0], dtype=float),
    }
    out = run_budget_funnel(
        candidates=candidates,
        signal_map=signal_map,
        cfg=BudgetFunnelConfig(stage_b_budget=5, stage_c_budget=4, exploration_pct=0.20, sim_threshold=0.90, seed=42),
    )
    stage_c = out["stage_c"]
    ids = set(stage_c["candidate_id"].astype(str).tolist())
    assert not ({"A", "B"} <= ids)
    assert not ({"C", "D"} <= ids)

