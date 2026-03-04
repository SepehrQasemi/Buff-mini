from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.stage28.budget_funnel import BudgetFunnelConfig, run_budget_funnel


def _candidates(n: int = 100) -> pd.DataFrame:
    rows = []
    for idx in range(n):
        rows.append(
            {
                "candidate_id": f"cand_{idx:03d}",
                "exp_lcb": float(1.0 - idx * 0.01),
                "expectancy": float(0.2 - idx * 0.001),
                "trades_in_context": int(20 + (idx % 15)),
                "context_occurrences": int(40 + (idx % 30)),
                "cost_sensitivity": float((idx % 7) * 0.01),
                "max_drawdown": float((idx % 11) * 0.01),
            }
        )
    return pd.DataFrame(rows)


def test_stage28_funnel_keeps_minimum_exploration_quota() -> None:
    df = _candidates(120)
    out = run_budget_funnel(
        candidates=df,
        signal_map={},
        cfg=BudgetFunnelConfig(stage_b_budget=40, stage_c_budget=20, exploration_pct=0.05, min_exploration_pct=0.10, seed=42),
    )
    stage_b = out["stage_b"]
    assert not stage_b.empty
    explore_count = int((stage_b["stage_b_source"] == "explore").sum())
    explore_pct = float(explore_count / max(1, stage_b.shape[0]))
    assert explore_pct >= 0.10
    summary = dict(out["summary"])
    assert float(summary.get("stage_b_exploration_pct", 0.0)) >= 0.10
    assert int(summary.get("stage_b_exploration_count", 0)) == explore_count

