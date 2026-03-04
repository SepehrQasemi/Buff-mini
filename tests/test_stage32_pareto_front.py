from __future__ import annotations

import pandas as pd

from buffmini.stage32.pareto import ParetoConfig, pareto_select


def test_stage32_pareto_front() -> None:
    frame = pd.DataFrame(
        [
            {"candidate_id": "a", "exp_lcb": 0.10, "pf_adj": 1.20, "maxdd_p95": 0.20, "repeatability": 0.60, "feasibility_score": 0.70},
            {"candidate_id": "b", "exp_lcb": 0.08, "pf_adj": 1.10, "maxdd_p95": 0.18, "repeatability": 0.58, "feasibility_score": 0.68},
            {"candidate_id": "c", "exp_lcb": 0.12, "pf_adj": 1.25, "maxdd_p95": 0.25, "repeatability": 0.55, "feasibility_score": 0.66},
            {"candidate_id": "d", "exp_lcb": -0.01, "pf_adj": 0.95, "maxdd_p95": 0.30, "repeatability": 0.40, "feasibility_score": 0.50},
        ]
    )
    out = pareto_select(frame, cfg=ParetoConfig(top_k=4))
    assert not out.empty
    assert "pareto_rank" in out.columns
    assert "pareto_score" in out.columns
    # Worst dominated candidate should not be rank 0.
    rank_d = int(out.loc[out["candidate_id"] == "d", "pareto_rank"].iloc[0])
    assert rank_d > 0
    # Best candidates should include at least one of a/c in rank 0.
    rank0_ids = set(out.loc[out["pareto_rank"] == 0, "candidate_id"].tolist())
    assert "a" in rank0_ids or "c" in rank0_ids

