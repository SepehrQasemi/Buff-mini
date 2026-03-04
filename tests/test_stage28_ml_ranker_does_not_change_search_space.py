from __future__ import annotations

import pandas as pd

from buffmini.stage28.ml_ranker import MlRankerConfig, prioritize_candidates, train_ml_ranker


def _candidates(n: int = 80) -> pd.DataFrame:
    rows = []
    for idx in range(n):
        rows.append(
            {
                "candidate_id": f"cand_{idx:03d}",
                "window_index": idx % 12,
                "exp_lcb": float(0.8 - idx * 0.01),
                "expectancy": float(0.2 - idx * 0.001),
                "trades_in_context": int(20 + (idx % 10)),
                "context_occurrences": int(45 + (idx % 20)),
                "cost_sensitivity": float((idx % 6) * 0.01),
                "max_drawdown": float((idx % 12) * 0.01),
            }
        )
    return pd.DataFrame(rows)


def test_stage28_ml_ranker_does_not_change_search_space() -> None:
    frame = _candidates(90)
    cfg = MlRankerConfig(enabled=True, exploration_pct=0.15, min_exploration_pct=0.10, alpha=1.0, max_features=20, seed=42)
    model = train_ml_ranker(frame, cfg=cfg)
    selected = prioritize_candidates(frame, budget=30, model=model, cfg=cfg)

    selected_ids = set(selected["candidate_id"].astype(str).tolist())
    input_ids = set(frame["candidate_id"].astype(str).tolist())
    assert selected_ids.issubset(input_ids)
    assert int(selected.shape[0]) == 30

    explore_count = int((selected["selection_source"] == "explore").sum())
    assert explore_count >= 3  # >= 10% exploration floor for budget=30

