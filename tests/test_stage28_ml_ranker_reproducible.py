from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.stage28.ml_ranker import MlRankerConfig, prioritize_candidates, train_ml_ranker


def _candidates(n: int = 64) -> pd.DataFrame:
    values = np.arange(float(n))
    return pd.DataFrame(
        {
            "candidate_id": [f"cand_{idx:03d}" for idx in range(n)],
            "window_index": [idx % 16 for idx in range(n)],
            "exp_lcb": 0.5 - values * 0.007,
            "expectancy": 0.1 - values * 0.001,
            "trades_in_context": 15 + (values % 12),
            "context_occurrences": 40 + (values % 18),
            "cost_sensitivity": (values % 5) * 0.01,
            "max_drawdown": (values % 9) * 0.01,
        }
    )


def test_stage28_ml_ranker_reproducible() -> None:
    frame = _candidates(64)
    cfg = MlRankerConfig(enabled=True, exploration_pct=0.15, min_exploration_pct=0.10, alpha=0.7, max_features=20, seed=42)
    model_a = train_ml_ranker(frame, cfg=cfg)
    model_b = train_ml_ranker(frame, cfg=cfg)
    assert model_a["features"] == model_b["features"]
    assert model_a["coef"] == model_b["coef"]
    assert float(model_a["intercept"]) == float(model_b["intercept"])

    sel_a = prioritize_candidates(frame, budget=20, model=model_a, cfg=cfg)
    sel_b = prioritize_candidates(frame, budget=20, model=model_b, cfg=cfg)
    assert sel_a["candidate_id"].astype(str).tolist() == sel_b["candidate_id"].astype(str).tolist()
    assert sel_a["selection_source"].astype(str).tolist() == sel_b["selection_source"].astype(str).tolist()
    assert np.allclose(sel_a["ml_score"].to_numpy(dtype=float), sel_b["ml_score"].to_numpy(dtype=float))

