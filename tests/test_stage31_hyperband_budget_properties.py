from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.stage31.hyperband import HyperbandConfig, run_successive_halving


def _candidates(n: int = 180) -> pd.DataFrame:
    rng = np.random.default_rng(12)
    score = rng.normal(loc=0.0, scale=1.0, size=n)
    return pd.DataFrame(
        {
            "candidate_id": [f"cand_{i:04d}" for i in range(n)],
            "stage_a_score": score,
        }
    )


def test_stage31_hyperband_budget_properties() -> None:
    res = run_successive_halving(
        _candidates(),
        cfg=HyperbandConfig(
            rungs=(0, 1, 2, 3),
            eta=3,
            min_candidates=12,
            exploration_pct=0.20,
            min_exploration_pct=0.15,
            seed=42,
        ),
    )
    summary = res["summary"]
    stats = summary["rung_stats"]
    assert summary["initial_count"] == 180
    assert summary["final_count"] > 0
    assert len(stats) >= 2
    prev_selected = None
    for item in stats:
        assert int(item["selected_count"]) <= int(item["input_count"])
        assert float(item["exploration_pct"]) >= 0.15 - 1e-9
        if prev_selected is not None:
            assert int(item["input_count"]) <= int(prev_selected)
        prev_selected = int(item["selected_count"])

