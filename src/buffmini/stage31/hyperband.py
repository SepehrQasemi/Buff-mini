"""Hyperband/successive halving scheduler for Stage-31 search budgets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HyperbandConfig:
    rungs: tuple[int, ...] = (0, 1, 2)
    eta: int = 3
    min_candidates: int = 10
    exploration_pct: float = 0.20
    min_exploration_pct: float = 0.15
    seed: int = 42

    def normalized(self) -> "HyperbandConfig":
        eta = int(max(2, int(self.eta)))
        rungs = tuple(sorted({int(v) for v in self.rungs if int(v) >= 0}))
        if not rungs:
            rungs = (0, 1, 2)
        min_candidates = int(max(1, int(self.min_candidates)))
        exploration = float(min(0.95, max(0.0, float(self.exploration_pct))))
        min_explore = float(min(0.95, max(0.0, float(self.min_exploration_pct))))
        exploration = float(max(exploration, min_explore))
        return HyperbandConfig(
            rungs=rungs,
            eta=eta,
            min_candidates=min_candidates,
            exploration_pct=exploration,
            min_exploration_pct=min_explore,
            seed=int(self.seed),
        )


def run_successive_halving(
    candidates: pd.DataFrame,
    *,
    cfg: HyperbandConfig | None = None,
) -> dict[str, Any]:
    conf = (cfg or HyperbandConfig()).normalized()
    rng = np.random.default_rng(int(conf.seed))
    if candidates.empty:
        return {"rungs": {}, "summary": {"initial_count": 0, "final_count": 0, "rung_stats": []}}

    work = candidates.copy().reset_index(drop=True)
    if "candidate_id" not in work.columns:
        work["candidate_id"] = [f"cand_{idx:06d}" for idx in range(work.shape[0])]
    score_col = "stage_a_score" if "stage_a_score" in work.columns else ("fitness" if "fitness" in work.columns else None)
    if score_col is None:
        raise ValueError("candidates must contain stage_a_score or fitness")
    work["score"] = pd.to_numeric(work[score_col], errors="coerce").fillna(0.0)
    pool = work.sort_values(["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)

    rung_outputs: dict[str, pd.DataFrame] = {}
    rung_stats: list[dict[str, Any]] = []
    for rung in conf.rungs:
        budget = int(max(conf.min_candidates, np.ceil(pool.shape[0] / (conf.eta ** int(rung)))))
        budget = int(min(budget, pool.shape[0]))
        if budget <= 0:
            break
        explore_n = int(max(1, round(budget * conf.exploration_pct)))
        exploit_n = int(max(0, budget - explore_n))
        exploit = pool.head(exploit_n).copy()
        exploit["selection"] = "exploit"
        remainder = pool.iloc[exploit_n:].copy()
        if remainder.empty:
            explore = pool.iloc[0:0].copy()
        else:
            picks = rng.permutation(remainder.index.to_numpy(dtype=int))[: min(explore_n, remainder.shape[0])]
            explore = remainder.loc[picks].copy()
        explore["selection"] = "explore"
        selected = pd.concat([exploit, explore], ignore_index=True)
        selected = selected.drop_duplicates(subset=["candidate_id"], keep="first")
        selected = selected.sort_values(["score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)
        actual_exploration = float((selected["selection"] == "explore").mean() if not selected.empty else 0.0)
        rung_outputs[f"rung_{int(rung)}"] = selected.copy()
        rung_stats.append(
            {
                "rung": int(rung),
                "input_count": int(pool.shape[0]),
                "selected_count": int(selected.shape[0]),
                "budget": int(budget),
                "exploration_count": int((selected["selection"] == "explore").sum()),
                "exploration_pct": float(actual_exploration),
            }
        )
        pool = selected.copy()
        if pool.shape[0] <= conf.min_candidates:
            break
    summary = {
        "initial_count": int(work.shape[0]),
        "final_count": int(pool.shape[0]),
        "rung_stats": rung_stats,
    }
    return {"rungs": rung_outputs, "summary": summary}

