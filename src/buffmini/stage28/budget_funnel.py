"""Stage-28 budgeted discovery funnel with exploration and diversity guards."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BudgetFunnelConfig:
    stage_b_top_pct: float = 0.35
    stage_b_budget: int = 60
    stage_c_budget: int = 25
    exploration_pct: float = 0.15
    sim_threshold: float = 0.90
    min_exploration_pct: float = 0.10
    seed: int = 42

    def normalized(self) -> "BudgetFunnelConfig":
        b_top = float(min(1.0, max(0.05, self.stage_b_top_pct)))
        b_budget = int(max(1, self.stage_b_budget))
        c_budget = int(max(1, self.stage_c_budget))
        exp_pct = float(max(self.min_exploration_pct, self.exploration_pct))
        exp_pct = float(min(0.95, exp_pct))
        sim = float(min(1.0, max(0.0, self.sim_threshold)))
        return BudgetFunnelConfig(
            stage_b_top_pct=b_top,
            stage_b_budget=b_budget,
            stage_c_budget=c_budget,
            exploration_pct=exp_pct,
            sim_threshold=sim,
            min_exploration_pct=float(self.min_exploration_pct),
            seed=int(self.seed),
        )


def stage_a_score(candidates: pd.DataFrame) -> pd.Series:
    """Compute a cheap deterministic score for Stage A ranking."""

    df = candidates.copy()
    exp_lcb = pd.to_numeric(df.get("exp_lcb", 0.0), errors="coerce").fillna(0.0)
    exp = pd.to_numeric(df.get("expectancy", 0.0), errors="coerce").fillna(0.0)
    trades = pd.to_numeric(df.get("trades_in_context", 0.0), errors="coerce").fillna(0.0)
    occ = pd.to_numeric(df.get("context_occurrences", 0.0), errors="coerce").fillna(0.0)
    cost_sens = pd.to_numeric(df.get("cost_sensitivity", 0.0), errors="coerce").fillna(0.0)
    evidence = np.log1p(np.maximum(0.0, trades)) + 0.2 * np.log1p(np.maximum(0.0, occ))
    return exp_lcb + 0.20 * exp + 0.05 * evidence - 0.10 * np.abs(cost_sens)


def run_budget_funnel(
    *,
    candidates: pd.DataFrame,
    signal_map: dict[str, pd.Series] | None = None,
    cfg: BudgetFunnelConfig | None = None,
) -> dict[str, Any]:
    """Run Stage A/B/C candidate funnel with exploration and diversity."""

    if candidates.empty:
        return {
            "stage_a": pd.DataFrame(),
            "stage_b": pd.DataFrame(),
            "stage_c": pd.DataFrame(),
            "summary": {
                "stage_a_count": 0,
                "stage_b_count": 0,
                "stage_c_count": 0,
                "stage_b_exploration_count": 0,
                "stage_b_exploration_pct": 0.0,
            },
        }

    conf = (cfg or BudgetFunnelConfig()).normalized()
    rng = np.random.default_rng(int(conf.seed))
    df = candidates.copy().reset_index(drop=True)
    if "candidate_id" not in df.columns:
        df["candidate_id"] = [f"cand_{idx:06d}" for idx in range(df.shape[0])]
    df["stage_a_score"] = stage_a_score(df)
    stage_a = df.sort_values(["stage_a_score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)

    top_n = int(max(1, round(float(conf.stage_b_top_pct) * stage_a.shape[0])))
    top_pool = stage_a.head(top_n).copy()
    remainder = stage_a.iloc[top_n:].copy()
    stage_b_budget = int(min(conf.stage_b_budget, stage_a.shape[0]))
    explore_n = int(max(1, round(float(stage_b_budget) * float(conf.exploration_pct))))
    exploit_n = int(max(0, stage_b_budget - explore_n))

    exploit = top_pool.head(exploit_n).copy()
    exploit["stage_b_source"] = "exploit"

    if remainder.empty:
        explore_pool = top_pool.iloc[exploit_n:].copy()
    else:
        explore_pool = remainder.copy()

    if not explore_pool.empty:
        choices = rng.permutation(explore_pool.index.to_numpy(dtype=int))
        picked = choices[: min(explore_n, choices.size)]
        explore = explore_pool.loc[picked].copy()
    else:
        explore = top_pool.iloc[0:0].copy()
    explore["stage_b_source"] = "explore"

    stage_b = pd.concat([exploit, explore], ignore_index=True)
    stage_b = stage_b.drop_duplicates(subset=["candidate_id"], keep="first")
    stage_b = stage_b.sort_values(["stage_a_score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)

    diversity_applied = _apply_diversity(
        frame=stage_b,
        signal_map=signal_map or {},
        sim_threshold=float(conf.sim_threshold),
        budget=int(conf.stage_c_budget),
    )
    stage_c = diversity_applied.reset_index(drop=True)
    stage_c["stage_c_score"] = _stage_c_score(stage_c)
    stage_c = stage_c.sort_values(["stage_c_score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)

    stage_b_explore = int((stage_b.get("stage_b_source", pd.Series(dtype=str)) == "explore").sum())
    stage_b_count = int(stage_b.shape[0])
    summary = {
        "stage_a_count": int(stage_a.shape[0]),
        "stage_b_count": stage_b_count,
        "stage_c_count": int(stage_c.shape[0]),
        "stage_b_exploration_count": int(stage_b_explore),
        "stage_b_exploration_pct": float(stage_b_explore / max(1, stage_b_count)),
        "sim_threshold": float(conf.sim_threshold),
        "exploration_pct_configured": float(conf.exploration_pct),
    }
    return {"stage_a": stage_a, "stage_b": stage_b, "stage_c": stage_c, "summary": summary}


def _stage_c_score(frame: pd.DataFrame) -> pd.Series:
    exp_lcb = pd.to_numeric(frame.get("exp_lcb", 0.0), errors="coerce").fillna(0.0)
    expectancy = pd.to_numeric(frame.get("expectancy", 0.0), errors="coerce").fillna(0.0)
    drawdown = pd.to_numeric(frame.get("max_drawdown", 0.0), errors="coerce").fillna(0.0)
    return exp_lcb + 0.25 * expectancy - 0.15 * drawdown


def _apply_diversity(
    *,
    frame: pd.DataFrame,
    signal_map: dict[str, pd.Series],
    sim_threshold: float,
    budget: int,
) -> pd.DataFrame:
    if frame.empty:
        return frame
    kept_rows: list[pd.Series] = []
    kept_ids: list[str] = []
    for _, row in frame.iterrows():
        cand_id = str(row.get("candidate_id", ""))
        if not cand_id:
            continue
        keep = True
        for existing in kept_ids:
            sim = _signal_similarity(signal_map.get(cand_id), signal_map.get(existing))
            if sim > float(sim_threshold):
                keep = False
                break
        if keep:
            kept_rows.append(row)
            kept_ids.append(cand_id)
        if len(kept_rows) >= int(budget):
            break
    if not kept_rows:
        return frame.head(int(max(1, budget))).copy()
    return pd.DataFrame(kept_rows).reset_index(drop=True)


def _signal_similarity(signal_a: pd.Series | None, signal_b: pd.Series | None) -> float:
    if signal_a is None or signal_b is None:
        return 0.0
    a = pd.to_numeric(signal_a, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    b = pd.to_numeric(signal_b, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    n = int(min(len(a), len(b)))
    if n <= 1:
        return 0.0
    a = a[:n]
    b = b[:n]
    a_std = float(np.std(a))
    b_std = float(np.std(b))
    if a_std <= 1e-12 or b_std <= 1e-12:
        nonzero_union = np.count_nonzero((a != 0.0) | (b != 0.0))
        if nonzero_union == 0:
            return 1.0
        overlap = np.count_nonzero((a != 0.0) & (b != 0.0))
        return float(overlap / nonzero_union)
    corr = float(np.corrcoef(a, b)[0, 1])
    if not np.isfinite(corr):
        return 0.0
    return float(abs(corr))

