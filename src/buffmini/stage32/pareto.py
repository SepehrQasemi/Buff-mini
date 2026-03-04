"""Pareto front utilities for Stage-32 validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ParetoConfig:
    top_k: int = 30


def pareto_select(candidates: pd.DataFrame, *, cfg: ParetoConfig | None = None) -> pd.DataFrame:
    conf = cfg or ParetoConfig()
    if candidates.empty:
        return candidates.copy()
    work = candidates.copy().reset_index(drop=True)
    _ensure_columns(work)
    score_cols = [
        "exp_lcb",
        "pf_adj",
        "repeatability",
        "feasibility_score",
    ]
    min_cols = ["maxdd_p95"]
    # Normalize with min-max to obtain stable tie-breaker score.
    norm = pd.DataFrame(index=work.index)
    for col in score_cols:
        norm[col] = _minmax(pd.to_numeric(work[col], errors="coerce").fillna(0.0))
    for col in min_cols:
        norm[col] = 1.0 - _minmax(pd.to_numeric(work[col], errors="coerce").fillna(0.0))
    work["pareto_score"] = norm.mean(axis=1)

    values = work.loc[:, score_cols + min_cols].to_numpy(dtype=float)
    maximize_mask = np.array([True, True, True, True, False], dtype=bool)
    ranks = _non_dominated_sort(values, maximize_mask)
    work["pareto_rank"] = ranks
    work = work.sort_values(["pareto_rank", "pareto_score"], ascending=[True, False]).reset_index(drop=True)
    return work.head(int(max(1, conf.top_k))).copy()


def _ensure_columns(frame: pd.DataFrame) -> None:
    defaults = {
        "exp_lcb": 0.0,
        "pf_adj": 0.0,
        "maxdd_p95": 1.0,
        "repeatability": 0.0,
        "feasibility_score": 0.0,
    }
    for key, val in defaults.items():
        if key not in frame.columns:
            frame[key] = val


def _minmax(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)
    lo = float(x.min())
    hi = float(x.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 1e-12:
        return pd.Series(0.5, index=x.index, dtype=float)
    return (x - lo) / (hi - lo)


def _dominates(a: np.ndarray, b: np.ndarray, maximize_mask: np.ndarray) -> bool:
    better_or_equal = True
    strictly_better = False
    for idx in range(len(a)):
        if maximize_mask[idx]:
            if a[idx] < b[idx]:
                better_or_equal = False
                break
            if a[idx] > b[idx]:
                strictly_better = True
        else:
            if a[idx] > b[idx]:
                better_or_equal = False
                break
            if a[idx] < b[idx]:
                strictly_better = True
    return bool(better_or_equal and strictly_better)


def _non_dominated_sort(values: np.ndarray, maximize_mask: np.ndarray) -> np.ndarray:
    n = int(values.shape[0])
    dominates_list: list[list[int]] = [[] for _ in range(n)]
    dominated_count = np.zeros(n, dtype=int)
    rank = np.full(n, fill_value=-1, dtype=int)
    fronts: list[list[int]] = [[]]

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(values[i], values[j], maximize_mask):
                dominates_list[i].append(j)
            elif _dominates(values[j], values[i], maximize_mask):
                dominated_count[i] += 1
        if dominated_count[i] == 0:
            rank[i] = 0
            fronts[0].append(i)

    front_idx = 0
    while front_idx < len(fronts) and fronts[front_idx]:
        next_front: list[int] = []
        for i in fronts[front_idx]:
            for j in dominates_list[i]:
                dominated_count[j] -= 1
                if dominated_count[j] == 0:
                    rank[j] = front_idx + 1
                    next_front.append(j)
        if next_front:
            fronts.append(next_front)
        front_idx += 1

    rank = np.where(rank < 0, int(np.max(rank[rank >= 0]) + 1 if np.any(rank >= 0) else 0), rank)
    return rank.astype(int)

