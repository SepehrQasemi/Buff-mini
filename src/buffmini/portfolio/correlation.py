"""Correlation and diversification helpers for Stage-2."""

from __future__ import annotations

import math
from typing import Iterable

import pandas as pd


DEFAULT_CORR_MIN_SUBSET_SIZE = 5


def build_correlation_matrix(return_series_by_candidate: dict[str, pd.Series]) -> pd.DataFrame:
    """Build a pairwise correlation matrix using overlapping timestamps only."""

    if not return_series_by_candidate:
        return pd.DataFrame()

    aligned = pd.concat(return_series_by_candidate, axis=1).sort_index()
    return aligned.corr(min_periods=1)


def average_correlation(matrix: pd.DataFrame) -> float:
    """Return the mean off-diagonal correlation."""

    if matrix.empty or len(matrix.columns) < 2:
        return 0.0

    mask = ~pd.DataFrame(
        [[i == j for j in range(len(matrix.columns))] for i in range(len(matrix.index))],
        index=matrix.index,
        columns=matrix.columns,
    )
    values = matrix.where(mask).stack().astype(float)
    if values.empty:
        return 0.0
    return float(values.mean())


def effective_number_of_strategies(weights: dict[str, float]) -> float:
    """Compute effective strategy count from portfolio weights."""

    if not weights:
        return 0.0
    squared_sum = sum(float(weight) ** 2 for weight in weights.values())
    if squared_sum <= 0:
        return 0.0
    return float(1.0 / squared_sum)


def select_correlation_minimized_subset(
    candidate_ids: Iterable[str],
    correlation_matrix: pd.DataFrame,
    effective_edge: dict[str, float],
    subset_size: int | None = None,
) -> list[str]:
    """Greedy low-correlation subset selection seeded by best effective edge."""

    ordered_candidates = sorted(
        {str(candidate_id) for candidate_id in candidate_ids},
        key=lambda candidate_id: float(effective_edge.get(candidate_id, float("-inf"))),
        reverse=True,
    )
    if not ordered_candidates:
        return []

    resolved_size = int(subset_size or min(DEFAULT_CORR_MIN_SUBSET_SIZE, len(ordered_candidates)))
    resolved_size = max(1, min(resolved_size, len(ordered_candidates)))

    selected = [ordered_candidates[0]]
    remaining = [candidate_id for candidate_id in ordered_candidates[1:]]

    while remaining and len(selected) < resolved_size:
        best_candidate = min(
            remaining,
            key=lambda candidate_id: _average_candidate_correlation(
                candidate_id=candidate_id,
                selected=selected,
                correlation_matrix=correlation_matrix,
            ),
        )
        selected.append(best_candidate)
        remaining.remove(best_candidate)

    return selected


def normalize_weights(raw_weights: dict[str, float]) -> dict[str, float]:
    """Normalize non-negative weights to sum to one."""

    cleaned = {candidate_id: max(0.0, float(weight)) for candidate_id, weight in raw_weights.items()}
    total = sum(cleaned.values())
    if total <= 0:
        count = len(cleaned)
        if count == 0:
            return {}
        return {candidate_id: 1.0 / count for candidate_id in cleaned}
    return {candidate_id: float(weight) / total for candidate_id, weight in cleaned.items()}


def _average_candidate_correlation(
    candidate_id: str,
    selected: list[str],
    correlation_matrix: pd.DataFrame,
) -> float:
    correlations: list[float] = []
    for selected_id in selected:
        if candidate_id not in correlation_matrix.index or selected_id not in correlation_matrix.columns:
            correlations.append(0.0)
            continue
        value = correlation_matrix.loc[candidate_id, selected_id]
        if pd.isna(value):
            correlations.append(0.0)
        else:
            correlations.append(float(value))
    if not correlations:
        return math.inf
    return float(sum(correlations) / len(correlations))
