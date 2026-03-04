"""Optional ML-lite candidate ranker for Stage-28 prioritization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MlRankerConfig:
    enabled: bool = False
    exploration_pct: float = 0.15
    min_exploration_pct: float = 0.10
    alpha: float = 1.0
    max_features: int = 20
    seed: int = 42


def train_ml_ranker(
    candidates: pd.DataFrame,
    *,
    target_col: str = "exp_lcb",
    window_col: str = "window_index",
    cfg: MlRankerConfig | None = None,
) -> dict[str, Any]:
    """Train deterministic ridge-style ranker on candidate-level features."""

    conf = cfg or MlRankerConfig()
    frame = candidates.copy()
    features = _build_feature_frame(frame, max_features=int(conf.max_features))
    y = pd.to_numeric(frame.get(target_col, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if features.empty or y.size == 0:
        return {"features": [], "coef": [], "intercept": 0.0, "feature_importance": {}, "train_rows": 0, "valid_rows": 0}

    if window_col in frame.columns:
        w = pd.to_numeric(frame.get(window_col), errors="coerce")
        threshold = float(w.quantile(0.70)) if w.notna().any() else 0.0
        train_mask = w.fillna(threshold) <= threshold
    else:
        train_mask = pd.Series(True, index=frame.index)

    x = features.to_numpy(dtype=float)
    x_train = x[train_mask.to_numpy(dtype=bool)]
    y_train = y[train_mask.to_numpy(dtype=bool)]
    if x_train.shape[0] == 0:
        x_train = x
        y_train = y

    x_mu = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std = np.where(x_std <= 1e-12, 1.0, x_std)
    x_train_norm = (x_train - x_mu) / x_std
    y_mu = float(y_train.mean()) if y_train.size else 0.0
    y_train_center = y_train - y_mu

    alpha = float(max(1e-9, conf.alpha))
    xtx = x_train_norm.T @ x_train_norm
    ridge = xtx + alpha * np.eye(xtx.shape[0], dtype=float)
    xty = x_train_norm.T @ y_train_center
    coef = np.linalg.solve(ridge, xty)
    importance = np.abs(coef)
    imp_sum = float(np.sum(importance))
    imp_norm = importance / imp_sum if imp_sum > 0 else importance

    return {
        "features": list(features.columns),
        "coef": [float(v) for v in coef.tolist()],
        "intercept": float(y_mu),
        "x_mean": [float(v) for v in x_mu.tolist()],
        "x_std": [float(v) for v in x_std.tolist()],
        "feature_importance": {str(name): float(val) for name, val in zip(features.columns, imp_norm, strict=False)},
        "train_rows": int(x_train.shape[0]),
        "valid_rows": int(x.shape[0]),
    }


def score_with_ml(
    candidates: pd.DataFrame,
    *,
    model: dict[str, Any],
) -> pd.Series:
    """Score candidates with a trained deterministic linear model."""

    features = [str(v) for v in model.get("features", [])]
    if not features:
        return pd.Series(0.0, index=candidates.index, dtype=float)
    x = _build_feature_frame(candidates, feature_subset=features, max_features=len(features))
    if x.empty:
        return pd.Series(0.0, index=candidates.index, dtype=float)
    coef = np.asarray(model.get("coef", []), dtype=float)
    x_mu = np.asarray(model.get("x_mean", []), dtype=float)
    x_std = np.asarray(model.get("x_std", []), dtype=float)
    if coef.size != x.shape[1] or x_mu.size != x.shape[1] or x_std.size != x.shape[1]:
        return pd.Series(0.0, index=candidates.index, dtype=float)
    x_norm = (x.to_numpy(dtype=float) - x_mu) / np.where(x_std <= 1e-12, 1.0, x_std)
    score = x_norm @ coef + float(model.get("intercept", 0.0))
    return pd.Series(score, index=candidates.index, dtype=float)


def prioritize_candidates(
    candidates: pd.DataFrame,
    *,
    budget: int,
    model: dict[str, Any] | None = None,
    cfg: MlRankerConfig | None = None,
) -> pd.DataFrame:
    """Select prioritized candidates with mandatory exploration quota."""

    conf = cfg or MlRankerConfig()
    frame = candidates.copy().reset_index(drop=True)
    if "candidate_id" not in frame.columns:
        frame["candidate_id"] = [f"cand_{idx:06d}" for idx in range(frame.shape[0])]

    if model:
        frame["ml_score"] = score_with_ml(frame, model=model)
    else:
        frame["ml_score"] = _fallback_score(frame)
    frame = frame.sort_values(["ml_score", "candidate_id"], ascending=[False, True]).reset_index(drop=True)

    max_budget = int(max(1, min(int(budget), frame.shape[0])))
    exploration_pct = float(max(conf.min_exploration_pct, conf.exploration_pct))
    exploration_pct = float(min(0.95, max(0.0, exploration_pct)))
    explore_n = int(max(1, round(max_budget * exploration_pct)))
    exploit_n = int(max(0, max_budget - explore_n))

    exploit = frame.head(exploit_n).copy()
    exploit["selection_source"] = "exploit"
    remaining = frame.iloc[exploit_n:].copy()

    rng = np.random.default_rng(int(conf.seed))
    if not remaining.empty:
        perm = rng.permutation(remaining.index.to_numpy(dtype=int))
        picks = perm[: min(explore_n, perm.size)]
        explore = remaining.loc[picks].copy()
    else:
        explore = frame.iloc[0:0].copy()
    explore["selection_source"] = "explore"

    selected = pd.concat([exploit, explore], ignore_index=True)
    selected = selected.drop_duplicates(subset=["candidate_id"], keep="first")
    if selected.shape[0] < max_budget:
        fill = frame.loc[~frame["candidate_id"].isin(selected["candidate_id"])].head(max_budget - selected.shape[0]).copy()
        fill["selection_source"] = "fill"
        selected = pd.concat([selected, fill], ignore_index=True)
    selected = selected.sort_values(["ml_score", "candidate_id"], ascending=[False, True]).head(max_budget).reset_index(drop=True)
    return selected


def _build_feature_frame(
    frame: pd.DataFrame,
    *,
    max_features: int,
    feature_subset: list[str] | None = None,
) -> pd.DataFrame:
    numeric = frame.select_dtypes(include=[np.number]).copy()
    drop_cols = {"window_index"}
    for col in drop_cols:
        if col in numeric.columns:
            numeric = numeric.drop(columns=[col])
    if feature_subset is not None:
        cols = [col for col in feature_subset if col in numeric.columns]
        return numeric.loc[:, cols].copy()
    ordered_cols = sorted(numeric.columns)
    cols = ordered_cols[: int(max(1, min(max_features, len(ordered_cols))))]
    return numeric.loc[:, cols].copy()


def _fallback_score(frame: pd.DataFrame) -> pd.Series:
    exp_lcb = pd.to_numeric(frame.get("exp_lcb", 0.0), errors="coerce").fillna(0.0)
    expectancy = pd.to_numeric(frame.get("expectancy", 0.0), errors="coerce").fillna(0.0)
    trades = pd.to_numeric(frame.get("trades_in_context", 0.0), errors="coerce").fillna(0.0)
    return exp_lcb + 0.25 * expectancy + 0.05 * np.log1p(np.maximum(trades, 0.0))

