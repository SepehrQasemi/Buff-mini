"""Stage-53 tradability learning v2 with deterministic stacked ensemble."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from buffmini.stage34.train import TrainConfig, predict_model_proba, train_stage34_models
from buffmini.utils.hashing import stable_hash


_TARGET_COLUMNS: tuple[str, ...] = (
    "tp_before_sl_label",
    "expected_net_after_cost_label",
    "mae_pct_label",
    "mfe_pct_label",
    "expected_hold_bars_label",
)
_FORBIDDEN_FEATURE_TOKENS: tuple[str, ...] = (
    "label",
    "future",
    "tp_before_sl",
    "expected_net_after_cost",
)


def validate_tradability_training_frame(dataset: pd.DataFrame, *, feature_columns: list[str]) -> None:
    if not isinstance(dataset, pd.DataFrame) or dataset.empty:
        raise ValueError("tradability training dataset is empty")
    missing = [col for col in ("timestamp", *_TARGET_COLUMNS) if col not in dataset.columns]
    if missing:
        raise ValueError(f"Missing tradability dataset columns: {missing}")
    if not feature_columns:
        raise ValueError("feature_columns must not be empty")
    forbidden = [name for name in feature_columns if any(token in str(name).lower() for token in _FORBIDDEN_FEATURE_TOKENS)]
    if forbidden:
        raise ValueError(f"feature_columns contain leakage-prone names: {forbidden}")
    for name in feature_columns:
        if str(name) not in dataset.columns:
            raise ValueError(f"feature column missing: {name}")


def fit_tradability_model_v2(
    dataset: pd.DataFrame,
    *,
    feature_columns: list[str],
    seed: int = 42,
    probability_bins: int = 6,
) -> dict[str, Any]:
    validate_tradability_training_frame(dataset, feature_columns=feature_columns)
    work = dataset.copy().sort_values("timestamp").reset_index(drop=True)
    work["label_primary"] = pd.to_numeric(work["tp_before_sl_label"], errors="coerce").fillna(0.0)
    models, train_summary = train_stage34_models(
        work,
        feature_columns=[str(col) for col in feature_columns],
        cfg=TrainConfig(seed=int(seed), models=("logreg", "hgbt", "rf"), calibration="platt"),
    )
    weights = _meta_weights(models)
    base_probs = {
        str(name): predict_model_proba(model, work)
        for name, model in sorted(models.items(), key=lambda kv: str(kv[0]))
    }
    meta_prob = _blend_probabilities(base_probs, weights)
    bucket_stats = _fit_bucket_stats(
        probabilities=meta_prob,
        dataset=work,
        bins=int(max(3, probability_bins)),
    )
    summary = {
        "status": "SUCCESS",
        "feature_columns": [str(col) for col in feature_columns],
        "meta_weights": dict(weights),
        "probability_bins": int(max(3, probability_bins)),
        "train_summary_hash": str(train_summary.get("train_summary_hash", "")),
        "summary_hash": stable_hash(
            {
                "feature_columns": [str(col) for col in feature_columns],
                "meta_weights": dict(weights),
                "probability_bins": int(max(3, probability_bins)),
                "train_summary_hash": str(train_summary.get("train_summary_hash", "")),
            },
            length=16,
        ),
    }
    return {
        "status": "SUCCESS",
        "feature_columns": [str(col) for col in feature_columns],
        "base_models": models,
        "meta_weights": dict(weights),
        "bucket_stats": bucket_stats,
        "summary": summary,
    }


def predict_tradability_model_v2(model_bundle: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(
            columns=[
                "candidate_id",
                "tp_before_sl_prob",
                "expected_net_after_cost",
                "mae_pct",
                "mfe_pct",
                "expected_hold_bars",
                "replay_priority",
            ]
        )
    feature_columns = [str(col) for col in model_bundle.get("feature_columns", [])]
    missing = [col for col in feature_columns if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing feature columns for tradability prediction: {missing}")
    base_models = {str(name): dict(model) for name, model in dict(model_bundle.get("base_models", {})).items()}
    weights = {str(name): float(weight) for name, weight in dict(model_bundle.get("meta_weights", {})).items()}
    base_probs = {
        str(name): predict_model_proba(model, frame)
        for name, model in sorted(base_models.items(), key=lambda kv: str(kv[0]))
    }
    meta_prob = _blend_probabilities(base_probs, weights)
    expected_net = _lookup_bucket_value(meta_prob, model_bundle, target="expected_net_after_cost")
    mae_pct = _lookup_bucket_value(meta_prob, model_bundle, target="mae_pct")
    mfe_pct = _lookup_bucket_value(meta_prob, model_bundle, target="mfe_pct")
    expected_hold = _lookup_bucket_value(meta_prob, model_bundle, target="expected_hold_bars")
    rr = _extract_rr_series(frame)
    cost_edge = pd.to_numeric(frame.get("cost_edge_proxy", 0.0), errors="coerce").fillna(0.0).astype(float)
    replay_priority = (
        (meta_prob * 0.50)
        + (np.where(expected_net > 0.0, 0.20, 0.0))
        + (np.clip(rr / 3.0, 0.0, 1.0) * 0.15)
        + (np.clip(cost_edge * 100.0, 0.0, 1.0) * 0.15)
    )
    payload = pd.DataFrame(
        {
            "candidate_id": frame.get("candidate_id", pd.Series(range(len(frame)))).astype(str),
            "tp_before_sl_prob": np.clip(meta_prob, 1e-6, 1.0 - 1e-6),
            "expected_net_after_cost": expected_net.astype(float),
            "mae_pct": mae_pct.astype(float),
            "mfe_pct": mfe_pct.astype(float),
            "expected_hold_bars": np.clip(expected_hold, 1.0, None).astype(float),
            "replay_priority": np.clip(replay_priority, 0.0, 2.0).astype(float),
        }
    )
    return payload.sort_values(["replay_priority", "candidate_id"], ascending=[False, True]).reset_index(drop=True)


def route_tradability_v2(
    candidates: pd.DataFrame,
    *,
    predictions: pd.DataFrame,
    stage_a_prob_threshold: float = 0.55,
    min_rr: float = 1.5,
    min_cost_edge: float = 0.0,
    hold_bar_ceiling: float = 24.0,
) -> dict[str, Any]:
    if not isinstance(candidates, pd.DataFrame) or candidates.empty:
        return {
            "stage_a_survivors": pd.DataFrame(),
            "stage_b_survivors": pd.DataFrame(),
            "counts": {"input": 0, "stage_a": 0, "stage_b": 0},
            "stage_a_threshold": float(stage_a_prob_threshold),
        }
    merged = candidates.merge(predictions, on="candidate_id", how="left", validate="one_to_one")
    merged["tp_before_sl_prob"] = pd.to_numeric(_pick_col(merged, "tp_before_sl_prob"), errors="coerce").fillna(0.0)
    merged["expected_net_after_cost"] = pd.to_numeric(_pick_col(merged, "expected_net_after_cost"), errors="coerce").fillna(0.0)
    merged["expected_hold_bars"] = pd.to_numeric(_pick_col(merged, "expected_hold_bars"), errors="coerce").fillna(0.0)
    merged["replay_priority"] = pd.to_numeric(_pick_col(merged, "replay_priority"), errors="coerce").fillna(0.0)
    rr = _extract_rr_series(merged)
    cost_edge = pd.to_numeric(merged.get("cost_edge_proxy", 0.0), errors="coerce").fillna(0.0)
    priority_floor = float(merged["replay_priority"].median()) if not merged.empty else 0.0

    stage_a_mask = (
        (merged["tp_before_sl_prob"] >= float(stage_a_prob_threshold))
        & (merged["expected_net_after_cost"] > 0.0)
        & (rr >= float(min_rr))
        & (cost_edge > float(min_cost_edge))
        & (merged["expected_hold_bars"] <= float(hold_bar_ceiling))
        & (merged["replay_priority"] >= priority_floor)
    )
    stage_a = merged.loc[stage_a_mask, :].copy()
    stage_b_mask = stage_a_mask & (pd.to_numeric(merged.get("exp_lcb_proxy", 0.0), errors="coerce").fillna(0.0) > 0.0)
    stage_b = merged.loc[stage_b_mask, :].copy()
    return {
        "stage_a_survivors": stage_a.reset_index(drop=True),
        "stage_b_survivors": stage_b.reset_index(drop=True),
        "counts": {"input": int(len(merged)), "stage_a": int(len(stage_a)), "stage_b": int(len(stage_b))},
        "stage_a_threshold": float(stage_a_prob_threshold),
        "replay_priority_floor": float(priority_floor),
    }


def _meta_weights(models: dict[str, dict[str, Any]]) -> dict[str, float]:
    raw: dict[str, float] = {}
    for name, model in sorted(models.items(), key=lambda kv: str(kv[0])):
        summary = dict(model.get("summary", {}))
        loss = max(float(summary.get("val_logloss", 1.0)), 1e-6)
        raw[str(name)] = 1.0 / loss
    total = max(sum(raw.values()), 1e-9)
    return {name: float(round(weight / total, 8)) for name, weight in raw.items()}


def _blend_probabilities(base_probs: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    if not base_probs:
        return np.asarray([], dtype=float)
    names = [str(name) for name in sorted(base_probs.keys())]
    stacked = np.vstack([np.asarray(base_probs[name], dtype=float) for name in names])
    weight_vec = np.asarray([float(weights.get(name, 0.0)) for name in names], dtype=float)
    if float(weight_vec.sum()) <= 0.0:
        weight_vec = np.full(len(names), 1.0 / max(1, len(names)), dtype=float)
    weight_vec = weight_vec / max(float(weight_vec.sum()), 1e-9)
    return np.average(stacked, axis=0, weights=weight_vec)


def _fit_bucket_stats(*, probabilities: np.ndarray, dataset: pd.DataFrame, bins: int) -> list[dict[str, float]]:
    probs = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    quantiles = np.unique(np.quantile(probs, np.linspace(0.0, 1.0, int(max(3, bins)) + 1)))
    if quantiles.size < 2:
        quantiles = np.asarray([0.0, 1.0], dtype=float)
    rows: list[dict[str, float]] = []
    for idx in range(len(quantiles) - 1):
        lo = float(quantiles[idx])
        hi = float(quantiles[idx + 1])
        mask = (probs >= lo) & (probs <= hi if idx == len(quantiles) - 2 else probs < hi)
        local = dataset.loc[mask, :]
        if local.empty:
            continue
        rows.append(
            {
                "lo": lo,
                "hi": hi,
                "expected_net_after_cost": float(pd.to_numeric(local["expected_net_after_cost_label"], errors="coerce").fillna(0.0).mean()),
                "mae_pct": float(pd.to_numeric(local["mae_pct_label"], errors="coerce").fillna(0.0).mean()),
                "mfe_pct": float(pd.to_numeric(local["mfe_pct_label"], errors="coerce").fillna(0.0).mean()),
                "expected_hold_bars": float(pd.to_numeric(local["expected_hold_bars_label"], errors="coerce").fillna(0.0).mean()),
            }
        )
    return rows


def _lookup_bucket_value(probabilities: np.ndarray, model_bundle: dict[str, Any], *, target: str) -> pd.Series:
    buckets = list(model_bundle.get("bucket_stats", []))
    values: list[float] = []
    for prob in np.asarray(probabilities, dtype=float):
        assigned = None
        for bucket in buckets:
            lo = float(bucket.get("lo", 0.0))
            hi = float(bucket.get("hi", 1.0))
            if lo <= float(prob) <= hi:
                assigned = float(bucket.get(target, 0.0))
                break
        if assigned is None:
            assigned = float(buckets[-1].get(target, 0.0)) if buckets else 0.0
        values.append(float(assigned))
    return pd.Series(values, dtype=float)


def _extract_rr_series(frame: pd.DataFrame) -> pd.Series:
    if "first_target_rr" in frame.columns:
        return pd.to_numeric(frame["first_target_rr"], errors="coerce").fillna(0.0).astype(float)
    if "rr_model" not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    values: list[float] = []
    for raw in frame["rr_model"].tolist():
        if isinstance(raw, dict):
            values.append(float(raw.get("first_target_rr", 0.0)))
        else:
            values.append(0.0)
    return pd.Series(values, index=frame.index, dtype=float)


def _pick_col(frame: pd.DataFrame, base_name: str) -> pd.Series:
    for name in (base_name, f"{base_name}_y", f"{base_name}_x"):
        if name in frame.columns:
            return frame[name]
    return pd.Series([0.0] * len(frame), index=frame.index, dtype=float)
