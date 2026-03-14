"""Stage-62 candidate-realized labels and training dataset v3."""

from __future__ import annotations

import ast
from typing import Any

import pandas as pd


def _load_candidate_id_set(frame: pd.DataFrame) -> set[str]:
    if frame.empty or "candidate_id" not in frame.columns:
        return set()
    return {str(v) for v in frame["candidate_id"].astype(str).tolist() if str(v).strip()}


def build_candidate_outcomes_v3(
    *,
    stage52_candidates: pd.DataFrame,
    stage48_stage_a: pd.DataFrame,
    stage48_stage_b: pd.DataFrame,
    stage53_predictions: pd.DataFrame,
    stage48_ranked: pd.DataFrame | None = None,
) -> pd.DataFrame:
    base = stage52_candidates.copy() if isinstance(stage52_candidates, pd.DataFrame) else pd.DataFrame()
    if base.empty:
        return pd.DataFrame()
    if "candidate_id" not in base.columns:
        base["candidate_id"] = [f"s62_auto_{idx}" for idx in range(len(base))]
    if "source_candidate_id" not in base.columns:
        base["source_candidate_id"] = ""

    pred = stage53_predictions.copy() if isinstance(stage53_predictions, pd.DataFrame) else pd.DataFrame()
    if not pred.empty and "candidate_id" in pred.columns:
        base = base.merge(
            pred.loc[
                :,
                [
                    col
                    for col in (
                        "candidate_id",
                        "tp_before_sl_prob",
                        "expected_net_after_cost",
                        "mae_pct",
                        "mfe_pct",
                        "expected_hold_bars",
                        "replay_priority",
                    )
                    if col in pred.columns
                ],
            ],
            on="candidate_id",
            how="left",
        )
    ranked = stage48_ranked.copy() if isinstance(stage48_ranked, pd.DataFrame) else pd.DataFrame()
    if not ranked.empty and "candidate_id" in ranked.columns:
        ranked = ranked.rename(columns={"candidate_id": "source_candidate_id"})
        keep = [col for col in ("source_candidate_id", "rank_score", "stage_a_score", "layer_score", "replay_worthiness") if col in ranked.columns]
        base = base.merge(ranked.loc[:, keep], on="source_candidate_id", how="left")

    stage_a_ids = _load_candidate_id_set(stage48_stage_a)
    stage_b_ids = _load_candidate_id_set(stage48_stage_b)
    src_ids = base["source_candidate_id"].astype(str)
    base["realized_stage_a"] = src_ids.isin(stage_a_ids).astype(int)
    base["realized_stage_b"] = src_ids.isin(stage_b_ids).astype(int)
    base["realized_label_present"] = ((base["realized_stage_a"] > 0) | (base["realized_stage_b"] > 0)).astype(int)
    base["tp_before_sl_label"] = base["realized_stage_a"].astype(float)
    base["expected_net_after_cost_label"] = (
        base["realized_stage_b"].astype(float) * pd.to_numeric(base.get("cost_edge_proxy", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0001)
        + (1.0 - base["realized_stage_b"].astype(float))
        * pd.to_numeric(base.get("cost_edge_proxy", 0.0), errors="coerce").fillna(0.0).clip(upper=0.0)
    )
    stop_src = base["geometry_stop_distance_pct"] if "geometry_stop_distance_pct" in base.columns else pd.Series([0.005] * len(base), index=base.index)
    target_src = base["geometry_first_target_pct"] if "geometry_first_target_pct" in base.columns else pd.Series([0.01] * len(base), index=base.index)
    hold_src = base["expected_hold_bars"] if "expected_hold_bars" in base.columns else pd.Series([8.0] * len(base), index=base.index)
    base["mae_pct_label"] = -1.0 * pd.to_numeric(stop_src, errors="coerce").fillna(0.005)
    base["mfe_pct_label"] = pd.to_numeric(target_src, errors="coerce").fillna(0.01)
    base["expected_hold_bars_label"] = pd.to_numeric(hold_src, errors="coerce").fillna(8.0)
    # Build richer, non-constant candidate-level features from v2 payload.
    rr_values = []
    geo_stop = []
    geo_first = []
    geo_stretch = []
    zone_span = []
    for rec in base.to_dict(orient="records"):
        rr_model = _safe_parse_dict(rec.get("rr_model", {}))
        geometry = _safe_parse_dict(rec.get("geometry", {}))
        entry_zone = _safe_parse_dict(geometry.get("entry_zone", {}))
        low = float(pd.to_numeric(pd.Series([entry_zone.get("low", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        high = float(pd.to_numeric(pd.Series([entry_zone.get("high", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        mid = max(1e-9, abs((low + high) / 2.0))
        rr_values.append(float(rr_model.get("first_target_rr", 0.0)))
        geo_stop.append(float(geometry.get("stop_distance_pct", 0.0)))
        geo_first.append(float(geometry.get("first_target_pct", 0.0)))
        geo_stretch.append(float(geometry.get("stretch_target_pct", 0.0)))
        zone_span.append(float(abs(high - low) / mid))
    base["rr_first_target"] = rr_values
    expected_hold_src = base["expected_hold_bars"] if "expected_hold_bars" in base.columns else pd.Series([8.0] * len(base), index=base.index)
    base["expected_hold_bars_feature"] = pd.to_numeric(expected_hold_src, errors="coerce").fillna(8.0)
    base["geometry_stop_distance_pct"] = geo_stop
    base["geometry_first_target_pct"] = geo_first
    base["geometry_stretch_target_pct"] = geo_stretch
    base["entry_zone_span_pct"] = zone_span
    family_code = {
        "structure_pullback_continuation": 0.0,
        "liquidity_sweep_reversal": 1.0,
        "squeeze_flow_breakout": 2.0,
    }
    timeframe_code = {"15m": 0.0, "30m": 1.0, "1h": 2.0, "2h": 3.0, "4h": 4.0}
    base["family_code"] = base.get("family", "").astype(str).map(lambda x: family_code.get(str(x), 9.0))
    base["timeframe_code"] = base.get("timeframe", "").astype(str).map(lambda x: timeframe_code.get(str(x), 9.0))
    net_src = base["expected_net_after_cost"] if "expected_net_after_cost" in base.columns else pd.Series([0.0] * len(base), index=base.index)
    rank_src = base["rank_score"] if "rank_score" in base.columns else pd.Series([0.0] * len(base), index=base.index)
    stage_a_src = base["stage_a_score"] if "stage_a_score" in base.columns else pd.Series([0.0] * len(base), index=base.index)
    layer_src = base["layer_score"] if "layer_score" in base.columns else pd.Series([0.0] * len(base), index=base.index)
    worth_src = base["replay_worthiness"] if "replay_worthiness" in base.columns else pd.Series([0.0] * len(base), index=base.index)
    base["exp_lcb_proxy"] = pd.to_numeric(net_src, errors="coerce").fillna(0.0) * 0.25
    base["stage48_rank_score"] = pd.to_numeric(rank_src, errors="coerce").fillna(0.0)
    base["stage48_stage_a_score"] = pd.to_numeric(stage_a_src, errors="coerce").fillna(0.0)
    base["stage48_layer_score"] = pd.to_numeric(layer_src, errors="coerce").fillna(0.0)
    base["stage48_replay_worthiness"] = pd.to_numeric(worth_src, errors="coerce").fillna(0.0)
    return base


def build_training_dataset_v3(outcomes: pd.DataFrame, *, feature_columns: list[str]) -> pd.DataFrame:
    if not isinstance(outcomes, pd.DataFrame) or outcomes.empty:
        return pd.DataFrame()
    work = outcomes.copy()
    work["timestamp"] = pd.date_range("2024-01-01", periods=len(work), freq="h", tz="UTC")
    for col in feature_columns:
        if col not in work.columns:
            work[col] = 0.0
    ordered = [
        "timestamp",
        "candidate_id",
        "source_candidate_id",
        *feature_columns,
        "tp_before_sl_label",
        "expected_net_after_cost_label",
        "mae_pct_label",
        "mfe_pct_label",
        "expected_hold_bars_label",
        "realized_label_present",
    ]
    return work.loc[:, [c for c in ordered if c in work.columns]].copy()


def evaluate_quality_gate_v3(dataset: pd.DataFrame, *, feature_columns: list[str]) -> dict[str, Any]:
    if dataset.empty:
        return {
            "passed": False,
            "label_coverage": 0.0,
            "non_constant_feature_count": 0,
            "feature_non_constant_ok": False,
            "label_variance_ok": False,
            "reason": "empty_dataset",
        }
    label_coverage = float(pd.to_numeric(dataset.get("realized_label_present", 0), errors="coerce").fillna(0).astype(int).mean())
    non_constant = int(
        sum(
            1
            for col in feature_columns
            if col in dataset.columns and int(dataset[col].nunique(dropna=True)) > 1
        )
    )
    label_variance_ok = int(dataset["tp_before_sl_label"].nunique(dropna=True)) > 1
    feature_non_constant_ok = non_constant >= 20
    passed = bool(label_coverage >= 0.35 and label_variance_ok and feature_non_constant_ok)
    reasons: list[str] = []
    if label_coverage < 0.35:
        reasons.append("insufficient_label_coverage")
    if not label_variance_ok:
        reasons.append("label_variance_missing")
    if not feature_non_constant_ok:
        reasons.append("feature_non_constant_below_20")
    return {
        "passed": passed,
        "label_coverage": float(round(label_coverage, 6)),
        "non_constant_feature_count": int(non_constant),
        "feature_non_constant_ok": bool(feature_non_constant_ok),
        "label_variance_ok": bool(label_variance_ok),
        "reason": ",".join(reasons),
    }


def _safe_parse_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    text = str(raw).strip()
    if not text:
        return {}
    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}
