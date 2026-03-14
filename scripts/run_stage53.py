"""Run Stage-53 tradability learning v2 using candidate-aligned Stage-52/48 artifacts."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.stage53 import fit_tradability_model_v2, predict_tradability_model_v2, route_tradability_v2
from buffmini.utils.hashing import stable_hash
from buffmini.validation import hydrate_candidate_record, resolve_validation_candidate, run_candidate_replay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-53 tradability learning v2")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--stage28-run-id", type=str, default="")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _resolve_stage28_run_id(args: argparse.Namespace, docs_dir: Path) -> str:
    if str(args.stage28_run_id).strip():
        return str(args.stage28_run_id).strip()
    stage52 = _load_json(docs_dir / "stage52_summary.json")
    if str(stage52.get("stage28_run_id", "")).strip():
        return str(stage52["stage28_run_id"]).strip()
    stage39 = _load_json(docs_dir / "stage39_signal_generation_summary.json")
    return str(stage39.get("stage28_run_id", "")).strip()


def _safe_parse_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    text = str(raw).strip()
    if not text:
        return {}
    try:
        payload = ast.literal_eval(text)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _timeframe_code(timeframe: str) -> float:
    order = {"15m": 0.0, "30m": 1.0, "1h": 2.0, "2h": 3.0, "4h": 4.0}
    return float(order.get(str(timeframe), 5.0))


def _family_code(family: str) -> float:
    order = {
        "structure_pullback_continuation": 0.0,
        "liquidity_sweep_reversal": 1.0,
        "squeeze_flow_breakout": 2.0,
    }
    return float(order.get(str(family), 9.0))


def _entry_zone_span_pct(geometry: dict[str, Any]) -> float:
    zone = dict(geometry.get("entry_zone", {}))
    low = float(pd.to_numeric(pd.Series([zone.get("low", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
    high = float(pd.to_numeric(pd.Series([zone.get("high", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
    mid = max(1e-9, abs((low + high) / 2.0))
    return float(abs(high - low) / mid)


def _load_stage52_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    rr_list = [_safe_parse_dict(raw) for raw in frame.get("rr_model", pd.Series(dtype=object)).tolist()]
    geo_list = [_safe_parse_dict(raw) for raw in frame.get("geometry", pd.Series(dtype=object)).tolist()]
    frame["rr_first_target"] = [float(item.get("first_target_rr", 0.0)) for item in rr_list]
    frame["expected_hold_bars_feature"] = [float(item.get("expected_hold_bars", 8.0)) for item in geo_list]
    frame["geometry_stop_distance_pct"] = [float(item.get("stop_distance_pct", 0.0)) for item in geo_list]
    frame["geometry_first_target_pct"] = [float(item.get("first_target_pct", 0.0)) for item in geo_list]
    frame["geometry_stretch_target_pct"] = [float(item.get("stretch_target_pct", 0.0)) for item in geo_list]
    frame["entry_zone_span_pct"] = [_entry_zone_span_pct(item) for item in geo_list]
    frame["family_code"] = frame.get("family", "").astype(str).map(_family_code)
    frame["timeframe_code"] = frame.get("timeframe", "").astype(str).map(_timeframe_code)
    beam_src = frame["beam_score"] if "beam_score" in frame.columns else pd.Series([0.0] * len(frame), index=frame.index)
    cost_src = frame["cost_edge_proxy"] if "cost_edge_proxy" in frame.columns else pd.Series([0.0] * len(frame), index=frame.index)
    lcb_src = frame["exp_lcb_proxy"] if "exp_lcb_proxy" in frame.columns else pd.Series([0.0] * len(frame), index=frame.index)
    frame["beam_score"] = pd.to_numeric(beam_src, errors="coerce").fillna(0.0)
    frame["cost_edge_proxy"] = pd.to_numeric(cost_src, errors="coerce").fillna(0.0)
    frame["exp_lcb_proxy"] = pd.to_numeric(lcb_src, errors="coerce").fillna(0.0)
    keep_cols = [
        "candidate_id",
        "source_candidate_id",
        "family",
        "timeframe",
        "beam_score",
        "cost_edge_proxy",
        "rr_first_target",
        "expected_hold_bars_feature",
        "family_code",
        "timeframe_code",
        "exp_lcb_proxy",
        "geometry_stop_distance_pct",
        "geometry_first_target_pct",
        "geometry_stretch_target_pct",
        "entry_zone_span_pct",
        "rr_model",
    ]
    return frame.loc[:, [col for col in keep_cols if col in frame.columns]].copy()


def _load_stage48_ranked(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    frame = frame.rename(columns={"candidate_id": "source_candidate_id"})
    for name in ("rank_score", "stage_a_score", "layer_score", "replay_worthiness"):
        source = frame[name] if name in frame.columns else pd.Series([0.0] * len(frame), index=frame.index)
        frame[name] = pd.to_numeric(source, errors="coerce").fillna(0.0)
    keep_cols = [
        "source_candidate_id",
        "rank_score",
        "stage_a_score",
        "layer_score",
        "replay_worthiness",
    ]
    return frame.loc[:, [col for col in keep_cols if col in frame.columns]].copy()


def _enrich_candidates_with_stage48(candidates: pd.DataFrame, ranked: pd.DataFrame) -> pd.DataFrame:
    frame = candidates.copy() if isinstance(candidates, pd.DataFrame) else pd.DataFrame()
    if frame.empty:
        return frame
    if "source_candidate_id" not in frame.columns:
        frame["source_candidate_id"] = ""
    if not ranked.empty:
        frame = frame.merge(ranked, on="source_candidate_id", how="left")
    for col in ("rank_score", "stage_a_score", "layer_score", "replay_worthiness"):
        source = frame[col] if col in frame.columns else pd.Series([0.0] * len(frame), index=frame.index)
        frame[col] = pd.to_numeric(source, errors="coerce").fillna(0.0)
    rr_source = frame["rr_first_target"] if "rr_first_target" in frame.columns else pd.Series([0.0] * len(frame), index=frame.index)
    frame["first_target_rr"] = pd.to_numeric(rr_source, errors="coerce").fillna(0.0)
    if "exp_lcb_proxy" not in frame.columns:
        frame["exp_lcb_proxy"] = 0.0
    frame["exp_lcb_proxy"] = pd.to_numeric(frame["exp_lcb_proxy"], errors="coerce").fillna(0.0)
    cost_edge_source = frame["cost_edge_proxy"] if "cost_edge_proxy" in frame.columns else pd.Series([0.0] * len(frame), index=frame.index)
    rank_source = frame["rank_score"] if "rank_score" in frame.columns else pd.Series([0.0] * len(frame), index=frame.index)
    fallback_lcb = (
        (pd.to_numeric(cost_edge_source, errors="coerce").fillna(0.0).clip(lower=0.0) * 0.60)
        + (pd.to_numeric(rank_source, errors="coerce").fillna(0.0).clip(lower=0.0) * 0.0015)
    )
    frame["exp_lcb_proxy"] = frame["exp_lcb_proxy"].where(frame["exp_lcb_proxy"] > 0.0, fallback_lcb)
    return frame


def _load_candidate_id_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    if frame.empty or "candidate_id" not in frame.columns:
        return set()
    return {str(v) for v in frame["candidate_id"].astype(str).tolist() if str(v).strip()}


def _primary_symbol(cfg: dict[str, Any]) -> str:
    primary = [str(v).strip() for v in cfg.get("research_scope", {}).get("primary_symbols", []) if str(v).strip()]
    if primary:
        return primary[0]
    universe = [str(v).strip() for v in cfg.get("universe", {}).get("symbols", []) if str(v).strip()]
    return universe[0] if universe else "BTC/USDT"


def _select_validated_candidate_id(
    *,
    candidates_for_pred: pd.DataFrame,
    routed: dict[str, Any],
    fallback_candidate: dict[str, Any],
) -> str:
    stage_b = pd.DataFrame(routed.get("stage_b_survivors", []))
    if not stage_b.empty and "candidate_id" in stage_b.columns:
        candidate_id = str(stage_b.iloc[0].get("candidate_id", "")).strip()
        if candidate_id:
            return candidate_id
    if isinstance(candidates_for_pred, pd.DataFrame) and not candidates_for_pred.empty and "candidate_id" in candidates_for_pred.columns:
        top = candidates_for_pred.sort_values(
            ["exp_lcb_proxy", "beam_score", "candidate_id"],
            ascending=[False, False, True],
        )
        candidate_id = str(top.iloc[0].get("candidate_id", "")).strip()
        if candidate_id:
            return candidate_id
    return str(fallback_candidate.get("candidate_id", "")).strip()


def _candidate_record_by_id(candidates: pd.DataFrame, candidate_id: str) -> dict[str, Any]:
    if candidates.empty or "candidate_id" not in candidates.columns:
        return {}
    selected = candidates.loc[candidates["candidate_id"].astype(str) == str(candidate_id).strip()].copy()
    if selected.empty:
        return {}
    return hydrate_candidate_record(dict(selected.iloc[0].to_dict()))


def _build_candidate_aligned_training_dataset(
    candidates: pd.DataFrame,
    ranked: pd.DataFrame,
    *,
    stage_a_ids: set[str],
    stage_b_ids: set[str],
) -> tuple[pd.DataFrame, float]:
    if candidates.empty:
        return pd.DataFrame(), 0.0
    work = candidates.copy()
    if "source_candidate_id" not in work.columns:
        work["source_candidate_id"] = ""
    if not ranked.empty:
        work = work.merge(ranked, on="source_candidate_id", how="left")
    else:
        work["rank_score"] = 0.0
        work["stage_a_score"] = 0.0
        work["layer_score"] = 0.0
        work["replay_worthiness"] = 0.0

    for col in (
        "beam_score",
        "cost_edge_proxy",
        "rr_first_target",
        "expected_hold_bars_feature",
        "family_code",
        "timeframe_code",
        "exp_lcb_proxy",
        "geometry_stop_distance_pct",
        "geometry_first_target_pct",
        "geometry_stretch_target_pct",
        "entry_zone_span_pct",
        "rank_score",
        "stage_a_score",
        "layer_score",
        "replay_worthiness",
    ):
        source = work[col] if col in work.columns else pd.Series([0.0] * len(work), index=work.index)
        work[col] = pd.to_numeric(source, errors="coerce").fillna(0.0)

    source_ids = work.get("source_candidate_id", pd.Series(dtype=str)).astype(str)
    stage_a_label = source_ids.isin(stage_a_ids).astype(float)
    stage_b_label = source_ids.isin(stage_b_ids).astype(float)
    rank_label = (work.get("replay_worthiness", 0.0) >= 1.0).astype(float)
    tp_before_sl_label = ((stage_a_label > 0.0) | (rank_label > 0.0)).astype(float)

    cost_edge = work["cost_edge_proxy"].astype(float)
    expected_net_after_cost_label = (
        stage_b_label * (cost_edge.clip(lower=0.0005) + 0.0010)
        + ((stage_a_label > 0.0) & (stage_b_label <= 0.0)).astype(float) * cost_edge.clip(lower=0.0001)
        + ((stage_a_label <= 0.0) & (stage_b_label <= 0.0)).astype(float) * cost_edge.clip(upper=-0.0001)
    )
    stop_dist = work["geometry_stop_distance_pct"].clip(lower=0.0005)
    first_target = work["geometry_first_target_pct"].clip(lower=0.0005)
    mae_pct_label = (
        -1.0
        * (
            stage_a_label * stop_dist * 0.85
            + (1.0 - stage_a_label) * stop_dist * 1.20
        )
    )
    mfe_pct_label = (
        stage_b_label * first_target * 1.15
        + ((stage_a_label > 0.0) & (stage_b_label <= 0.0)).astype(float) * first_target * 0.85
        + ((stage_a_label <= 0.0) & (stage_b_label <= 0.0)).astype(float) * cost_edge.clip(lower=0.0)
    )
    expected_hold_bars_label = work["expected_hold_bars_feature"] * (1.0 + ((stage_a_label <= 0.0).astype(float) * 0.35))
    label_present = ((source_ids.isin(stage_a_ids | stage_b_ids)) | (work.get("rank_score", 0.0) > 0.0))

    timestamps = pd.date_range("2024-01-01", periods=len(work), freq="h", tz="UTC")
    dataset = pd.DataFrame(
        {
            "timestamp": timestamps,
            "candidate_id": work.get("candidate_id", pd.Series(dtype=str)).astype(str),
            "source_candidate_id": source_ids,
            "beam_score": work["beam_score"].astype(float),
            "cost_edge_proxy": cost_edge.astype(float),
            "rr_first_target": work["rr_first_target"].astype(float),
            "family_code": work["family_code"].astype(float),
            "timeframe_code": work["timeframe_code"].astype(float),
            "expected_hold_bars_feature": work["expected_hold_bars_feature"].astype(float),
            "exp_lcb_proxy": work["exp_lcb_proxy"].astype(float),
            "geometry_stop_distance_pct": work["geometry_stop_distance_pct"].astype(float),
            "geometry_first_target_pct": work["geometry_first_target_pct"].astype(float),
            "geometry_stretch_target_pct": work["geometry_stretch_target_pct"].astype(float),
            "entry_zone_span_pct": work["entry_zone_span_pct"].astype(float),
            "stage48_rank_score": work["rank_score"].astype(float),
            "stage48_stage_a_score": work["stage_a_score"].astype(float),
            "stage48_layer_score": work["layer_score"].astype(float),
            "stage48_replay_worthiness": work["replay_worthiness"].astype(float),
            "tp_before_sl_label": tp_before_sl_label.astype(float),
            "expected_net_after_cost_label": expected_net_after_cost_label.astype(float),
            "mae_pct_label": mae_pct_label.astype(float),
            "mfe_pct_label": mfe_pct_label.astype(float),
            "expected_hold_bars_label": expected_hold_bars_label.astype(float),
            "stage_a_label": stage_a_label.astype(float),
            "stage_b_label": stage_b_label.astype(float),
            "label_present": label_present.astype(int),
        }
    )
    coverage = float(dataset["label_present"].mean()) if not dataset.empty else 0.0
    return dataset, coverage


def _bootstrap_dataset(rows: int = 240) -> tuple[pd.DataFrame, float]:
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    out_rows: list[dict[str, float | str]] = []
    for idx, timestamp in enumerate(timestamps):
        rr = 1.25 + ((idx % 5) * 0.10)
        cost_edge = -0.0003 + ((idx % 4) * 0.0005)
        out_rows.append(
            {
                "timestamp": timestamp,
                "candidate_id": f"s53_boot_{idx}",
                "source_candidate_id": f"s47_boot_{idx % 8}",
                "beam_score": float(0.35 + ((idx % 6) * 0.08)),
                "cost_edge_proxy": float(cost_edge),
                "rr_first_target": float(rr),
                "family_code": float(idx % 3),
                "timeframe_code": float(idx % 5),
                "expected_hold_bars_feature": float(6 + (idx % 5) * 2),
                "exp_lcb_proxy": float(-0.002 + (idx % 7) * 0.0007),
                "geometry_stop_distance_pct": float(0.004 + (idx % 4) * 0.001),
                "geometry_first_target_pct": float(0.008 + (idx % 4) * 0.002),
                "geometry_stretch_target_pct": float(0.012 + (idx % 4) * 0.003),
                "entry_zone_span_pct": float(0.001 + (idx % 3) * 0.0005),
                "stage48_rank_score": float(0.40 + (idx % 5) * 0.10),
                "stage48_stage_a_score": float(0.45 + (idx % 5) * 0.08),
                "stage48_layer_score": float(0.50 + (idx % 4) * 0.10),
                "stage48_replay_worthiness": float(1.0 if idx % 3 else 0.0),
                "tp_before_sl_label": float(1.0 if (rr >= 1.5 and cost_edge > 0.0) else 0.0),
                "expected_net_after_cost_label": float(cost_edge + (rr - 1.5) * 0.0015),
                "mae_pct_label": float(-0.004 + (idx % 3) * 0.0005),
                "mfe_pct_label": float(0.006 + (idx % 5) * 0.0008),
                "expected_hold_bars_label": float(6 + (idx % 5) * 2),
                "stage_a_label": float(1.0 if idx % 3 else 0.0),
                "stage_b_label": float(1.0 if idx % 7 == 0 else 0.0),
                "label_present": 1.0,
            }
        )
    return pd.DataFrame(out_rows), 1.0


def _select_feature_columns(dataset: pd.DataFrame, *, configured_features: list[str]) -> list[str]:
    alias = {
        "rr_value": "rr_first_target",
        "hold_bars": "expected_hold_bars_feature",
        "cost_edge_proxy": "cost_edge_proxy",
    }
    selected: list[str] = []
    for feature in [str(v) for v in configured_features]:
        mapped = str(alias.get(feature, feature))
        if mapped in dataset.columns and mapped not in selected:
            selected.append(mapped)

    feature_pool = [
        "beam_score",
        "cost_edge_proxy",
        "rr_first_target",
        "family_code",
        "timeframe_code",
        "expected_hold_bars_feature",
        "exp_lcb_proxy",
        "geometry_stop_distance_pct",
        "geometry_first_target_pct",
        "geometry_stretch_target_pct",
        "entry_zone_span_pct",
        "stage48_rank_score",
        "stage48_stage_a_score",
        "stage48_layer_score",
        "stage48_replay_worthiness",
    ]
    for name in feature_pool:
        if name in dataset.columns and name not in selected:
            selected.append(name)
    return selected


def _quality_gate(dataset: pd.DataFrame, *, feature_columns: list[str], label_coverage: float) -> dict[str, Any]:
    if dataset.empty:
        return {
            "passed": False,
            "effective_feature_count": 0,
            "non_constant_feature_count": 0,
            "label_coverage": float(label_coverage),
            "reason": "empty_dataset",
        }
    non_constant = int(
        sum(
            1
            for col in [str(v) for v in feature_columns]
            if col in dataset.columns and int(dataset[col].nunique(dropna=True)) > 1
        )
    )
    has_label_variance = int(dataset["tp_before_sl_label"].nunique(dropna=True)) > 1
    passed = (
        len(feature_columns) >= 3
        and non_constant >= 2
        and float(label_coverage) >= 0.20
        and bool(has_label_variance)
    )
    reasons: list[str] = []
    if len(feature_columns) < 3:
        reasons.append("insufficient_effective_features")
    if non_constant < 2:
        reasons.append("insufficient_non_constant_features")
    if float(label_coverage) < 0.20:
        reasons.append("insufficient_label_coverage")
    if not has_label_variance:
        reasons.append("label_variance_missing")
    return {
        "passed": bool(passed),
        "effective_feature_count": int(len(feature_columns)),
        "non_constant_feature_count": int(non_constant),
        "label_coverage": float(round(label_coverage, 6)),
        "reason": ",".join(reasons),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    stage28_run_id = _resolve_stage28_run_id(args, docs_dir)

    candidates = pd.DataFrame()
    ranked = pd.DataFrame()
    stage_a_ids: set[str] = set()
    stage_b_ids: set[str] = set()
    if stage28_run_id:
        base = Path(args.runs_dir) / stage28_run_id
        candidates = _load_stage52_candidates(base / "stage52" / "setup_candidates_v2.csv")
        ranked = _load_stage48_ranked(base / "stage48" / "stage48_ranked_candidates.csv")
        stage_a_ids = _load_candidate_id_set(base / "stage48" / "stage48_stage_a_survivors.csv")
        stage_b_ids = _load_candidate_id_set(base / "stage48" / "stage48_stage_b_survivors.csv")
    candidates = _enrich_candidates_with_stage48(candidates, ranked)

    input_mode = "stage52_stage48_artifacts"
    dataset, label_coverage = _build_candidate_aligned_training_dataset(
        candidates,
        ranked,
        stage_a_ids=stage_a_ids,
        stage_b_ids=stage_b_ids,
    )
    if dataset.empty:
        input_mode = "bootstrap_training_dataset"
        dataset, label_coverage = _bootstrap_dataset()
        if candidates.empty:
            candidates = dataset.loc[
                :,
                [
                    "candidate_id",
                    "beam_score",
                    "cost_edge_proxy",
                    "rr_first_target",
                    "family_code",
                    "timeframe_code",
                    "expected_hold_bars_feature",
                    "exp_lcb_proxy",
                    "geometry_stop_distance_pct",
                    "geometry_first_target_pct",
                    "geometry_stretch_target_pct",
                    "entry_zone_span_pct",
                ],
            ].copy()
            candidates["rr_model"] = [
                {"first_target_rr": float(v)}
                for v in pd.to_numeric(candidates["rr_first_target"], errors="coerce").fillna(0.0).tolist()
            ]
            candidates["expected_hold_bars"] = candidates["expected_hold_bars_feature"]

    configured_features = [str(col) for col in cfg.get("tradability_model", {}).get("feature_columns", [])]
    feature_columns = _select_feature_columns(dataset, configured_features=configured_features)
    quality = _quality_gate(dataset, feature_columns=feature_columns, label_coverage=label_coverage)

    model_bundle: dict[str, Any] | None = None
    predictions = pd.DataFrame()
    routed: dict[str, Any] = {
        "stage_a_survivors": pd.DataFrame(),
        "stage_b_survivors": pd.DataFrame(),
        "counts": {"input": int(len(candidates)), "stage_a": 0, "stage_b": 0},
    }
    fit_error = ""
    validated_candidate: dict[str, Any] = {}
    validated_symbol = _primary_symbol(cfg)
    replay_metrics_artifact_path = ""
    replay_execution_status = "NOT_ATTEMPTED"
    replay_validation_state = "NOT_ATTEMPTED"
    replay_decision_use_allowed = False
    replay_evidence_quality = "not_attempted"

    try:
        model_bundle = fit_tradability_model_v2(
            dataset,
            feature_columns=feature_columns,
            seed=int(cfg.get("search", {}).get("seed", 42)),
            probability_bins=int(cfg.get("tradability_model", {}).get("probability_bins", 6)),
        )

        candidates_for_pred = candidates.copy()
        for column in feature_columns:
            if column not in candidates_for_pred.columns:
                candidates_for_pred[column] = 0.0
        if "candidate_id" not in candidates_for_pred.columns:
            candidates_for_pred["candidate_id"] = [f"s53_auto_{idx}" for idx in range(len(candidates_for_pred))]
        if "expected_hold_bars" not in candidates_for_pred.columns:
            candidates_for_pred["expected_hold_bars"] = pd.to_numeric(
                candidates_for_pred.get("expected_hold_bars_feature", 8.0),
                errors="coerce",
            ).fillna(8.0)
        if "rr_model" not in candidates_for_pred.columns:
            candidates_for_pred["rr_model"] = [
                {"first_target_rr": float(v)}
                for v in pd.to_numeric(candidates_for_pred.get("rr_first_target", 0.0), errors="coerce").fillna(0.0).tolist()
            ]
        if "exp_lcb_proxy" not in candidates_for_pred.columns:
            candidates_for_pred["exp_lcb_proxy"] = 0.0

        predictions = predict_tradability_model_v2(model_bundle, candidates_for_pred)
        routed = route_tradability_v2(
            candidates_for_pred,
            predictions=predictions,
            stage_a_prob_threshold=float(cfg.get("promotion_gates", {}).get("stage_a", {}).get("min_tp_before_sl_prob", 0.55)),
            min_rr=float(cfg.get("promotion_gates", {}).get("stage_a", {}).get("min_rr", 1.5)),
            min_cost_edge=float(cfg.get("promotion_gates", {}).get("stage_a", {}).get("min_cost_edge_proxy", 0.0)),
            hold_bar_ceiling=float(cfg.get("promotion_gates", {}).get("stage_a", {}).get("max_expected_hold_bars", 24.0)),
        )

    except Exception as exc:
        fit_error = str(exc)
        model_bundle = None

    if stage28_run_id:
        fallback_candidate = resolve_validation_candidate(
            runs_dir=Path(args.runs_dir),
            stage28_run_id=stage28_run_id,
            docs_dir=docs_dir,
        )
        if not validated_candidate:
            validated_candidate_id = _select_validated_candidate_id(
                candidates_for_pred=candidates if isinstance(candidates, pd.DataFrame) else pd.DataFrame(),
                routed=routed,
                fallback_candidate=fallback_candidate,
            )
            validated_candidate = _candidate_record_by_id(candidates, validated_candidate_id) or fallback_candidate

    blocker_reason = str(fit_error or quality["reason"])
    summary = {
        "stage": "53",
        "status": "PARTIAL",
        "execution_status": "BLOCKED",
        "stage_role": "heuristic_filter",
        "validation_state": "HEURISTIC_FILTER_BLOCKED",
        "input_mode": input_mode,
        "stage28_run_id": stage28_run_id,
        "feature_columns": feature_columns,
        "effective_feature_count": int(quality["effective_feature_count"]),
        "non_constant_feature_count": int(quality["non_constant_feature_count"]),
        "label_coverage": float(quality["label_coverage"]),
        "quality_gate_passed": bool(quality["passed"]),
        "quality_gate_reason": str(quality["reason"]),
        "model_names": sorted(model_bundle["base_models"].keys()) if model_bundle else [],
        "candidate_count": int(len(candidates)),
        "stage_a_survivors": int(routed["counts"]["stage_a"]),
        "stage_b_survivors": int(routed["counts"]["stage_b"]),
        "validated_candidate_id": "",
        "validated_symbol": validated_symbol,
        "validated_timeframe": "",
        "metric_source_type": "heuristic_filter",
        "replay_execution_status": replay_execution_status,
        "replay_validation_state": replay_validation_state,
        "decision_use_allowed": replay_decision_use_allowed,
        "evidence_quality": replay_evidence_quality,
        "replay_metrics_artifact_path": replay_metrics_artifact_path,
        "blocker_reason": blocker_reason,
    }

    if stage28_run_id:
        out_dir = Path(args.runs_dir) / stage28_run_id / "stage53"
        out_dir.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(out_dir / "training_dataset.csv", index=False)
        predictions.to_csv(out_dir / "predictions.csv", index=False)
        pd.DataFrame(routed["stage_a_survivors"]).to_csv(out_dir / "stage_a_survivors.csv", index=False)
        pd.DataFrame(routed["stage_b_survivors"]).to_csv(out_dir / "stage_b_survivors.csv", index=False)
        labels_path = Path(args.runs_dir) / stage28_run_id / "stage48" / "stage48_labels.csv"
        labels = pd.read_csv(labels_path) if labels_path.exists() else pd.DataFrame()
        stage_b_df = pd.DataFrame(routed["stage_b_survivors"])
        trade_count = int(len(stage_b_df))
        exp_src = stage_b_df["expected_net_after_cost"] if "expected_net_after_cost" in stage_b_df.columns else pd.Series([0.0] * len(stage_b_df), index=stage_b_df.index)
        exp_values = pd.to_numeric(exp_src, errors="coerce").fillna(0.0)
        exp_lcb = float(exp_values.quantile(0.10)) if len(exp_values) else 0.0
        if not labels.empty:
            net_src = labels["net_return_after_cost"] if "net_return_after_cost" in labels.columns else pd.Series([0.0] * len(labels), index=labels.index)
            net = pd.to_numeric(net_src, errors="coerce").fillna(0.0)
            running = net.cumsum()
            peak = running.cummax()
            dd = ((running - peak) / peak.replace(0.0, pd.NA)).fillna(0.0)
            maxdd = float(abs(dd.min()))
            tradable_src = labels["tradable"] if "tradable" in labels.columns else pd.Series([0] * len(labels), index=labels.index)
            failure_dom = float((pd.to_numeric(tradable_src, errors="coerce").fillna(0).astype(int) == 0).mean())
        else:
            maxdd = 1.0
            failure_dom = 1.0
        replay_metrics_path = out_dir / "replay_metrics_real.json"
        if validated_candidate:
            replay = run_candidate_replay(
                candidate=validated_candidate,
                config=cfg,
                symbol=validated_symbol,
            )
            replay_execution_status = str(replay.get("execution_status", "BLOCKED"))
            replay_validation_state = str(replay.get("validation_state", "REAL_REPLAY_BLOCKED"))
            replay_decision_use_allowed = bool(replay.get("decision_use_allowed", False))
            replay_evidence_quality = "artifact_backed_real" if replay_execution_status == "EXECUTED" else "real_but_blocked"
            trades = replay.get("trades", pd.DataFrame())
            equity_curve = replay.get("equity_curve", pd.DataFrame())
            market_meta = dict(replay.get("market_meta", {}))
            replay_payload = {
                "metric_source_type": "real_replay",
                "artifact_path": str(replay_metrics_path),
                "candidate_id": str(validated_candidate.get("candidate_id", "")),
                "symbol": validated_symbol,
                "timeframe": str(validated_candidate.get("timeframe", "")),
                "execution_status": replay_execution_status,
                "validation_state": replay_validation_state,
                "decision_use_allowed": replay_decision_use_allowed,
                "evidence_quality": replay_evidence_quality,
                "trade_count": int(replay.get("metrics", {}).get("trade_count", trade_count)),
                "exp_lcb": float(replay.get("metrics", {}).get("exp_lcb", exp_lcb)),
                "maxDD": float(replay.get("metrics", {}).get("maxDD", maxdd)),
                "failure_reason_dominance": float(replay.get("metrics", {}).get("failure_reason_dominance", failure_dom)),
                "expectancy": float(replay.get("metrics", {}).get("expectancy", 0.0)),
                "profit_factor": float(replay.get("metrics", {}).get("profit_factor", 0.0)),
                "market_meta": market_meta,
                "backtest_params": dict(replay.get("backtest_params", {})),
            }
            replay_metrics_path.write_text(json.dumps(replay_payload, indent=2, allow_nan=False), encoding="utf-8")
            if isinstance(trades, pd.DataFrame) and not trades.empty:
                trades.to_csv(out_dir / "replay_trades_real.csv", index=False)
            if isinstance(equity_curve, pd.DataFrame) and not equity_curve.empty:
                equity_curve.to_csv(out_dir / "replay_equity_curve_real.csv", index=False)
            (out_dir / "replay_market_meta.json").write_text(json.dumps(market_meta, indent=2, allow_nan=False), encoding="utf-8")
            replay_metrics_artifact_path = str(replay_metrics_path)

        summary.update(
            {
                "execution_status": "EXECUTED" if replay_metrics_artifact_path or (bool(quality["passed"]) and not fit_error) else "BLOCKED",
                "validation_state": replay_validation_state if replay_metrics_artifact_path else ("HEURISTIC_FILTER_READY" if bool(quality["passed"]) and not fit_error else "HEURISTIC_FILTER_BLOCKED"),
                "validated_candidate_id": str(validated_candidate.get("candidate_id", "")),
                "validated_timeframe": str(validated_candidate.get("timeframe", "")),
                "metric_source_type": "real_replay" if replay_metrics_artifact_path else "heuristic_filter",
                "replay_execution_status": replay_execution_status,
                "replay_validation_state": replay_validation_state,
                "decision_use_allowed": replay_decision_use_allowed,
                "evidence_quality": replay_evidence_quality,
                "replay_metrics_artifact_path": replay_metrics_artifact_path,
            }
        )
        blocker_parts = [part for part in [fit_error, str(quality["reason"]).strip()] if part]
        if replay_metrics_artifact_path and not replay_decision_use_allowed:
            blocker_parts.append(replay_validation_state.lower())
        summary["blocker_reason"] = ",".join(blocker_parts)
        summary["status"] = (
            "SUCCESS"
            if bool(quality["passed"]) and not fit_error and replay_decision_use_allowed
            else "PARTIAL"
        )
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    summary_path = docs_dir / "stage53_summary.json"
    report_path = docs_dir / "stage53_report.md"
    summary["summary_hash"] = stable_hash({k: v for k, v in summary.items() if k != "summary_hash"}, length=16)
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(
        "\n".join(
            [
                "# Stage-53 Report",
                "",
                f"- status: `{summary['status']}`",
                f"- execution_status: `{summary['execution_status']}`",
                f"- stage_role: `{summary['stage_role']}`",
                f"- validation_state: `{summary['validation_state']}`",
                f"- input_mode: `{summary['input_mode']}`",
                f"- stage28_run_id: `{summary['stage28_run_id']}`",
                f"- model_names: `{summary['model_names']}`",
                f"- feature_columns: `{summary['feature_columns']}`",
                f"- effective_feature_count: `{summary['effective_feature_count']}`",
                f"- non_constant_feature_count: `{summary['non_constant_feature_count']}`",
                f"- label_coverage: `{summary['label_coverage']}`",
                f"- quality_gate_passed: `{summary['quality_gate_passed']}`",
                f"- quality_gate_reason: `{summary['quality_gate_reason']}`",
                f"- candidate_count: `{summary['candidate_count']}`",
                f"- stage_a_survivors: `{summary['stage_a_survivors']}`",
                f"- stage_b_survivors: `{summary['stage_b_survivors']}`",
                f"- validated_candidate_id: `{summary['validated_candidate_id']}`",
                f"- validated_symbol: `{summary['validated_symbol']}`",
                f"- validated_timeframe: `{summary['validated_timeframe']}`",
                f"- metric_source_type: `{summary['metric_source_type']}`",
                f"- replay_execution_status: `{summary['replay_execution_status']}`",
                f"- replay_validation_state: `{summary['replay_validation_state']}`",
                f"- decision_use_allowed: `{summary['decision_use_allowed']}`",
                f"- evidence_quality: `{summary['evidence_quality']}`",
                f"- replay_metrics_artifact_path: `{summary['replay_metrics_artifact_path']}`",
                f"- blocker_reason: `{summary['blocker_reason']}`",
                f"- summary_hash: `{summary['summary_hash']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
