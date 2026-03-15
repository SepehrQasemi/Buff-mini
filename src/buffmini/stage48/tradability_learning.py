"""Stage-48 tradability learning and deterministic pre-replay ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from buffmini.research.behavior import build_behavioral_fingerprints
from buffmini.research.diagnostics import classify_candidate_hierarchy, classify_candidate_tier, compute_candidate_risk_card


@dataclass(frozen=True)
class Stage48Config:
    horizon_bars: int = 12
    tp_pct: float = 0.004
    sl_pct: float = 0.003
    round_trip_cost_pct: float = 0.001
    max_adverse_excursion_pct: float = 0.004
    min_rr: float = 1.2
    stage_a_threshold: float = 0.42
    stage_b_threshold: float = 0.0


RANKING_PROFILES: dict[str, dict[str, float]] = {
    "stage94_baseline": {
        "layer_score": 0.22,
        "exp_lcb_proxy": 0.18,
        "cost_edge_proxy": 0.12,
        "rr_first_target": 0.08,
        "no_reject_penalty": 0.05,
        "cost_fragility": 0.10,
        "trade_density": 0.07,
        "regime_concentration": 0.05,
        "overlap_duplication": 0.08,
        "clustering": 0.06,
        "thin_evidence": 0.05,
        "transfer_prior": 0.04,
        "tradable_rate": 0.04,
        "net_mean": 0.03,
        "rr_ok": 0.01,
        "usefulness_prior": 0.0,
        "quality_bonus": 0.0,
    },
    "stage95_usefulness_push": {
        "layer_score": 0.18,
        "exp_lcb_proxy": 0.16,
        "cost_edge_proxy": 0.10,
        "rr_first_target": 0.06,
        "no_reject_penalty": 0.03,
        "cost_fragility": 0.12,
        "trade_density": 0.11,
        "regime_concentration": 0.06,
        "overlap_duplication": 0.11,
        "clustering": 0.09,
        "thin_evidence": 0.08,
        "transfer_prior": 0.07,
        "tradable_rate": 0.02,
        "net_mean": 0.02,
        "rr_ok": 0.01,
        "usefulness_prior": 0.08,
        "quality_bonus": 0.0,
    },
    "stage99_quality_acceleration": {
        "layer_score": 0.16,
        "exp_lcb_proxy": 0.18,
        "cost_edge_proxy": 0.12,
        "rr_first_target": 0.06,
        "no_reject_penalty": 0.03,
        "cost_fragility": 0.12,
        "trade_density": 0.10,
        "regime_concentration": 0.05,
        "overlap_duplication": 0.10,
        "clustering": 0.09,
        "thin_evidence": 0.08,
        "transfer_prior": 0.06,
        "tradable_rate": 0.02,
        "net_mean": 0.02,
        "rr_ok": 0.01,
        "usefulness_prior": 0.06,
        "quality_bonus": 0.12,
    },
}


def available_ranking_profiles() -> tuple[str, ...]:
    return tuple(RANKING_PROFILES)


def compute_stage48_labels(frame: pd.DataFrame, *, cfg: Stage48Config | None = None) -> pd.DataFrame:
    """Compute leakage-safe Stage-48 tradability labels."""

    conf = cfg or Stage48Config()
    data = frame.copy()
    for key in ("close", "high", "low"):
        data[key] = pd.to_numeric(data.get(key), errors="coerce").astype(float)
    data["timestamp"] = pd.to_datetime(data.get("timestamp"), utc=True, errors="coerce")

    n = len(data)
    out_rows: list[dict[str, Any]] = []
    for i in range(n):
        close_i = float(data["close"].iloc[i]) if np.isfinite(data["close"].iloc[i]) else np.nan
        end = min(n, i + int(conf.horizon_bars) + 1)
        window = data.iloc[i + 1 : end].copy() if end > i + 1 else pd.DataFrame()
        if not np.isfinite(close_i) or close_i <= 0.0 or window.empty:
            out_rows.append(
                {
                    "timestamp": data["timestamp"].iloc[i],
                    "tp_before_sl": np.nan,
                    "net_return_after_cost": np.nan,
                    "tradability_class": "UNKNOWN",
                    "acceptable_excursion": False,
                    "expected_hold_validity": False,
                    "rr_adequacy": False,
                    "tradable": 0,
                }
            )
            continue

        tp_price = close_i * (1.0 + float(conf.tp_pct))
        sl_price = close_i * (1.0 - float(conf.sl_pct))
        hit_tp = _first_hit(window["high"], threshold=tp_price, direction="up")
        hit_sl = _first_hit(window["low"], threshold=sl_price, direction="down")
        tp_before_sl = float((np.isfinite(hit_tp) and (not np.isfinite(hit_sl) or hit_tp < hit_sl)))
        fwd_close = float(window["close"].iloc[-1]) if not window.empty else close_i
        gross = float((fwd_close - close_i) / close_i)
        net = float(gross - float(conf.round_trip_cost_pct))
        mae = float((pd.to_numeric(window["low"], errors="coerce").min() - close_i) / close_i)
        acceptable_excursion = bool(mae >= -float(conf.max_adverse_excursion_pct))
        rr_score = float(conf.tp_pct / max(conf.sl_pct, 1e-9))
        rr_adequacy = bool(rr_score >= float(conf.min_rr))
        expected_hold_validity = bool(len(window) >= int(max(1, conf.horizon_bars // 2)))
        tradability_class = "GOOD" if (net > 0.0 and acceptable_excursion and rr_adequacy) else ("MARGINAL" if net > -0.001 else "POOR")
        tradable = int((tp_before_sl >= 1.0 or net > 0.0) and acceptable_excursion and rr_adequacy)
        out_rows.append(
            {
                "timestamp": data["timestamp"].iloc[i],
                "tp_before_sl": tp_before_sl,
                "net_return_after_cost": net,
                "tradability_class": tradability_class,
                "acceptable_excursion": acceptable_excursion,
                "expected_hold_validity": expected_hold_validity,
                "rr_adequacy": rr_adequacy,
                "tradable": tradable,
            }
        )
    return pd.DataFrame(out_rows)


def score_candidates_with_ranker(
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
    market_frame: pd.DataFrame | None = None,
    *,
    profile: str = "stage99_quality_acceleration",
) -> pd.DataFrame:
    """Deterministic pre-replay ranker using tradability-aware scores."""

    work = candidates.copy() if isinstance(candidates, pd.DataFrame) else pd.DataFrame()
    if work.empty:
        return pd.DataFrame(columns=["candidate_id", "rank_score", "predicted_tradability", "replay_worthiness"])
    work = _augment_candidate_specific_fields(work, market_frame=market_frame)
    weights = dict(RANKING_PROFILES.get(str(profile), RANKING_PROFILES["stage99_quality_acceleration"]))
    tradable_rate = float(pd.to_numeric(labels.get("tradable", 0), errors="coerce").fillna(0).astype(int).mean()) if not labels.empty else 0.0
    net_mean = float(pd.to_numeric(labels.get("net_return_after_cost", 0.0), errors="coerce").fillna(0.0).mean()) if not labels.empty else 0.0
    rr_ok = float(pd.to_numeric(labels.get("rr_adequacy", 0), errors="coerce").fillna(0).astype(int).mean()) if not labels.empty else 0.0

    rank_score = (
        (work["layer_score"] * weights["layer_score"])
        + (work["exp_lcb_proxy"].clip(lower=-0.01) * 40.0 * weights["exp_lcb_proxy"])
        + (work["cost_edge_proxy"].clip(lower=-0.01) * 100.0 * weights["cost_edge_proxy"])
        + (work["rr_first_target"].clip(lower=0.0, upper=3.0) / 3.0 * weights["rr_first_target"])
        + (work["no_reject_penalty"] * weights["no_reject_penalty"])
        + ((1.0 - work["cost_fragility_risk"]) * weights["cost_fragility"])
        + ((1.0 - work["trade_density_risk"]) * weights["trade_density"])
        + ((1.0 - work["regime_concentration_risk"]) * weights["regime_concentration"])
        + ((1.0 - work["overlap_duplication_risk"]) * weights["overlap_duplication"])
        + ((1.0 - work["clustering_risk"]) * weights["clustering"])
        + ((1.0 - work["thin_evidence_risk"]) * weights["thin_evidence"])
        + ((1.0 - work["transfer_risk_prior"]) * weights["transfer_prior"])
        + (work["usefulness_prior"] * weights["usefulness_prior"])
        + (work["trade_quality_bonus"] * weights["quality_bonus"])
        + (tradable_rate * weights["tradable_rate"])
        + (max(0.0, net_mean) * 25.0 * weights["net_mean"])
        + (rr_ok * weights["rr_ok"])
    )
    work["rank_score"] = rank_score.astype(float).clip(lower=-2.0, upper=2.0)
    work["predicted_tradability"] = (
        (tradable_rate * 0.45)
        + ((1.0 - work["cost_fragility_risk"]) * 0.15)
        + ((1.0 - work["thin_evidence_risk"]) * 0.10)
        + ((1.0 - work["trade_density_risk"]) * 0.10)
        + (work["rr_first_target"].clip(lower=0.0, upper=3.0) / 3.0 * 0.10)
        + ((1.0 - work["regime_concentration_risk"]) * 0.10)
    ).clip(lower=0.0, upper=1.0)
    median_score = float(work["rank_score"].median()) if not work.empty else 0.0
    worth_mask = (
        (work["rank_score"] >= median_score)
        & (work["exp_lcb_proxy"] > -0.0001)
        & (work["cost_edge_proxy"] > -0.0001)
        & (work["aggregate_risk"] <= 0.72)
    )
    work["replay_worthiness"] = worth_mask.astype(int)
    work["candidate_class"] = [
        classify_candidate_tier(
            rank_score=float(row.get("rank_score", 0.0)),
            replay_exp_lcb=float(row.get("exp_lcb_proxy", 0.0)),
            walkforward_usable_windows=int(float(row.get("walkforward_usable_windows", 0) or 0)),
            decision_use_allowed=bool(row.get("decision_use_allowed", False)),
            aggregate_risk=float(row.get("aggregate_risk", 1.0)),
        )
        for row in work.to_dict(orient="records")
    ]
    work["candidate_hierarchy"] = [
        classify_candidate_hierarchy(
            rank_score=float(row.get("rank_score", 0.0)),
            replay_exp_lcb=float(row.get("exp_lcb_proxy", 0.0)),
            walkforward_usable_windows=int(float(row.get("walkforward_usable_windows", 0) or 0)),
            decision_use_allowed=bool(row.get("decision_use_allowed", False)),
            aggregate_risk=float(row.get("aggregate_risk", 1.0)),
            robustness_level=int(float(row.get("robustness_level", 0) or 0)),
        )
        for row in work.to_dict(orient="records")
    ]
    cols = [
        c
        for c in (
            "candidate_id",
            "source_candidate_id",
            "economic_fingerprint",
            "behavioral_fingerprint",
            "rank_score",
            "predicted_tradability",
            "replay_worthiness",
            "candidate_class",
            "candidate_hierarchy",
            "trade_density_risk",
            "cost_fragility_risk",
            "regime_concentration_risk",
            "hold_sanity_risk",
            "overlap_duplication_risk",
            "clustering_risk",
            "thin_evidence_risk",
            "transfer_risk_prior",
            "usefulness_prior",
            "trade_quality_bonus",
            "aggregate_risk",
            "activation_density",
            "entry_overlap_score",
            "exit_overlap_score",
            "pnl_correlation_risk",
            "failure_pattern_similarity",
            "side_distribution",
            "hold_distribution",
            "regime_activation_map",
        )
        if c in work.columns
    ]
    return work.sort_values(["rank_score", "candidate_id"], ascending=[False, True]).loc[:, cols].reset_index(drop=True)


def route_stage_a_stage_b(
    candidates: pd.DataFrame,
    *,
    labels: pd.DataFrame,
    market_frame: pd.DataFrame | None = None,
    cfg: Stage48Config | None = None,
) -> dict[str, Any]:
    """Route candidates through Stage-A tradability and Stage-B robustness accounting."""

    conf = cfg or Stage48Config()
    work = candidates.copy() if isinstance(candidates, pd.DataFrame) else pd.DataFrame()
    if work.empty:
        return {
            "strict_direct_survivors_before": 0,
            "stage_a_survivors": pd.DataFrame(),
            "stage_b_survivors": pd.DataFrame(),
            "counts": {"input": 0, "stage_a": 0, "stage_b": 0},
            "strongest_bottleneck": "stage_a_tradability",
        }
    work = _augment_candidate_specific_fields(work, market_frame=market_frame)

    tradable_rate = float(pd.to_numeric(labels.get("tradable", 0), errors="coerce").fillna(0).astype(int).mean()) if not labels.empty else 0.0
    net_mean = float(pd.to_numeric(labels.get("net_return_after_cost", 0.0), errors="coerce").fillna(0.0).mean()) if not labels.empty else 0.0
    rr_ok = float(pd.to_numeric(labels.get("rr_adequacy", 0), errors="coerce").fillna(0).astype(int).mean()) if not labels.empty else 0.0
    expected_hold = float(pd.to_numeric(labels.get("expected_hold_validity", 0), errors="coerce").fillna(0).astype(int).mean()) if not labels.empty else 0.0

    work["stage_a_score"] = (
        (work["layer_score"] * 0.25)
        + (work["exp_lcb_proxy"].clip(lower=0.0) * 6.0 * 0.25)
        + (work["cost_edge_proxy"].clip(lower=0.0) * 100.0 * 0.20)
        + (work["rr_first_target"].clip(lower=0.0, upper=3.0) / 3.0 * 0.15)
        + (work["no_reject_penalty"] * 0.05)
        + ((1.0 - work["aggregate_risk"]) * 0.10)
        + (tradable_rate * 0.05)
        + (max(0.0, net_mean) * 25.0 * 0.03)
        + (rr_ok * 0.015)
        + (expected_hold * 0.005)
    )
    stage_a = work.loc[
        (work["stage_a_score"] >= float(conf.stage_a_threshold))
        & (work["cost_edge_proxy"] > -0.0001)
        & (work["rr_first_target"] >= float(conf.min_rr) * 0.85),
        :,
    ].copy()
    if stage_a.empty and not work.empty:
        stage_a = work.nlargest(min(4, len(work)), "stage_a_score").copy()
    stage_b = stage_a.loc[
        (stage_a["exp_lcb_proxy"] >= float(conf.stage_b_threshold))
        & (stage_a["cost_edge_proxy"] > 0.0),
        :,
    ].copy()
    if "economic_fingerprint" in stage_b.columns:
        stage_b = (
            stage_b.sort_values(["stage_a_score", "candidate_id"], ascending=[False, True])
            .drop_duplicates(subset=["economic_fingerprint"], keep="first")
            .reset_index(drop=True)
        )
    strict_before = int((work["exp_lcb_proxy"] >= float(conf.stage_b_threshold)).sum())

    drop_a = int(len(work) - len(stage_a))
    drop_b = int(len(stage_a) - len(stage_b))
    strongest_bottleneck = "stage_a_tradability" if drop_a >= drop_b else "stage_b_robustness"
    return {
        "strict_direct_survivors_before": strict_before,
        "stage_a_survivors": stage_a.reset_index(drop=True),
        "stage_b_survivors": stage_b.reset_index(drop=True),
        "counts": {"input": int(len(work)), "stage_a": int(len(stage_a)), "stage_b": int(len(stage_b))},
        "strongest_bottleneck": strongest_bottleneck,
    }


def _first_hit(series: pd.Series, *, threshold: float, direction: str) -> float:
    values = pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
    for idx, value in enumerate(values):
        if not np.isfinite(value):
            continue
        if direction == "up" and value >= threshold:
            return float(idx)
        if direction == "down" and value <= threshold:
            return float(idx)
    return float(np.nan)


def _augment_candidate_specific_fields(frame: pd.DataFrame, *, market_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    work = frame.copy()
    beam_or_layer = work["beam_score"] if "beam_score" in work.columns else _series_or_default(work, "layer_score", 0.0)
    work["layer_score"] = pd.to_numeric(beam_or_layer, errors="coerce").fillna(0.0)
    work["exp_lcb_proxy"] = pd.to_numeric(_series_or_default(work, "exp_lcb_proxy", 0.0), errors="coerce").fillna(0.0)
    work["cost_edge_proxy"] = pd.to_numeric(_series_or_default(work, "cost_edge_proxy", 0.0), errors="coerce").fillna(0.0)
    rr_values: list[float] = []
    reject_penalty: list[float] = []
    for raw_rr, raw_reject in zip(
        work.get("rr_model", pd.Series([{}] * len(work), index=work.index)).tolist(),
        work.get("pre_replay_reject_reason", pd.Series([""] * len(work), index=work.index)).tolist(),
    ):
        rr_model = raw_rr if isinstance(raw_rr, dict) else {}
        rr_values.append(float(rr_model.get("first_target_rr", 0.0)))
        reject_penalty.append(0.0 if str(raw_reject).strip() else 1.0)
    work["rr_first_target"] = pd.to_numeric(pd.Series(rr_values, index=work.index), errors="coerce").fillna(0.0)
    work["no_reject_penalty"] = pd.to_numeric(pd.Series(reject_penalty, index=work.index), errors="coerce").fillna(0.0)

    behavior = (
        build_behavioral_fingerprints(
            work,
            market_frame,
            max_candidates=96,
            max_bars=4096,
        )
        if market_frame is not None and not market_frame.empty
        else pd.DataFrame(columns=["candidate_id"])
    )
    if not behavior.empty and "candidate_id" in work.columns:
        work = work.merge(behavior, on="candidate_id", how="left")
    for column, default in (
        ("behavioral_fingerprint", ""),
        ("activation_density", 0.0),
        ("active_count", 0),
        ("entry_overlap_score", 0.0),
        ("exit_overlap_score", 0.0),
        ("pnl_correlation_risk", 0.0),
        ("clustering_risk", 0.0),
        ("failure_pattern_similarity", 0.0),
        ("side_distribution", '{"long_share": 0.0, "short_share": 0.0}'),
        ("hold_distribution", '{"expected_hold_bars": 0}'),
        ("regime_activation_map", "{}"),
    ):
        work[column] = work.get(column, default)
        work[column] = work[column].fillna(default) if hasattr(work[column], "fillna") else work[column]
    risk_rows = [compute_candidate_risk_card(dict(row), behavior_profile=dict(row)) for row in work.to_dict(orient="records")]
    risk_frame = pd.DataFrame(risk_rows, index=work.index)
    for column in risk_frame.columns:
        work[column] = pd.to_numeric(risk_frame[column], errors="coerce").fillna(0.0)
    work["usefulness_prior"] = [
        _usefulness_prior(dict(row))
        for row in work.to_dict(orient="records")
    ]
    work["trade_quality_bonus"] = [
        _trade_quality_bonus(dict(row))
        for row in work.to_dict(orient="records")
    ]
    return work


def _series_or_default(frame: pd.DataFrame, column: str, default: float) -> pd.Series:
    if column in frame.columns:
        return frame[column]
    return pd.Series([default] * len(frame), index=frame.index, dtype=float)


def _usefulness_prior(candidate: dict[str, Any]) -> float:
    trade_density = str(candidate.get("trade_density_expectation", "")).strip().lower()
    transfer_expectation = str(candidate.get("transfer_expectation", "")).strip().lower()
    expected_failures = {str(v).strip().lower() for v in list(candidate.get("expected_failure_modes", []) or [])}
    prior = 0.0
    if trade_density == "medium":
        prior += 0.08
    elif trade_density == "high":
        prior += 0.10
    else:
        prior -= 0.03
    if transfer_expectation in {"moderate", "high"}:
        prior += 0.04
    else:
        prior -= 0.02
    if "transfer_decay" in expected_failures:
        prior -= 0.02
    if "opportunity_sparsity" in expected_failures:
        prior -= 0.03
    return float(max(-0.15, min(0.20, prior)))


def _trade_quality_bonus(candidate: dict[str, Any]) -> float:
    rr = float(candidate.get("rr_first_target", 0.0) or 0.0)
    cost_edge = float(candidate.get("cost_edge_proxy", 0.0) or 0.0)
    hold_risk = float(candidate.get("hold_sanity_risk", 1.0) or 1.0)
    thin_risk = float(candidate.get("thin_evidence_risk", 1.0) or 1.0)
    value = (
        (min(3.0, max(0.0, rr)) / 3.0) * 0.45
        + (max(-0.01, min(0.02, cost_edge)) * 25.0) * 0.35
        + ((1.0 - max(0.0, min(1.0, hold_risk))) * 0.10)
        + ((1.0 - max(0.0, min(1.0, thin_risk))) * 0.10)
    )
    return float(max(0.0, min(1.0, value)))
