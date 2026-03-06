"""Stage-37 activation-hunt metrics and deterministic threshold calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ActivationHuntConfig:
    """Activation-hunt controls for score-threshold and quality-gate sweeps."""

    threshold_grid: tuple[float, ...] = (
        0.00,
        0.02,
        0.04,
        0.06,
        0.08,
        0.10,
        0.12,
        0.15,
        0.20,
        0.25,
        0.30,
    )
    quality_floor: float = 0.0
    min_quality_floor: float = -0.02
    min_selected_rows: int = 1


def mode_settings(mode: str) -> dict[str, Any]:
    """Return deterministic mode settings for strict vs activation-hunt analysis."""

    text = str(mode).strip().lower()
    if text == "hunt":
        return {
            "name": "hunt",
            "quality_floor": -0.02,
            "threshold_floor": 0.0,
            "cost_mode": "research_light",
        }
    return {
        "name": "strict",
        "quality_floor": 0.0,
        "threshold_floor": 0.0,
        "cost_mode": "live_strict",
    }


def calibrate_thresholds(
    *,
    trace_df: pd.DataFrame,
    context_quality: dict[str, float],
    cfg: ActivationHuntConfig | None = None,
) -> dict[str, Any]:
    """Sweep thresholds deterministically and select best activation under quality floor."""

    conf = cfg or ActivationHuntConfig()
    normalized = _normalize_trace(trace_df)
    curves: list[dict[str, Any]] = []
    for threshold in conf.threshold_grid:
        metrics = compute_activation_metrics(
            trace_df=normalized,
            context_quality=context_quality,
            threshold=float(threshold),
            quality_floor=float(conf.min_quality_floor),
        )
        curves.append(
            {
                "threshold": float(threshold),
                "selected_rows": int(metrics.get("post_cost_gate_count", 0)),
                "post_feasibility_count": int(metrics.get("post_feasibility_count", 0)),
                "avg_context_quality": float(metrics.get("avg_context_quality", 0.0)),
            }
        )

    viable = [
        row
        for row in curves
        if int(row["selected_rows"]) >= int(conf.min_selected_rows)
        and float(row["avg_context_quality"]) >= float(conf.quality_floor)
    ]
    ranked = viable if viable else curves
    ranked = sorted(
        ranked,
        key=lambda row: (
            float(row["post_feasibility_count"]),
            float(row["selected_rows"]),
            float(row["avg_context_quality"]),
            -float(row["threshold"]),
        ),
        reverse=True,
    )
    best = ranked[0] if ranked else {"threshold": 0.0, "selected_rows": 0, "post_feasibility_count": 0, "avg_context_quality": 0.0}
    return {
        "chosen_threshold": float(best["threshold"]),
        "chosen_selected_rows": int(best["selected_rows"]),
        "chosen_post_feasibility_count": int(best["post_feasibility_count"]),
        "chosen_avg_context_quality": float(best["avg_context_quality"]),
        "curves": curves,
    }


def compute_activation_metrics(
    *,
    trace_df: pd.DataFrame,
    context_quality: dict[str, float],
    threshold: float,
    quality_floor: float,
    rejected_keys: set[tuple[str, str]] | None = None,
    final_trade_count: float = 0.0,
) -> dict[str, Any]:
    """Compute gate counts for one threshold and quality floor."""

    trace = _normalize_trace(trace_df)
    if trace.empty:
        return {
            "raw_signal_count": 0,
            "post_threshold_count": 0,
            "post_cost_gate_count": 0,
            "post_feasibility_count": 0,
            "final_trade_count": 0.0,
            "activation_rate": 0.0,
            "avg_context_quality": 0.0,
        }

    raw_mask = trace["raw_signal_flag"].astype(bool)
    thr_mask = raw_mask & (trace["abs_net_score"] >= float(threshold))
    quality = trace["context"].map(lambda ctx: float(context_quality.get(str(ctx), 0.0)))
    cost_mask = thr_mask & (quality >= float(quality_floor))

    reject_set = rejected_keys or set()
    key_series = list(zip(trace["timestamp_key"].astype(str), trace["context"].astype(str), strict=False))
    feasibility_mask = cost_mask.copy()
    if reject_set:
        blocked = pd.Series([(key[0], key[1]) in reject_set for key in key_series], index=trace.index, dtype=bool)
        feasibility_mask = feasibility_mask & (~blocked)

    raw_count = int(raw_mask.sum())
    post_threshold = int(thr_mask.sum())
    post_cost = int(cost_mask.sum())
    post_feasible = int(feasibility_mask.sum())
    final_count = float(min(max(0.0, final_trade_count), float(post_feasible))) if final_trade_count > 0 else float(post_feasible)
    activation_rate = float(final_count / max(1, raw_count))
    avg_quality = float(quality.loc[cost_mask].mean()) if bool(cost_mask.any()) else 0.0
    return {
        "raw_signal_count": raw_count,
        "post_threshold_count": post_threshold,
        "post_cost_gate_count": post_cost,
        "post_feasibility_count": post_feasible,
        "final_trade_count": float(final_count),
        "activation_rate": activation_rate,
        "avg_context_quality": avg_quality,
    }


def compute_reject_chain_metrics(
    *,
    trace_df: pd.DataFrame,
    shadow_df: pd.DataFrame,
    finalists_df: pd.DataFrame,
    threshold: float,
    quality_floor: float,
    final_trade_count: float,
) -> dict[str, Any]:
    """Compute overall and per-family reject-chain metrics."""

    trace = _normalize_trace(trace_df)
    shadow = _normalize_shadow(shadow_df)
    finalists = _normalize_finalists(finalists_df)

    context_quality = (
        finalists.groupby("context", dropna=False)["exp_lcb"].max().to_dict()
        if not finalists.empty
        else {}
    )
    rejected_keys = set(zip(shadow["timestamp_key"].astype(str), shadow["context"].astype(str), strict=False))

    overall = compute_activation_metrics(
        trace_df=trace,
        context_quality=context_quality,
        threshold=float(threshold),
        quality_floor=float(quality_floor),
        rejected_keys=rejected_keys,
        final_trade_count=float(final_trade_count),
    )
    overall["top_reject_reasons"] = (
        shadow["reason"].astype(str).value_counts(dropna=False).head(8).to_dict() if not shadow.empty else {}
    )

    family_map = {
        str(row["candidate_id"]): str(row["family"])
        for row in finalists.to_dict(orient="records")
        if str(row.get("candidate_id", "")).strip()
    }
    families = sorted({v for v in family_map.values() if v})
    per_family: dict[str, Any] = {}
    for family in families:
        family_rows = _family_slice(trace, family=family, family_map=family_map)
        if family_rows.empty:
            continue
        metrics = compute_activation_metrics(
            trace_df=family_rows,
            context_quality=context_quality,
            threshold=float(threshold),
            quality_floor=float(quality_floor),
            rejected_keys=rejected_keys,
            final_trade_count=0.0,
        )
        keys = set(zip(family_rows["timestamp_key"].astype(str), family_rows["context"].astype(str), strict=False))
        fam_reject = shadow.loc[
            shadow.apply(lambda row: (str(row["timestamp_key"]), str(row["context"])) in keys, axis=1)
        ]
        metrics["top_reject_reasons"] = fam_reject["reason"].astype(str).value_counts(dropna=False).head(5).to_dict()
        per_family[family] = metrics

    return {
        "threshold": float(threshold),
        "quality_floor": float(quality_floor),
        "overall": overall,
        "per_family": per_family,
        "context_quality": {str(k): float(v) for k, v in context_quality.items()},
    }


def _normalize_trace(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(
            columns=[
                "timestamp",
                "timestamp_key",
                "context",
                "net_score",
                "abs_net_score",
                "active_candidates",
                "raw_signal_flag",
            ]
        )
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out.get("timestamp"), utc=True, errors="coerce")
    out["timestamp_key"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").fillna("")
    out["context"] = out.get("context", "UNKNOWN").astype(str).fillna("UNKNOWN")
    out["net_score"] = pd.to_numeric(out.get("net_score", 0.0), errors="coerce").fillna(0.0)
    out["abs_net_score"] = out["net_score"].abs()
    active = out.get("active_candidates", "").astype(str).fillna("")
    out["active_candidates"] = active
    out["raw_signal_flag"] = active.str.len().gt(0) | out["net_score"].ne(0.0)
    return out


def _normalize_shadow(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(columns=["timestamp_key", "context", "reason"])
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out.get("timestamp"), utc=True, errors="coerce")
    out["timestamp_key"] = out["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%SZ").fillna("")
    out["context"] = out.get("context", "UNKNOWN").astype(str).fillna("UNKNOWN")
    out["reason"] = out.get("reason", "UNKNOWN").astype(str).fillna("UNKNOWN")
    return out


def _normalize_finalists(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(columns=["candidate_id", "family", "context", "exp_lcb"])
    out = frame.copy()
    out["candidate_id"] = out.get("candidate_id", "").astype(str).fillna("")
    out["family"] = out.get("family", "unknown").astype(str).fillna("unknown")
    out["context"] = out.get("context", "UNKNOWN").astype(str).fillna("UNKNOWN")
    out["exp_lcb"] = pd.to_numeric(out.get("exp_lcb", 0.0), errors="coerce").fillna(0.0)
    return out


def _family_slice(trace: pd.DataFrame, *, family: str, family_map: dict[str, str]) -> pd.DataFrame:
    if trace.empty:
        return trace
    fam = str(family)

    def _has_family(active: str) -> bool:
        ids = [token.strip() for token in str(active).split(",") if token.strip()]
        for cid in ids:
            if str(family_map.get(cid, "")) == fam:
                return True
        return False

    mask = trace["active_candidates"].map(_has_family).astype(bool)
    return trace.loc[mask].copy()
