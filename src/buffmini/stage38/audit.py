"""Stage-38 lineage, OI gating, and self-learning audit helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def detect_lineage_collapse_reason(*, raw_signal_count: int, post_cost_gate_count: int, composer_signal_count: int, final_trade_count: float, shadow_reject_count: int) -> str:
    """Return explicit collapse reason between activation-hunt and engine trades."""

    raw = int(max(0, raw_signal_count))
    post_cost = int(max(0, post_cost_gate_count))
    composer = int(max(0, composer_signal_count))
    final_trades = float(max(0.0, final_trade_count))
    rejects = int(max(0, shadow_reject_count))

    if raw <= 0:
        return "no_raw_candidates"
    if post_cost <= 0:
        return "threshold_or_cost_gate_zeroed_candidates"
    if composer <= 0:
        return "composer_netting_zero_signal"
    if final_trades <= 0.0 and rejects >= composer:
        return "live_constraints_rejected_all"
    if final_trades <= 0.0:
        return "backtest_or_execution_zero_trade"
    return "no_collapse"


def oi_runtime_usage(*, frame: pd.DataFrame, oi_columns: list[str], timeframe: str, short_horizon_max: str, short_only_enabled: bool) -> dict[str, Any]:
    """Summarize OI runtime usage with explicit enforcement status."""

    cols = [col for col in oi_columns if col in frame.columns]
    non_null = int(frame[cols].notna().any(axis=1).sum()) if cols else 0
    tf_ok = _is_timeframe_shorter_or_equal(timeframe=timeframe, threshold=short_horizon_max)
    active = bool(non_null > 0 and (tf_ok or not short_only_enabled))
    rule = "short_only_enforced" if short_only_enabled else "short_only_disabled"
    return {
        "short_only_enabled": bool(short_only_enabled),
        "short_horizon_max": str(short_horizon_max),
        "timeframe": str(timeframe),
        "timeframe_allowed": bool(tf_ok),
        "oi_columns_present": cols,
        "oi_non_null_rows": int(non_null),
        "oi_active_runtime": bool(active),
        "rule": rule,
    }


def build_failure_aware_registry_rows(*, activation_payload: dict[str, Any], run_id: str) -> list[dict[str, Any]]:
    """Build deterministic learning rows even when zero-trade runs occur."""

    hunt = dict((activation_payload.get("hunt", {}) or {}))
    per_family = dict((hunt.get("per_family", {}) or {}))
    chosen_threshold = float(activation_payload.get("chosen_threshold", 0.0))
    rows: list[dict[str, Any]] = []

    if per_family:
        for family in sorted(per_family.keys()):
            item = dict(per_family.get(family, {}))
            top_rejects = dict(item.get("top_reject_reasons", {}))
            top_reason = (
                sorted(top_rejects.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[0][0]
                if top_rejects
                else "unknown"
            )
            raw = int(item.get("raw_signal_count", 0))
            final_trade_count = int(round(float(item.get("final_trade_count", 0.0))))
            status = "active" if final_trade_count > 0 else "dead_end"
            motif_tags = _failure_motif_tags(
                raw_signal_count=raw,
                composer_signal_count=int(item.get("composer_signal_count", 0)),
                final_trade_count=final_trade_count,
                top_reason=top_reason,
            )
            rows.append(
                {
                    "run_id": str(run_id),
                    "generation": 0,
                    "family": str(family),
                    "feature_subset_signature": f"family::{family}",
                    "threshold_configuration": {"activation_threshold": chosen_threshold},
                    "raw_signal_count": int(raw),
                    "activation_rate": float(item.get("activation_rate", 0.0)),
                    "top_reject_reason": str(top_reason),
                    "cost_gate_fail_rate": _rate(
                        pre=int(item.get("post_threshold_count", 0)),
                        post=int(item.get("post_cost_gate_count", 0)),
                    ),
                    "feasibility_fail_rate": _rate(
                        pre=int(item.get("post_cost_gate_count", 0)),
                        post=int(item.get("post_feasibility_count", 0)),
                    ),
                    "final_trade_count": int(final_trade_count),
                    "exp_lcb": float(item.get("avg_context_quality", 0.0)),
                    "stability_score": float(item.get("activation_rate", 0.0)),
                    "status": str(status),
                    "elite": False,
                    "failure_motif_tags": motif_tags,
                }
            )

    if rows:
        return rows

    overall = dict(hunt.get("overall", {}))
    top_rejects = dict(overall.get("top_reject_reasons", {}))
    top_reason = (
        sorted(top_rejects.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))[0][0]
        if top_rejects
        else "no_signal"
    )
    raw = int(overall.get("raw_signal_count", 0))
    composer = int(overall.get("composer_signal_count", 0))
    final_trade_count = int(round(float(overall.get("final_trade_count", 0.0))))
    motif_tags = _failure_motif_tags(
        raw_signal_count=raw,
        composer_signal_count=composer,
        final_trade_count=final_trade_count,
        top_reason=top_reason,
    )
    return [
        {
            "run_id": str(run_id),
            "generation": 0,
            "family": "__all__",
            "feature_subset_signature": "family::__all__",
            "threshold_configuration": {"activation_threshold": chosen_threshold},
            "raw_signal_count": int(raw),
            "activation_rate": float(overall.get("activation_rate", 0.0)),
            "top_reject_reason": str(top_reason),
            "cost_gate_fail_rate": _rate(
                pre=int(overall.get("post_threshold_count", 0)),
                post=int(overall.get("post_cost_gate_count", 0)),
            ),
            "feasibility_fail_rate": _rate(
                pre=int(overall.get("post_cost_gate_count", 0)),
                post=int(overall.get("post_feasibility_count", 0)),
            ),
            "final_trade_count": int(final_trade_count),
            "exp_lcb": float(overall.get("avg_context_quality", 0.0)),
            "stability_score": float(overall.get("activation_rate", 0.0)),
            "status": "dead_end" if final_trade_count <= 0 else "active",
            "elite": False,
            "failure_motif_tags": motif_tags,
        }
    ]


def _failure_motif_tags(*, raw_signal_count: int, composer_signal_count: int, final_trade_count: int, top_reason: str) -> list[str]:
    tags: list[str] = []
    if int(raw_signal_count) <= 0:
        tags.append("NO_RAW_SIGNAL")
    if int(raw_signal_count) > 0 and int(composer_signal_count) <= 0:
        tags.append("COMPOSER_ZERO")
    if int(composer_signal_count) > 0 and int(final_trade_count) <= 0:
        tags.append("NO_TRADE_AFTER_COMPOSER")
    reason = str(top_reason).strip().upper()
    if reason:
        tags.append(f"REJECT::{reason}")
    return sorted(set(tags))


def _is_timeframe_shorter_or_equal(*, timeframe: str, threshold: str) -> bool:
    minutes_map = {
        "1m": 1,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "2h": 120,
        "4h": 240,
        "1d": 1440,
    }
    tf = minutes_map.get(str(timeframe).strip().lower())
    cutoff = minutes_map.get(str(threshold).strip().lower())
    if tf is None or cutoff is None:
        return False
    return bool(tf <= cutoff)


def _rate(*, pre: int, post: int) -> float:
    p = int(max(0, pre))
    q = int(max(0, post))
    if p <= 0:
        return 0.0
    return float(np.clip((p - q) / max(1, p), 0.0, 1.0))

