"""Unified Stage-23 eligibility gate (context/confirm/riskgate merge)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def evaluate_eligibility(
    *,
    frame: pd.DataFrame,
    raw_side: pd.Series,
    family: str,
    policy_snapshot: dict[str, Any],
    symbol: str,
) -> dict[str, Any]:
    """Evaluate one deterministic eligibility score gate for all raw entry candidates."""

    idx = frame.index
    side = pd.to_numeric(raw_side, errors="coerce").fillna(0).astype(int)
    active = side != 0
    if frame.empty:
        empty = pd.Series(dtype=float)
        return {"eligible": pd.Series(dtype=bool), "score": empty, "reasons": pd.Series(dtype=object), "trace_rows": []}

    trend = _series(frame, "score_trend", fallback_col="ctx_score_trend", default=0.5)
    range_score = _series(frame, "score_range", fallback_col="ctx_score_range", default=0.5)
    chop = _series(frame, "score_chop", fallback_col="ctx_score_chop", default=0.0)
    vol_rank = _series(frame, "atr_pct_rank_252", fallback_col="", default=0.5)
    regime = frame.get("regime_label_stage10", frame.get("ctx_state", pd.Series(["UNKNOWN"] * len(frame), index=idx))).astype(str)
    ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce")

    family_text = str(family).strip().lower()
    if family_text in {"price", "trend", "breakout", "ma_slopepullback", "breakoutretest"}:
        align = trend.to_numpy(dtype=float)
    elif family_text in {"volatility", "volcompressionbreakout"}:
        align = np.maximum(trend.to_numpy(dtype=float), vol_rank.to_numpy(dtype=float))
    else:
        align = range_score.to_numpy(dtype=float)

    vol_ok = np.clip(1.0 - np.abs(vol_rank.to_numpy(dtype=float) - 0.5) * 2.0, 0.0, 1.0)
    chop_ok = np.clip(1.0 - chop.to_numpy(dtype=float), 0.0, 1.0)
    score_values = np.clip((0.50 * align) + (0.30 * vol_ok) + (0.20 * chop_ok), 0.0, 1.0)
    score = pd.Series(score_values, index=idx, dtype=float)

    eligibility_cfg = dict((policy_snapshot.get("eligibility", {}) or {}))
    min_score_default = float(eligibility_cfg.get("min_score_default", 0.35))
    per_regime = dict(eligibility_cfg.get("per_regime_thresholds", {}) or {})
    thresholds = pd.Series(
        [
            float(per_regime.get(str(r), min_score_default))
            for r in regime.to_list()
        ],
        index=idx,
        dtype=float,
    ).clip(lower=0.0, upper=1.0)
    eligible = (active) & (score >= thresholds)

    reasons: list[str] = []
    trace_rows: list[dict[str, Any]] = []
    for i in range(len(frame)):
        if not bool(active.iloc[i]):
            reasons.append("")
            continue
        reason_parts: list[str] = []
        if float(score.iloc[i]) < float(thresholds.iloc[i]):
            if float(align[i]) < 0.35:
                reason_parts.append("no_trend_confirmation")
            if float(vol_ok[i]) < 0.35:
                reason_parts.append("low_volatility")
            if float(chop_ok[i]) < 0.35:
                reason_parts.append("risk_policy_conflict")
        if not reason_parts and not bool(eligible.iloc[i]):
            reason_parts.append("eligibility_score_below_threshold")
        reason_text = "|".join(reason_parts)
        reasons.append(reason_text)
        trace_rows.append(
            {
                "ts": ts.iloc[i].isoformat() if i < len(ts) and pd.notna(ts.iloc[i]) else "",
                "symbol": str(symbol),
                "score": float(score.iloc[i]),
                "threshold": float(thresholds.iloc[i]),
                "eligible": bool(eligible.iloc[i]),
                "reasons": reason_text,
                "regime": str(regime.iloc[i]),
                "family": str(family),
            }
        )

    return {
        "eligible": pd.Series(eligible.to_numpy(dtype=bool), index=idx, dtype=bool),
        "score": score,
        "reasons": pd.Series(reasons, index=idx, dtype=object),
        "trace_rows": trace_rows,
    }


def _series(frame: pd.DataFrame, column: str, *, fallback_col: str, default: float) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default).clip(lower=0.0, upper=1.0)
    if fallback_col and fallback_col in frame.columns:
        return pd.to_numeric(frame[fallback_col], errors="coerce").fillna(default).clip(lower=0.0, upper=1.0)
    return pd.Series(float(default), index=frame.index, dtype=float)

