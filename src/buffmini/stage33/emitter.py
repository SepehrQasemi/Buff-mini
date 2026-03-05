"""Signal emitter helpers for Stage-33 local policy inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.execution.feasibility import min_required_equity, min_required_risk_pct


def load_policy(path: Path) -> dict[str, Any]:
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))


def emit_signal_payload(
    *,
    frame: pd.DataFrame,
    policy: dict[str, Any],
    symbol: str,
    timeframe: str,
    asof_ts: str | None = None,
    equity: float = 1000.0,
    min_notional: float = 10.0,
) -> dict[str, Any]:
    if frame.empty:
        raise ValueError("frame is empty")
    work = frame.copy().reset_index(drop=True)
    work["timestamp"] = pd.to_datetime(work.get("timestamp"), utc=True, errors="coerce")
    if asof_ts:
        asof = pd.to_datetime(asof_ts, utc=True, errors="coerce")
        work = work.loc[work["timestamp"] <= asof].reset_index(drop=True)
    if work.empty:
        raise ValueError("no rows available at asof timestamp")

    row = work.iloc[-1]
    context = str(row.get("ctx_state", row.get("context_label", "GLOBAL")))
    contexts = dict(policy.get("contexts", {}))
    selected = dict(contexts.get(context, {}))
    candidates = list(selected.get("candidates", []))
    context_weights = [float(item.get("weight", 0.0)) for item in candidates]
    confidence = float(np.clip(np.sum(context_weights), 0.0, 1.0))

    close = float(pd.to_numeric(pd.Series([row.get("close", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
    ema = float(pd.to_numeric(pd.Series([row.get("ema_50", close)]), errors="coerce").fillna(close).iloc[0])
    momentum = close - ema
    if confidence <= 1e-9 or not candidates:
        action = "FLAT"
    else:
        action = "LONG" if momentum > 0 else "SHORT" if momentum < 0 else "FLAT"
    size_pct = float(np.clip(0.01 + 0.09 * confidence, 0.0, 0.20))
    leverage = int(np.clip(np.ceil(1.0 + confidence * 2.0), 1, 3))

    atr_pct = float(pd.to_numeric(pd.Series([row.get("atr_pct", 0.01)]), errors="coerce").fillna(0.01).iloc[0])
    stop_dist_pct = float(max(atr_pct, 0.001))
    fee_rt_pct = 0.1
    min_risk = float(
        min_required_risk_pct(
            equity=float(equity),
            min_notional=float(min_notional),
            stop_dist_pct=float(stop_dist_pct),
            cost_rt_pct=float(fee_rt_pct),
            max_notional_pct=1.0,
        )
    )
    min_eq = float(
        min_required_equity(
            risk_pct=float(max(size_pct, 1e-6)),
            min_notional=float(min_notional),
            stop_dist_pct=float(stop_dist_pct),
            cost_rt_pct=float(fee_rt_pct),
            max_notional_pct=1.0,
        )
    )
    feasible = bool(size_pct >= min_risk)

    payload = {
        "symbol": str(symbol),
        "timeframe": str(timeframe),
        "asof_ts": pd.Timestamp(row["timestamp"]).isoformat() if pd.notna(row["timestamp"]) else None,
        "context_probabilities": _context_probs(row, context),
        "recommended_action": str(action),
        "confidence": float(confidence),
        "entry_conditions_summary": [str(item.get("candidate_id", "")) for item in candidates],
        "stop_exit_policy": {
            "exit_mode": "fixed_atr",
            "stop_distance_pct": float(stop_dist_pct),
            "take_profit_atr_multiple": 3.0,
            "max_hold_bars": 24,
        },
        "sizing": {
            "equity_pct": float(size_pct),
            "leverage_suggestion": int(leverage),
        },
        "feasibility_notes": {
            "feasible_now": bool(feasible),
            "min_equity_required": float(min_eq),
            "min_risk_floor_pct": float(min_risk),
            "min_notional": float(min_notional),
        },
        "explanation": {
            "context": context,
            "momentum_vs_ema50": float(momentum),
            "active_candidates_count": int(len(candidates)),
            "top_candidates": [str(item.get("candidate_id", "")) for item in candidates[:3]],
        },
    }
    return payload


def _context_probs(row: pd.Series, fallback_context: str) -> dict[str, float]:
    cols = [c for c in row.index if str(c).startswith("context_prob_")]
    if not cols:
        return {str(fallback_context): 1.0}
    out: dict[str, float] = {}
    for idx, col in enumerate(sorted(cols)):
        val = float(pd.to_numeric(pd.Series([row.get(col, 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        out[f"CTX_{idx}"] = float(max(0.0, val))
    total = float(sum(out.values()))
    if total <= 0.0:
        return {str(fallback_context): 1.0}
    return {k: float(v / total) for k, v in out.items()}

