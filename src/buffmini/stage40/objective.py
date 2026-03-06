"""Stage-40 tradability labels and two-stage objective routing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TradabilityConfig:
    horizon_bars: int = 12
    tp_pct: float = 0.004
    sl_pct: float = 0.003
    round_trip_cost_pct: float = 0.001
    max_adverse_excursion_pct: float = 0.004
    stage_a_threshold: float = 0.35
    stage_b_threshold: float = 0.0


def compute_tradability_labels(frame: pd.DataFrame, *, cfg: TradabilityConfig | None = None) -> pd.DataFrame:
    """Compute tradability-centric labels without future leakage."""

    conf = cfg or TradabilityConfig()
    data = frame.copy()
    for col in ("close", "high", "low"):
        if col not in data.columns:
            raise ValueError(f"missing required column: {col}")
        data[col] = pd.to_numeric(data[col], errors="coerce").astype(float)
    ts = pd.to_datetime(data.get("timestamp"), utc=True, errors="coerce")
    data["timestamp"] = ts

    n = len(data)
    tp_before_sl: list[float] = []
    net_after_cost: list[float] = []
    mae_pct_list: list[float] = []
    tradable: list[int] = []

    for i in range(n):
        entry = float(data["close"].iloc[i]) if np.isfinite(data["close"].iloc[i]) else np.nan
        if not np.isfinite(entry) or entry <= 0:
            tp_before_sl.append(np.nan)
            net_after_cost.append(np.nan)
            mae_pct_list.append(np.nan)
            tradable.append(0)
            continue

        end = min(n, i + int(conf.horizon_bars) + 1)
        if end <= i + 1:
            tp_before_sl.append(np.nan)
            net_after_cost.append(np.nan)
            mae_pct_list.append(np.nan)
            tradable.append(0)
            continue
        window = data.iloc[i + 1 : end]
        tp_price = entry * (1.0 + float(conf.tp_pct))
        sl_price = entry * (1.0 - float(conf.sl_pct))

        hit_tp_idx = _first_hit_index(window["high"], threshold=tp_price, direction="up")
        hit_sl_idx = _first_hit_index(window["low"], threshold=sl_price, direction="down")
        tp_first = float(hit_tp_idx < hit_sl_idx) if np.isfinite(hit_tp_idx) else 0.0
        if np.isfinite(hit_tp_idx) and not np.isfinite(hit_sl_idx):
            tp_first = 1.0
        if np.isfinite(hit_sl_idx) and not np.isfinite(hit_tp_idx):
            tp_first = 0.0
        tp_before_sl.append(float(tp_first))

        fwd_close = float(window["close"].iloc[-1]) if not window.empty else entry
        gross = float((fwd_close - entry) / entry)
        net = float(gross - float(conf.round_trip_cost_pct))
        net_after_cost.append(net)

        mae = float((pd.to_numeric(window["low"], errors="coerce").min() - entry) / entry) if not window.empty else 0.0
        mae_pct_list.append(mae)
        tradable_flag = int((tp_first >= 1.0 or net > 0.0) and mae >= -float(conf.max_adverse_excursion_pct))
        tradable.append(tradable_flag)

    out = pd.DataFrame(
        {
            "timestamp": data["timestamp"],
            "tp_before_sl": pd.Series(tp_before_sl, dtype=float),
            "net_return_after_cost": pd.Series(net_after_cost, dtype=float),
            "mae_pct": pd.Series(mae_pct_list, dtype=float),
            "adverse_excursion_ok": pd.Series(mae_pct_list, dtype=float) >= -float(conf.max_adverse_excursion_pct),
            "tradable": pd.Series(tradable, dtype=int),
        }
    )
    return out


def route_two_stage_objective(
    candidates: pd.DataFrame,
    *,
    labels: pd.DataFrame,
    cfg: TradabilityConfig | None = None,
) -> dict[str, Any]:
    """Route candidates through Stage-A tradability then Stage-B robustness."""

    conf = cfg or TradabilityConfig()
    work = candidates.copy() if isinstance(candidates, pd.DataFrame) else pd.DataFrame()
    if work.empty:
        return {
            "stage_a_survivors": pd.DataFrame(),
            "stage_b_survivors": pd.DataFrame(),
            "counts": {"input": 0, "stage_a": 0, "stage_b": 0, "before_strict_direct": 0},
            "bottleneck_step": "stage_a_activation",
        }

    work["layer_score"] = pd.to_numeric(work.get("layer_score", 0.0), errors="coerce").fillna(0.0)
    work["exp_lcb_proxy"] = pd.to_numeric(work.get("exp_lcb_proxy", 0.0), errors="coerce").fillna(0.0)
    work["context"] = work.get("broad_context", "range").astype(str)

    tradable_rate = float(pd.to_numeric(labels.get("tradable", 0), errors="coerce").fillna(0).astype(int).mean()) if not labels.empty else 0.0
    net_rate = float(pd.to_numeric(labels.get("net_return_after_cost", 0.0), errors="coerce").fillna(0.0).mean()) if not labels.empty else 0.0
    tp_rate = float(pd.to_numeric(labels.get("tp_before_sl", 0.0), errors="coerce").fillna(0.0).mean()) if not labels.empty else 0.0
    context_weight = work["context"].map(_context_weight).astype(float)

    stage_a_score = (
        (work["layer_score"] * 0.45)
        + (tradable_rate * 0.30)
        + (max(0.0, net_rate) * 20.0 * 0.15)
        + (tp_rate * 0.10)
    ) * context_weight
    work["stage_a_score"] = stage_a_score
    stage_a = work.loc[work["stage_a_score"] >= float(conf.stage_a_threshold), :].copy()
    if stage_a.empty and not work.empty:
        stage_a = work.nlargest(min(3, len(work)), "stage_a_score").copy()

    stage_b_score = pd.to_numeric(stage_a.get("exp_lcb_proxy", 0.0), errors="coerce").fillna(0.0)
    stage_a["stage_b_score"] = stage_b_score
    stage_b = stage_a.loc[stage_a["stage_b_score"] >= float(conf.stage_b_threshold), :].copy()

    before_direct = int((work["exp_lcb_proxy"] >= float(conf.stage_b_threshold)).sum())
    drop_a = int(max(0, len(work) - len(stage_a)))
    drop_b = int(max(0, len(stage_a) - len(stage_b)))
    bottleneck = "stage_a_activation" if drop_a >= drop_b else "stage_b_robustness"
    return {
        "stage_a_survivors": stage_a.reset_index(drop=True),
        "stage_b_survivors": stage_b.reset_index(drop=True),
        "counts": {
            "input": int(len(work)),
            "stage_a": int(len(stage_a)),
            "stage_b": int(len(stage_b)),
            "before_strict_direct": int(before_direct),
        },
        "bottleneck_step": bottleneck,
        "label_stats": {
            "tradable_rate": tradable_rate,
            "tp_before_sl_rate": tp_rate,
            "net_return_after_cost_mean": net_rate,
        },
    }


def _first_hit_index(series: pd.Series, *, threshold: float, direction: str) -> float:
    vals = pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
    for idx, value in enumerate(vals):
        if not np.isfinite(value):
            continue
        if direction == "up" and value >= threshold:
            return float(idx)
        if direction == "down" and value <= threshold:
            return float(idx)
    return float(np.nan)


def _context_weight(value: str) -> float:
    key = str(value).strip().lower()
    if key in {"trend", "flow-dominant", "funding-stress"}:
        return 1.05
    if key in {"squeeze", "shock"}:
        return 1.0
    if key in {"range", "exhaustion", "sentiment-extreme"}:
        return 0.95
    return 1.0

