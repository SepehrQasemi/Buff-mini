"""Adaptive order builder for Stage-23 choke reduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from buffmini.backtest.costs import cost_breakdown_bps, normalized_cost_cfg, one_way_slippage_for_bar
from buffmini.stage23.rejects import EXECUTION_REJECT_REASONS, RejectBreakdown, normalize_reject_reason


@dataclass(frozen=True)
class OrderBuilderConfig:
    min_stop_atr_mult: float = 0.8
    min_stop_bps: float = 8.0
    min_rr: float = 0.8
    min_trade_notional: float = 10.0
    allow_size_bump_to_min_notional: bool = True
    rr_fallback_exit_mode: str = "fixed_atr"


@dataclass(frozen=True)
class ExecutionRelaxConfig:
    allow_partial_fill: bool = True
    partial_fill_min_ratio: float = 0.30
    allow_size_reduction_on_margin_fail: bool = True
    max_size_reduction_steps: int = 5
    slippage_soft_threshold_bps: float = 15.0
    slippage_hard_threshold_bps: float = 40.0


def build_adaptive_orders(
    *,
    frame: pd.DataFrame,
    raw_side: pd.Series,
    score: pd.Series,
    cfg: dict[str, Any],
    symbol: str,
) -> dict[str, Any]:
    """Build adaptive order intents and reject events from raw directional signal."""

    if frame.empty:
        empty = pd.Series(dtype=int)
        return {
            "accepted_signal": empty,
            "orders_df": pd.DataFrame(),
            "reject_events": [],
            "breakdown": RejectBreakdown().to_payload(),
        }

    eval_cfg = dict((cfg.get("evaluation", {}) or {}).get("stage23", {}))
    order_builder = OrderBuilderConfig(**_pick_order_builder_cfg(eval_cfg.get("order_builder", {})))
    relax = ExecutionRelaxConfig(**_pick_execution_cfg(eval_cfg.get("execution", {})))

    risk_cfg = dict(cfg.get("risk", {}) or {})
    max_gross_exposure = float(risk_cfg.get("max_gross_exposure", 5.0))
    leverage = float(cfg.get("_runtime_leverage", 1.0) or 1.0)
    cost_cfg = normalized_cost_cfg(
        round_trip_cost_pct=float(cfg.get("costs", {}).get("round_trip_cost_pct", 0.1)),
        slippage_pct=float(cfg.get("costs", {}).get("slippage_pct", 0.0005)),
        cost_model_cfg=cfg.get("cost_model", {}),
    )

    close = pd.to_numeric(frame.get("close", np.nan), errors="coerce").to_numpy(dtype=float)
    atr = pd.to_numeric(frame.get("atr_14", np.nan), errors="coerce").to_numpy(dtype=float)
    volume = pd.to_numeric(frame.get("volume", np.nan), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    vol_ref = (
        pd.Series(volume, index=frame.index)
        .rolling(window=20, min_periods=1)
        .mean()
        .replace(0.0, np.nan)
        .bfill()
        .fillna(1.0)
        .to_numpy(dtype=float)
    )
    side = pd.to_numeric(raw_side, errors="coerce").fillna(0).astype(int).to_numpy(dtype=int)
    strength = pd.to_numeric(score, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")

    breakdown = RejectBreakdown()
    accepted = np.zeros(len(frame), dtype=int)
    order_rows: list[dict[str, Any]] = []
    reject_events: list[dict[str, Any]] = []

    for idx in range(len(frame)):
        direction = int(np.sign(side[idx]))
        if direction == 0:
            continue
        breakdown.register_attempt(1)

        timestamp = ts.iloc[idx].isoformat() if idx < len(ts) and pd.notna(ts.iloc[idx]) else ""
        side_label = "LONG" if direction > 0 else "SHORT"
        close_px = float(close[idx]) if np.isfinite(close[idx]) else 0.0
        atr_px = float(atr[idx]) if np.isfinite(atr[idx]) else float("nan")

        # Proposed notional from score magnitude; deterministic and bounded.
        proposed_notional = float(abs(strength[idx]) * max(close_px, 1.0))
        if proposed_notional <= 0.0:
            _reject(
                breakdown=breakdown,
                reject_events=reject_events,
                timestamp=timestamp,
                symbol=symbol,
                side=side_label,
                reason="SIZE_ZERO",
                details="score_strength_non_positive",
            )
            continue

        if close_px <= 0.0:
            _reject(
                breakdown=breakdown,
                reject_events=reject_events,
                timestamp=timestamp,
                symbol=symbol,
                side=side_label,
                reason="STOP_INVALID",
                details="close_non_positive",
            )
            continue

        min_stop_px = max(
            float(order_builder.min_stop_bps) * close_px / 10_000.0,
            float(order_builder.min_stop_atr_mult) * (atr_px if np.isfinite(atr_px) and atr_px > 0 else 0.0),
        )
        if not np.isfinite(min_stop_px) or min_stop_px <= 0.0:
            _reject(
                breakdown=breakdown,
                reject_events=reject_events,
                timestamp=timestamp,
                symbol=symbol,
                side=side_label,
                reason="STOP_INVALID",
                details="min_stop_non_finite",
            )
            continue

        stop_bps = float(min_stop_px * 10_000.0 / max(close_px, 1e-12))
        if stop_bps < float(order_builder.min_stop_bps):
            stop_bps = float(order_builder.min_stop_bps)
            min_stop_px = stop_bps * close_px / 10_000.0
        if stop_bps <= 0.0:
            _reject(
                breakdown=breakdown,
                reject_events=reject_events,
                timestamp=timestamp,
                symbol=symbol,
                side=side_label,
                reason="STOP_TOO_CLOSE",
                details="stop_bps_non_positive_after_clamp",
            )
            continue

        stop_atr = float(max(order_builder.min_stop_atr_mult, 1e-9))
        tp_atr = float(((cfg.get("evaluation", {}) or {}).get("stage10", {}) or {}).get("evaluation", {}).get("take_profit_atr_multiple", 3.0))
        rr = float(tp_atr / stop_atr)
        fallback_used = False
        if rr < float(order_builder.min_rr):
            if str(order_builder.rr_fallback_exit_mode).strip():
                fallback_used = True
            else:
                _reject(
                    breakdown=breakdown,
                    reject_events=reject_events,
                    timestamp=timestamp,
                    symbol=symbol,
                    side=side_label,
                    reason="RR_INVALID",
                    details=f"rr={rr:.6f}",
                )
                continue

        notional = proposed_notional
        if notional < float(order_builder.min_trade_notional):
            cap_for_min_check = float(max_gross_exposure) * close_px * max(1.0, leverage)
            if bool(order_builder.allow_size_bump_to_min_notional) and float(order_builder.min_trade_notional) <= cap_for_min_check:
                notional = float(order_builder.min_trade_notional)
            else:
                _reject(
                    breakdown=breakdown,
                    reject_events=reject_events,
                    timestamp=timestamp,
                    symbol=symbol,
                    side=side_label,
                    reason="SIZE_TOO_SMALL",
                    details=f"notional={notional:.6f}<min={float(order_builder.min_trade_notional):.6f}",
                )
                continue

        max_notional = float(max(0.0, float(max_gross_exposure) * close_px * max(1.0, leverage)))
        if max_notional <= 0.0:
            _reject(
                breakdown=breakdown,
                reject_events=reject_events,
                timestamp=timestamp,
                symbol=symbol,
                side=side_label,
                reason="MARGIN_FAIL",
                details="non_positive_max_notional",
            )
            continue
        if notional > max_notional:
            if bool(relax.allow_size_reduction_on_margin_fail):
                reduction_steps = 0
                while notional > max_notional and reduction_steps < int(relax.max_size_reduction_steps):
                    notional *= 0.7
                    reduction_steps += 1
                if notional > max_notional:
                    _reject(
                        breakdown=breakdown,
                        reject_events=reject_events,
                        timestamp=timestamp,
                        symbol=symbol,
                        side=side_label,
                        reason="MARGIN_FAIL",
                        details=f"max_notional={max_notional:.6f}",
                    )
                    continue
            else:
                _reject(
                    breakdown=breakdown,
                    reject_events=reject_events,
                    timestamp=timestamp,
                    symbol=symbol,
                    side=side_label,
                    reason="POLICY_CAP_HIT",
                    details=f"max_notional={max_notional:.6f}",
                )
                continue

        slippage_bps = float(one_way_slippage_for_bar(frame=frame, bar_index=idx, cost_cfg=cost_cfg, atr_col="atr_14", close_col="close") * 10_000.0)
        breakdown_bps = cost_breakdown_bps(frame=frame, bar_index=idx, cost_cfg=cost_cfg, atr_col="atr_14", close_col="close")
        spread_bps = float(breakdown_bps.get("spread_bps", 0.0))
        if spread_bps > float(relax.slippage_hard_threshold_bps):
            _reject(
                breakdown=breakdown,
                reject_events=reject_events,
                timestamp=timestamp,
                symbol=symbol,
                side=side_label,
                reason="SPREAD_TOO_HIGH",
                details=f"spread_bps={spread_bps:.6f}",
            )
            continue
        if slippage_bps > float(relax.slippage_hard_threshold_bps):
            _reject(
                breakdown=breakdown,
                reject_events=reject_events,
                timestamp=timestamp,
                symbol=symbol,
                side=side_label,
                reason="SLIPPAGE_TOO_HIGH",
                details=f"slippage_bps={slippage_bps:.6f}",
            )
            continue
        if slippage_bps > float(relax.slippage_soft_threshold_bps):
            # Soft threshold: reduce size to keep exposure realistic.
            ratio = float(relax.slippage_soft_threshold_bps) / max(slippage_bps, 1e-12)
            notional = max(float(order_builder.min_trade_notional), notional * max(0.25, min(1.0, ratio)))

        if bool(relax.allow_partial_fill):
            liq_ratio = float(volume[idx] / max(vol_ref[idx], 1e-12))
            fill_ratio = float(np.clip(liq_ratio, float(relax.partial_fill_min_ratio), 1.0))
        else:
            fill_ratio = 1.0
            if volume[idx] <= 0.0:
                _reject(
                    breakdown=breakdown,
                    reject_events=reject_events,
                    timestamp=timestamp,
                    symbol=symbol,
                    side=side_label,
                    reason="NO_FILL",
                    details="volume_non_positive",
                )
                continue

        filled_notional = notional * fill_ratio
        if filled_notional < float(order_builder.min_trade_notional):
            if bool(order_builder.allow_size_bump_to_min_notional) and float(order_builder.min_trade_notional) <= max_notional:
                filled_notional = float(order_builder.min_trade_notional)
            else:
                _reject(
                    breakdown=breakdown,
                    reject_events=reject_events,
                    timestamp=timestamp,
                    symbol=symbol,
                    side=side_label,
                    reason="SIZE_TOO_SMALL",
                    details=f"filled_notional={filled_notional:.6f}",
                )
                continue

        accepted[idx] = direction
        breakdown.register_accept(1)
        order_rows.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "side": side_label,
                "status": "VALID",
                "invalid_reason": "",
                "stop_distance_px": float(min_stop_px),
                "stop_distance_bps": float(stop_bps),
                "rr": float(rr),
                "fallback_exit_used": bool(fallback_used),
                "requested_notional": float(proposed_notional),
                "filled_notional": float(filled_notional),
                "fill_ratio": float(fill_ratio),
                "slippage_bps": float(slippage_bps),
                "spread_bps": float(spread_bps),
            }
        )

    orders_df = pd.DataFrame(order_rows)
    return {
        "accepted_signal": pd.Series(accepted, index=frame.index, dtype=int),
        "orders_df": orders_df,
        "reject_events": reject_events,
        "breakdown": breakdown.to_payload(),
    }


def _pick_order_builder_cfg(values: dict[str, Any]) -> dict[str, Any]:
    return {
        "min_stop_atr_mult": float(values.get("min_stop_atr_mult", 0.8)),
        "min_stop_bps": float(values.get("min_stop_bps", 8.0)),
        "min_rr": float(values.get("min_rr", 0.8)),
        "min_trade_notional": float(values.get("min_trade_notional", 10.0)),
        "allow_size_bump_to_min_notional": bool(values.get("allow_size_bump_to_min_notional", True)),
        "rr_fallback_exit_mode": str(values.get("rr_fallback_exit_mode", "fixed_atr")),
    }


def _pick_execution_cfg(values: dict[str, Any]) -> dict[str, Any]:
    return {
        "allow_partial_fill": bool(values.get("allow_partial_fill", True)),
        "partial_fill_min_ratio": float(values.get("partial_fill_min_ratio", 0.30)),
        "allow_size_reduction_on_margin_fail": bool(values.get("allow_size_reduction_on_margin_fail", True)),
        "max_size_reduction_steps": int(values.get("max_size_reduction_steps", 5)),
        "slippage_soft_threshold_bps": float(values.get("slippage_soft_threshold_bps", 15.0)),
        "slippage_hard_threshold_bps": float(values.get("slippage_hard_threshold_bps", 40.0)),
    }


def _reject(
    *,
    breakdown: RejectBreakdown,
    reject_events: list[dict[str, Any]],
    timestamp: str,
    symbol: str,
    side: str,
    reason: str,
    details: str,
) -> None:
    normalized = normalize_reject_reason(reason)
    if normalized not in EXECUTION_REJECT_REASONS:
        normalized = "UNKNOWN"
    breakdown.register_reject(normalized, 1)
    reject_events.append(
        {
            "timestamp": timestamp,
            "symbol": symbol,
            "side": side,
            "reason": normalized,
            "details": details,
        }
    )
