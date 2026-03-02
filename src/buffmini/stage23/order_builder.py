"""Adaptive order builder for Stage-23 choke reduction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
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
    min_trade_qty: float = 0.0
    qty_step: float = 0.0


@dataclass(frozen=True)
class ExecutionRelaxConfig:
    allow_partial_fill: bool = True
    partial_fill_min_ratio: float = 0.30
    allow_size_reduction_on_margin_fail: bool = True
    max_size_reduction_steps: int = 5
    slippage_soft_threshold_bps: float = 15.0
    slippage_hard_threshold_bps: float = 40.0


@dataclass(frozen=True)
class SizingRepairConfig:
    qty_rounding_default: str = "floor"
    qty_rounding_on_min_notional_bump: str = "ceil"
    allow_single_step_ceil_rescue: bool = True
    ceil_rescue_max_overage_steps: int = 1


@dataclass(frozen=True)
class SizingTraceRecord:
    ts: str
    symbol: str
    side: str
    price: float
    stop_price: float
    tp_price: float
    raw_size: float
    capped_size: float
    min_notional: float
    min_trade_qty: float
    qty_step: float
    bumped_to_min_notional: bool
    rounded_size_before: float
    rounded_size_after: float
    rounding_mode_used: str
    final_notional: float
    ceil_rescue_applied: bool
    cap_binding: str
    decision: str
    reject_reason: str
    reject_details: str

    def to_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, float):
                payload[key] = float(value) if np.isfinite(value) else 0.0
        return payload


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
            "adjustment_events": [],
            "sizing_trace": pd.DataFrame(),
            "sizing_trace_summary": _summarize_sizing_trace([]),
            "breakdown": RejectBreakdown().to_payload(),
        }

    eval_cfg = dict((cfg.get("evaluation", {}) or {}).get("stage23", {}))
    order_builder = OrderBuilderConfig(**_pick_order_builder_cfg(eval_cfg.get("order_builder", {})))
    relax = ExecutionRelaxConfig(**_pick_execution_cfg(eval_cfg.get("execution", {})))
    sizing_cfg = SizingRepairConfig(**_pick_sizing_cfg(eval_cfg.get("sizing", {})))
    sizing_fix_enabled = bool(eval_cfg.get("sizing_fix_enabled", True))

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
    adjustment_events: list[dict[str, Any]] = []
    sizing_trace_rows: list[dict[str, Any]] = []

    tp_atr = float(((cfg.get("evaluation", {}) or {}).get("stage10", {}) or {}).get("evaluation", {}).get("take_profit_atr_multiple", 3.0))

    for idx in range(len(frame)):
        direction = int(np.sign(side[idx]))
        if direction == 0:
            continue
        breakdown.register_attempt(1)

        timestamp = ts.iloc[idx].isoformat() if idx < len(ts) and pd.notna(ts.iloc[idx]) else ""
        side_label = "LONG" if direction > 0 else "SHORT"
        close_px = float(close[idx]) if np.isfinite(close[idx]) else 0.0
        atr_px = float(atr[idx]) if np.isfinite(atr[idx]) else float("nan")

        proposed_notional = float(abs(strength[idx]) * max(close_px, 1.0))
        raw_size = float(proposed_notional / max(close_px, 1e-12)) if close_px > 0 else 0.0
        capped_size = float(raw_size)
        notional = float(proposed_notional)
        filled_notional = float(proposed_notional)
        stop_price = float(close_px)
        tp_price = float(close_px)
        rounded_size_before = float(raw_size)
        rounded_size_after = float(raw_size)
        rounding_mode_used = "none"
        final_notional = float(proposed_notional)
        bumped_to_min_notional = False
        ceil_rescue_applied = False
        cap_binding = ""

        def emit_trace(decision: str, reject_reason: str = "", reject_details: str = "") -> None:
            sizing_trace_rows.append(
                SizingTraceRecord(
                    ts=timestamp,
                    symbol=symbol,
                    side=side_label,
                    price=close_px,
                    stop_price=stop_price,
                    tp_price=tp_price,
                    raw_size=raw_size,
                    capped_size=capped_size,
                    min_notional=float(order_builder.min_trade_notional),
                    min_trade_qty=float(order_builder.min_trade_qty),
                    qty_step=float(order_builder.qty_step),
                    bumped_to_min_notional=bool(bumped_to_min_notional),
                    rounded_size_before=rounded_size_before,
                    rounded_size_after=rounded_size_after,
                    rounding_mode_used=rounding_mode_used,
                    final_notional=final_notional,
                    ceil_rescue_applied=ceil_rescue_applied,
                    cap_binding=cap_binding,
                    decision=decision,
                    reject_reason=reject_reason,
                    reject_details=reject_details,
                ).to_payload()
            )

        def reject(reason: str, details: str) -> None:
            _reject(
                breakdown=breakdown,
                reject_events=reject_events,
                timestamp=timestamp,
                symbol=symbol,
                side=side_label,
                reason=reason,
                details=details,
            )
            emit_trace("REJECTED", normalize_reject_reason(reason), details)

        if proposed_notional <= 0.0 or raw_size <= 0.0:
            reject("SIZE_TOO_SMALL", "score_strength_non_positive")
            continue

        if close_px <= 0.0:
            reject("STOP_INVALID", "close_non_positive")
            continue

        min_stop_px = max(
            float(order_builder.min_stop_bps) * close_px / 10_000.0,
            float(order_builder.min_stop_atr_mult) * (atr_px if np.isfinite(atr_px) and atr_px > 0 else 0.0),
        )
        if not np.isfinite(min_stop_px) or min_stop_px <= 0.0:
            reject("STOP_INVALID", "min_stop_non_finite")
            continue

        stop_bps = float(min_stop_px * 10_000.0 / max(close_px, 1e-12))
        if stop_bps < float(order_builder.min_stop_bps):
            stop_bps = float(order_builder.min_stop_bps)
            min_stop_px = stop_bps * close_px / 10_000.0
        if stop_bps <= 0.0:
            reject("STOP_TOO_CLOSE", "stop_bps_non_positive_after_clamp")
            continue

        stop_atr = float(max(order_builder.min_stop_atr_mult, 1e-9))
        rr = float(tp_atr / stop_atr)
        stop_price = float(close_px - direction * min_stop_px)
        tp_price = float(close_px + direction * (min_stop_px * tp_atr / max(stop_atr, 1e-12)))
        fallback_used = False
        if rr < float(order_builder.min_rr):
            if str(order_builder.rr_fallback_exit_mode).strip():
                fallback_used = True
            else:
                reject("RR_INVALID", f"rr={rr:.6f}")
                continue

        if notional < float(order_builder.min_trade_notional):
            if bool(order_builder.allow_size_bump_to_min_notional):
                notional = float(order_builder.min_trade_notional)
                bumped_to_min_notional = True
            else:
                reject("SIZE_TOO_SMALL", f"notional={notional:.6f}<min={float(order_builder.min_trade_notional):.6f}")
                continue

        max_notional = float(max(0.0, float(max_gross_exposure) * close_px * max(1.0, leverage)))
        if max_notional <= 0.0:
            cap_binding = "margin"
            reject("MARGIN_FAIL", "non_positive_max_notional")
            continue

        if notional > max_notional:
            if bool(relax.allow_size_reduction_on_margin_fail):
                cap_binding = "margin"
                reduction_steps = 0
                while notional > max_notional and reduction_steps < int(relax.max_size_reduction_steps):
                    prev = float(notional)
                    notional *= 0.7
                    reduction_steps += 1
                    adjustment_events.append(
                        {
                            "timestamp": timestamp,
                            "symbol": symbol,
                            "side": side_label,
                            "event": "size_reduction_margin",
                            "step": int(reduction_steps),
                            "before_notional": prev,
                            "after_notional": float(notional),
                            "max_notional": float(max_notional),
                        }
                    )
                if notional > max_notional:
                    reject("MARGIN_FAIL", f"max_notional={max_notional:.6f}")
                    continue
            else:
                cap_binding = "policy_cap"
                reject("POLICY_CAP_HIT", f"max_notional={max_notional:.6f}")
                continue

        slippage_bps = float(one_way_slippage_for_bar(frame=frame, bar_index=idx, cost_cfg=cost_cfg, atr_col="atr_14", close_col="close") * 10_000.0)
        breakdown_bps = cost_breakdown_bps(frame=frame, bar_index=idx, cost_cfg=cost_cfg, atr_col="atr_14", close_col="close")
        spread_bps = float(breakdown_bps.get("spread_bps", 0.0))
        if spread_bps > float(relax.slippage_hard_threshold_bps):
            reject("SPREAD_TOO_HIGH", f"spread_bps={spread_bps:.6f}")
            continue
        if slippage_bps > float(relax.slippage_hard_threshold_bps):
            reject("SLIPPAGE_TOO_HIGH", f"slippage_bps={slippage_bps:.6f}")
            continue

        if slippage_bps > float(relax.slippage_soft_threshold_bps):
            ratio = float(relax.slippage_soft_threshold_bps) / max(slippage_bps, 1e-12)
            prev_notional = float(notional)
            notional = max(float(order_builder.min_trade_notional), notional * max(0.25, min(1.0, ratio)))
            if notional < prev_notional:
                adjustment_events.append(
                    {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "side": side_label,
                        "event": "size_reduction_slippage",
                        "step": 1,
                        "before_notional": prev_notional,
                        "after_notional": float(notional),
                        "slippage_bps": float(slippage_bps),
                    }
                )

        if bool(relax.allow_partial_fill):
            liq_ratio = float(volume[idx] / max(vol_ref[idx], 1e-12))
            fill_ratio = float(np.clip(liq_ratio, float(relax.partial_fill_min_ratio), 1.0))
        else:
            fill_ratio = 1.0
            if volume[idx] <= 0.0:
                reject("NO_FILL", "volume_non_positive")
                continue

        filled_notional = float(notional * fill_ratio)
        if filled_notional < float(order_builder.min_trade_notional):
            if bool(order_builder.allow_size_bump_to_min_notional) and float(order_builder.min_trade_notional) <= max_notional + 1e-12:
                filled_notional = float(order_builder.min_trade_notional)
                bumped_to_min_notional = True
            else:
                reject("SIZE_TOO_SMALL", f"filled_notional={filled_notional:.6f}")
                continue

        capped_size = float(min(filled_notional, max_notional) / max(close_px, 1e-12))
        rounded_size_before = float(capped_size)
        if float(order_builder.min_trade_qty) > 0.0 and rounded_size_before < float(order_builder.min_trade_qty):
            rounded_size_before = float(order_builder.min_trade_qty)

        rounding_mode_used = str(sizing_cfg.qty_rounding_default)
        if bumped_to_min_notional:
            rounding_mode_used = str(sizing_cfg.qty_rounding_on_min_notional_bump)

        rounded_size_after = round_qty_to_step(rounded_size_before, float(order_builder.qty_step), rounding_mode_used)

        if rounded_size_after <= 0.0 and rounded_size_before > 0.0:
            if bool(sizing_fix_enabled) and bool(sizing_cfg.allow_single_step_ceil_rescue):
                rescue_qty = round_qty_to_step(rounded_size_before, float(order_builder.qty_step), "ceil")
                rescue_notional = float(rescue_qty * close_px)
                if float(order_builder.qty_step) > 0.0:
                    overage_steps = float(max(0.0, (rescue_qty - rounded_size_before) / float(order_builder.qty_step)))
                else:
                    overage_steps = 0.0
                if (
                    rescue_qty > 0.0
                    and overage_steps <= float(max(0, int(sizing_cfg.ceil_rescue_max_overage_steps)))
                    and rescue_notional <= max_notional + 1e-12
                ):
                    rounded_size_after = float(rescue_qty)
                    rounding_mode_used = "ceil_rescue"
                    ceil_rescue_applied = True
                elif rescue_notional > max_notional + 1e-12:
                    if cap_binding == "":
                        cap_binding = "policy_cap"
                    reject(_binding_reject_reason(cap_binding), "ceil_rescue_exceeds_cap")
                    continue
            if rounded_size_after <= 0.0:
                if bool(sizing_fix_enabled) and cap_binding in {"policy_cap", "margin"}:
                    reason = _binding_reject_reason(cap_binding)
                else:
                    reason = "SIZE_ZERO" if not bool(sizing_fix_enabled) else "SIZE_TOO_SMALL"
                reject(reason, f"qty_before_round={rounded_size_before:.12f},step={float(order_builder.qty_step):.12f}")
                continue

        if float(order_builder.min_trade_qty) > 0.0 and rounded_size_after < float(order_builder.min_trade_qty) - 1e-12:
            reject("SIZE_TOO_SMALL", f"rounded_qty={rounded_size_after:.12f}<min_qty={float(order_builder.min_trade_qty):.12f}")
            continue

        final_notional = float(rounded_size_after * close_px)
        if final_notional < float(order_builder.min_trade_notional) - 1e-12:
            if bool(order_builder.allow_size_bump_to_min_notional):
                bump_qty = round_qty_to_step(
                    float(order_builder.min_trade_notional) / max(close_px, 1e-12),
                    float(order_builder.qty_step),
                    str(sizing_cfg.qty_rounding_on_min_notional_bump),
                )
                bump_notional = float(bump_qty * close_px)
                if bump_qty > 0.0 and bump_notional <= max_notional + 1e-12 and bump_notional >= float(order_builder.min_trade_notional) - 1e-12:
                    rounded_size_before = rounded_size_after
                    rounded_size_after = float(bump_qty)
                    final_notional = float(bump_notional)
                    rounding_mode_used = str(sizing_cfg.qty_rounding_on_min_notional_bump)
                    bumped_to_min_notional = True
                else:
                    if bump_notional > max_notional + 1e-12 and cap_binding in {"policy_cap", "margin"}:
                        reject(_binding_reject_reason(cap_binding), "min_notional_bump_exceeds_binding")
                    else:
                        reject("SIZE_TOO_SMALL", "min_notional_bump_failed")
                    continue
            else:
                reject("SIZE_TOO_SMALL", f"final_notional={final_notional:.6f}<min_notional={float(order_builder.min_trade_notional):.6f}")
                continue

        if final_notional > max_notional + 1e-12:
            cap_binding = "policy_cap"
            reject("POLICY_CAP_HIT", f"final_notional={final_notional:.6f}>max_notional={max_notional:.6f}")
            continue

        accepted[idx] = direction
        breakdown.register_accept(1)
        emit_trace("ACCEPTED")
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
                "rounded_qty": float(rounded_size_after),
                "final_notional": float(final_notional),
                "fill_ratio": float(fill_ratio),
                "slippage_bps": float(slippage_bps),
                "spread_bps": float(spread_bps),
                "adjustment_events_count": int(
                    sum(1 for event in adjustment_events if event.get("timestamp", "") == timestamp and event.get("symbol") == symbol)
                ),
            }
        )

    orders_df = pd.DataFrame(order_rows)
    sizing_trace_df = pd.DataFrame(sizing_trace_rows)
    return {
        "accepted_signal": pd.Series(accepted, index=frame.index, dtype=int),
        "orders_df": orders_df,
        "reject_events": reject_events,
        "adjustment_events": adjustment_events,
        "sizing_trace": sizing_trace_df,
        "sizing_trace_summary": _summarize_sizing_trace(sizing_trace_rows),
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
        "min_trade_qty": float(values.get("min_trade_qty", 0.0)),
        "qty_step": float(values.get("qty_step", 0.0)),
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


def _pick_sizing_cfg(values: dict[str, Any]) -> dict[str, Any]:
    return {
        "qty_rounding_default": str(values.get("qty_rounding_default", "floor")),
        "qty_rounding_on_min_notional_bump": str(values.get("qty_rounding_on_min_notional_bump", "ceil")),
        "allow_single_step_ceil_rescue": bool(values.get("allow_single_step_ceil_rescue", True)),
        "ceil_rescue_max_overage_steps": int(values.get("ceil_rescue_max_overage_steps", 1)),
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


def _summarize_sizing_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "attempted": 0,
            "accepted": 0,
            "rejected": 0,
            "raw_size_min": 0.0,
            "raw_size_median": 0.0,
            "raw_size_p95": 0.0,
            "zero_size_count": 0,
            "rescued_by_ceil_count": 0,
            "bumped_to_min_notional_count": 0,
            "cap_binding_reject_count": 0,
            "reject_reason_counts": {},
        }
    frame = pd.DataFrame(rows)
    raw_size = pd.to_numeric(frame.get("raw_size", 0.0), errors="coerce").fillna(0.0)
    rounded_after = pd.to_numeric(frame.get("rounded_size_after", 0.0), errors="coerce").fillna(0.0)
    decision = frame.get("decision", pd.Series(dtype=str)).astype(str)
    reject_reason = frame.get("reject_reason", pd.Series(dtype=str)).astype(str).replace("", "ACCEPTED")
    cap_binding = frame.get("cap_binding", pd.Series(dtype=str)).astype(str)
    attempted = int(len(frame))
    accepted = int((decision == "ACCEPTED").sum())
    rejected = int(attempted - accepted)
    return {
        "attempted": attempted,
        "accepted": accepted,
        "rejected": rejected,
        "raw_size_min": float(raw_size.min()),
        "raw_size_median": float(raw_size.median()),
        "raw_size_p95": float(raw_size.quantile(0.95)),
        "zero_size_count": int(((raw_size > 0.0) & (rounded_after <= 0.0)).sum()),
        "rescued_by_ceil_count": int(pd.to_numeric(frame.get("ceil_rescue_applied", False), errors="coerce").fillna(False).astype(bool).sum()),
        "bumped_to_min_notional_count": int(pd.to_numeric(frame.get("bumped_to_min_notional", False), errors="coerce").fillna(False).astype(bool).sum()),
        "cap_binding_reject_count": int(((decision == "REJECTED") & (cap_binding != "")).sum()),
        "reject_reason_counts": {
            str(k): int(v) for k, v in reject_reason.value_counts().sort_index().items()
        },
    }


def _binding_reject_reason(binding: str) -> str:
    return "POLICY_CAP_HIT" if str(binding) == "policy_cap" else "MARGIN_FAIL"


def round_qty_to_step(qty: float, step: float, mode: str) -> float:
    quantity = float(max(0.0, qty))
    quantum = float(step)
    rounding_mode = str(mode or "floor").strip().lower()
    if quantum <= 0.0:
        return quantity
    units = quantity / quantum
    if rounding_mode == "ceil":
        rounded_units = int(np.ceil(units - 1e-12))
    elif rounding_mode == "nearest":
        rounded_units = int(np.floor(units + 0.5))
    else:
        rounded_units = int(np.floor(units + 1e-12))
    if rounded_units < 0:
        rounded_units = 0
    return float(rounded_units * quantum)
