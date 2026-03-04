"""Adaptive order builder for Stage-23 choke reduction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from buffmini.backtest.costs import cost_breakdown_bps, normalized_cost_cfg, one_way_slippage_for_bar
from buffmini.execution.feasibility import explain_reject
from buffmini.execution.feasibility_floor import calculate_min_risk_pct
from buffmini.execution.margin_model import (
    PolicyCaps,
    apply_exposure_caps,
    compute_margin_required,
    is_trade_feasible,
)
from buffmini.stage23.rejects import EXECUTION_REJECT_REASONS, RejectBreakdown, normalize_reject_reason
from buffmini.stage24.sizing import (
    compute_notional_alloc_pct,
    compute_notional_risk_pct,
    compute_risk_pct,
    cost_rt_pct_from_config,
    is_known_reject_reason,
)


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
    max_allowed_notional: float
    margin_required: float
    margin_limit: float
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


def _empty_shadow_live_summary(enabled: bool) -> dict[str, Any]:
    return {
        "enabled": bool(enabled),
        "research_accepted_count": 0,
        "live_pass_count": 0,
        "research_accepted_but_live_rejected_count": 0,
        "live_reject_reason_counts": {},
        "live_reject_rate": 0.0,
    }


def _shadow_live_reject_reason(
    *,
    final_notional: float,
    final_qty: float,
    live_min_notional: float,
    live_min_trade_qty: float,
    live_qty_step: float,
) -> str:
    if final_notional <= 0.0 or final_qty <= 0.0:
        return "SIZE_ZERO"
    if live_min_notional > 0.0 and final_notional < live_min_notional - 1e-12:
        return "SIZE_TOO_SMALL"
    if live_min_trade_qty > 0.0 and final_qty < live_min_trade_qty - 1e-12:
        return "SIZE_TOO_SMALL"
    if live_qty_step > 0.0:
        units = final_qty / live_qty_step
        if abs(units - round(units)) > 1e-9:
            return "SIZE_TOO_SMALL"
    return ""


def _register_shadow_live_reject(summary: dict[str, Any], reason: str) -> None:
    key = normalize_reject_reason(reason)
    counts = dict(summary.get("live_reject_reason_counts", {}))
    counts[key] = int(counts.get(key, 0)) + 1
    summary["live_reject_reason_counts"] = counts


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
            "risk_bump_events": [],
            "sizing_trace": pd.DataFrame(),
            "sizing_trace_summary": _summarize_sizing_trace([]),
            "stage24_sizing_trace": pd.DataFrame(),
            "stage24_sizing_summary": _summarize_stage24_sizing_trace([]),
            "shadow_live_summary": _empty_shadow_live_summary(False),
            "research_infeasible_flags": [],
            "breakdown": RejectBreakdown().to_payload(),
        }

    eval_cfg = dict((cfg.get("evaluation", {}) or {}).get("stage23", {}))
    stage24_cfg = dict((cfg.get("evaluation", {}) or {}).get("stage24", {}))
    constraints_cfg = dict((cfg.get("evaluation", {}) or {}).get("constraints", {}))
    modes_cfg = dict((cfg.get("evaluation", {}) or {}).get("modes", {}))
    stage24_enabled = bool(stage24_cfg.get("enabled", False))
    order_builder = OrderBuilderConfig(**_pick_order_builder_cfg(eval_cfg.get("order_builder", {})))
    relax = ExecutionRelaxConfig(**_pick_execution_cfg(eval_cfg.get("execution", {})))
    sizing_cfg = SizingRepairConfig(**_pick_sizing_cfg(eval_cfg.get("sizing", {})))
    sizing_fix_enabled = bool(eval_cfg.get("sizing_fix_enabled", True))
    constraint_mode = str(constraints_cfg.get("mode", "live")).strip().lower()
    is_research_mode = constraint_mode == "research"
    research_mode_cfg = dict(modes_cfg.get("research", {}))
    live_mode_cfg = dict(modes_cfg.get("live", {}))
    use_exchange_rules_live = bool(live_mode_cfg.get("use_exchange_rules", True))
    enforce_margin_caps = bool(research_mode_cfg.get("enforce_margin_caps", False)) if is_research_mode else bool(use_exchange_rules_live)
    enforce_min_notional = bool(research_mode_cfg.get("enforce_min_notional", False)) if is_research_mode else bool(use_exchange_rules_live)
    enforce_size_step = bool(research_mode_cfg.get("enforce_size_step", False)) if is_research_mode else bool(use_exchange_rules_live)
    ignore_exchange_precision = bool(research_mode_cfg.get("ignore_exchange_precision", True)) if is_research_mode else False
    if ignore_exchange_precision:
        enforce_size_step = False
    live_constraint_cfg = dict(constraints_cfg.get("live", {}))
    research_constraint_cfg = dict(constraints_cfg.get("research", {}))
    active_constraint_cfg = research_constraint_cfg if is_research_mode else live_constraint_cfg
    min_notional_override = float(research_mode_cfg.get("min_notional_override", 1.0))
    active_min_notional = float(active_constraint_cfg.get("min_trade_notional", order_builder.min_trade_notional))
    min_trade_notional = float(active_min_notional if enforce_min_notional else max(0.0, min_notional_override))
    qty_step = float(active_constraint_cfg.get("qty_step", order_builder.qty_step)) if enforce_size_step else 0.0
    min_trade_qty = float(active_constraint_cfg.get("min_trade_qty", order_builder.min_trade_qty)) if enforce_size_step else 0.0
    live_min_notional = float(live_constraint_cfg.get("min_trade_notional", order_builder.min_trade_notional))
    live_qty_step = float(live_constraint_cfg.get("qty_step", order_builder.qty_step))
    live_min_trade_qty = float(live_constraint_cfg.get("min_trade_qty", order_builder.min_trade_qty))
    allow_size_bump_to_min_notional = bool(order_builder.allow_size_bump_to_min_notional)
    max_notional_pct_cfg = float(((stage24_cfg.get("order_constraints", {}) or {}).get("max_notional_pct_of_equity", 1.0)))
    risk_auto_bump_cfg = dict((cfg.get("execution", {}) or {}).get("risk_auto_bump", {}))
    risk_auto_bump_enabled = bool(risk_auto_bump_cfg.get("enabled", True))
    risk_auto_bump_cap = float(risk_auto_bump_cfg.get("max_risk_cap", 0.20))

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
    risk_bump_events: list[dict[str, Any]] = []
    sizing_trace_rows: list[dict[str, Any]] = []
    stage24_trace_rows: list[dict[str, Any]] = []
    shadow_live_summary = _empty_shadow_live_summary(is_research_mode)
    research_infeasible_flags: list[dict[str, Any]] = []

    tp_atr = float(((cfg.get("evaluation", {}) or {}).get("stage10", {}) or {}).get("evaluation", {}).get("take_profit_atr_multiple", 3.0))
    initial_equity = float(((stage24_cfg.get("simulation", {}) or {}).get("initial_equities", [10000.0])[0]) if stage24_enabled else 10000.0)
    equity_state = max(1e-9, float(initial_equity))
    peak_equity = float(equity_state)
    losing_streak = 0
    cost_rt_pct = float(cost_rt_pct_from_config(cfg))

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
        max_notional = 0.0
        margin_required_now = 0.0
        margin_limit_now = 0.0
        risk_used_for_reject = 0.0
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
                    min_notional=float(min_trade_notional),
                    min_trade_qty=float(min_trade_qty),
                    qty_step=float(qty_step),
                    bumped_to_min_notional=bool(bumped_to_min_notional),
                    rounded_size_before=rounded_size_before,
                    rounded_size_after=rounded_size_after,
                    rounding_mode_used=rounding_mode_used,
                    final_notional=final_notional,
                    max_allowed_notional=max_notional,
                    margin_required=margin_required_now,
                    margin_limit=margin_limit_now,
                    ceil_rescue_applied=ceil_rescue_applied,
                    cap_binding=cap_binding,
                    decision=decision,
                    reject_reason=reject_reason,
                    reject_details=reject_details,
                ).to_payload()
            )

        def reject(reason: str, details: str) -> None:
            feasibility_payload: dict[str, Any] | None = None
            normalized_reason = normalize_reject_reason(reason)
            if normalized_reason in {"SIZE_TOO_SMALL", "POLICY_CAP_HIT", "MARGIN_FAIL"}:
                stop_dist_for_reject = abs(float(close_px - stop_price) / max(close_px, 1e-12)) if close_px > 0 else 0.0
                feasibility_payload = explain_reject(
                    {
                        "equity": float(equity_state if stage24_enabled else initial_equity),
                        "risk_pct_used": float(risk_used_for_reject),
                        "min_notional": float(min_trade_notional),
                        "stop_dist_pct": float(stop_dist_for_reject),
                        "cost_rt_pct": float(cost_rt_pct),
                        "max_notional_pct": float(max_notional_pct_cfg),
                    }
                )
            _reject(
                breakdown=breakdown,
                reject_events=reject_events,
                timestamp=timestamp,
                symbol=symbol,
                side=side_label,
                reason=reason,
                details=details,
                feasibility=feasibility_payload,
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

        stop_distance_pct = float(min_stop_px / max(close_px, 1e-12))
        if stage24_enabled:
            stage24_sizing = dict(stage24_cfg.get("sizing", {}))
            stage24_constraints = dict(stage24_cfg.get("order_constraints", {}))
            stage24_constraints["min_trade_notional"] = float(min_trade_notional)
            stage24_constraints["allow_size_bump_to_min_notional"] = bool(
                stage24_constraints.get("allow_size_bump_to_min_notional", allow_size_bump_to_min_notional)
            )
            dd_now = float(max(0.0, (peak_equity - equity_state) / max(peak_equity, 1e-12)))
            mode = str(stage24_sizing.get("mode", "risk_pct")).strip().lower()
            risk_used = 0.0
            risk_parts = {"base": 0.0, "dd_mult": 1.0, "streak_mult": 1.0, "used": 0.0}
            if mode == "risk_pct":
                risk_used, risk_parts = compute_risk_pct(
                    equity=float(equity_state),
                    dd=float(dd_now),
                    losing_streak=int(losing_streak),
                    cfg=cfg,
                )
                notional_24, status_24, reason_24, details_24 = compute_notional_risk_pct(
                    equity=float(equity_state),
                    risk_pct_used=float(risk_used),
                    stop_distance_pct=float(stop_distance_pct),
                    cost_rt_pct=float(cost_rt_pct),
                    constraints_cfg=stage24_constraints,
                )
            else:
                alloc_pct = float(stage24_sizing.get("alloc_pct", 0.25))
                notional_24, status_24, reason_24, details_24 = compute_notional_alloc_pct(
                    equity=float(equity_state),
                    alloc_pct=float(alloc_pct),
                    constraints_cfg=stage24_constraints,
                )
                risk_parts["used"] = float(alloc_pct)
                risk_used = float(alloc_pct)
            risk_used_for_reject = float(risk_used)
            stage24_reason = str(reason_24 or "")
            if stage24_reason and not is_known_reject_reason(stage24_reason):
                stage24_reason = "UNKNOWN"
            risk_bump_applied = False
            if (
                str(status_24) != "VALID"
                and not is_research_mode
                and mode == "risk_pct"
                and bool(risk_auto_bump_enabled)
                and stage24_reason == "SIZE_TOO_SMALL"
            ):
                min_risk_required = float(
                    calculate_min_risk_pct(
                        equity=float(equity_state),
                        stop_distance_pct=float(stop_distance_pct),
                        min_notional=float(min_trade_notional),
                        fee_roundtrip_pct=float(cost_rt_pct),
                        size_step=float(qty_step),
                    )
                )
                if np.isfinite(min_risk_required) and min_risk_required <= float(risk_auto_bump_cap):
                    prev_risk = float(risk_used)
                    risk_used = float(min_risk_required)
                    risk_parts["used"] = float(risk_used)
                    notional_24, status_24, reason_24, details_24 = compute_notional_risk_pct(
                        equity=float(equity_state),
                        risk_pct_used=float(risk_used),
                        stop_distance_pct=float(stop_distance_pct),
                        cost_rt_pct=float(cost_rt_pct),
                        constraints_cfg=stage24_constraints,
                    )
                    stage24_reason = str(reason_24 or "")
                    if stage24_reason and not is_known_reject_reason(stage24_reason):
                        stage24_reason = "UNKNOWN"
                    risk_bump_applied = str(status_24) == "VALID"
                    risk_bump_events.append(
                        {
                            "timestamp": timestamp,
                            "symbol": symbol,
                            "side": side_label,
                            "risk_before": float(prev_risk),
                            "risk_after": float(risk_used),
                            "min_risk_required": float(min_risk_required),
                            "max_risk_cap": float(risk_auto_bump_cap),
                            "status_after_bump": str(status_24),
                            "reason_after_bump": str(stage24_reason),
                        }
                    )
                else:
                    stage24_reason = "EXECUTION_INFEASIBLE_CAP"
                    status_24 = "INVALID"
                    details_24 = {
                        "min_risk_required": float(min_risk_required),
                        "max_risk_cap": float(risk_auto_bump_cap),
                    }
            stage24_trace_rows.append(
                {
                    "ts": timestamp,
                    "symbol": symbol,
                    "mode": mode,
                    "equity": float(equity_state),
                    "dd": float(dd_now),
                    "losing_streak": int(losing_streak),
                    "risk_base": float(risk_parts.get("base", 0.0)),
                    "dd_mult": float(risk_parts.get("dd_mult", 1.0)),
                    "streak_mult": float(risk_parts.get("streak_mult", 1.0)),
                    "risk_used": float(risk_used),
                    "risk_bump_applied": bool(risk_bump_applied),
                    "stop_dist_pct": float(stop_distance_pct),
                    "cost_rt_pct": float(cost_rt_pct),
                    "notional": float(notional_24),
                    "status": str(status_24),
                    "reason": stage24_reason,
                    "details": str(details_24),
                }
            )
            if str(status_24) != "VALID":
                reject(stage24_reason or "UNKNOWN", f"stage24_invalid:{details_24}")
                continue
            notional = float(notional_24)
            if bool((details_24 or {}).get("bumped_to_min_notional", False)):
                bumped_to_min_notional = True
        else:
            if notional < float(min_trade_notional):
                if bool(allow_size_bump_to_min_notional):
                    notional = float(min_trade_notional)
                    bumped_to_min_notional = True
                else:
                    reject("SIZE_TOO_SMALL", f"notional={notional:.6f}<min={float(min_trade_notional):.6f}")
                    continue

        if not stage24_enabled:
            # Keep legacy Stage-23 behavior for non-Stage24 paths.
            if enforce_margin_caps:
                max_notional = float(max(0.0, float(max_gross_exposure) * close_px * max(1.0, leverage)))
            else:
                max_notional = float(max(0.0, float(notional) * 1000.0))
            if max_notional <= 0.0:
                cap_binding = "margin"
                reject("MARGIN_FAIL", "non_positive_max_notional")
                continue

            if enforce_margin_caps and notional > max_notional:
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
        else:
            equity_for_checks = float(equity_state)
            policy_caps = PolicyCaps(
                max_notional_pct_of_equity=float((stage24_cfg.get("order_constraints", {}) or {}).get("max_notional_pct_of_equity", 1.0)),
                max_gross_exposure_mult=float(max(1e-9, float(max_gross_exposure))),
                absolute_max_notional=0.0,
                margin_alloc_limit=float(max(1e-9, float((cfg.get("risk", {}) or {}).get("margin_alloc_limit", 1.0)))),
            )

            cap_details: dict[str, Any] = {}
            margin_details: dict[str, Any] = {}
            reduction_steps = 0
            while True:
                capped_notional, cap_reason, cap_details = apply_exposure_caps(
                    desired_notional=float(notional),
                    policy_caps=policy_caps,
                    current_exposure=0.0,
                    equity=float(equity_for_checks),
                )
                max_notional = float(cap_details.get("max_allowed_notional", 0.0))
                if not enforce_margin_caps:
                    capped_notional = float(max(0.0, notional))
                    max_notional = float(max(max_notional, capped_notional * 1000.0))
                    cap_reason = ""
                if cap_reason and capped_notional <= 0.0:
                    cap_binding = "policy_cap"
                    reject("POLICY_CAP_HIT", f"max_notional={max_notional:.6f}")
                    break

                margin_required_now = float(
                    compute_margin_required(
                        notional=float(capped_notional),
                        leverage=float(max(leverage, 1e-12)),
                        fees_estimate=float(cost_rt_pct),
                        buffer=float((cfg.get("risk", {}) or {}).get("margin_buffer_pct", 0.0)),
                    )
                )
                feasible, feasible_reason, margin_details = is_trade_feasible(
                    equity=float(equity_for_checks),
                    capped_notional=float(capped_notional),
                    leverage=float(max(leverage, 1e-12)),
                    margin_required=float(margin_required_now),
                    policy_caps=policy_caps,
                )
                margin_limit_now = float(margin_details.get("margin_limit", 0.0))
                if feasible:
                    notional = float(capped_notional)
                    break

                if not enforce_margin_caps:
                    notional = float(max(0.0, capped_notional))
                    break

                if not bool(relax.allow_size_reduction_on_margin_fail) or reduction_steps >= int(relax.max_size_reduction_steps):
                    cap_binding = "margin" if feasible_reason == "MARGIN_FAIL" else "policy_cap"
                    reject(
                        feasible_reason or "MARGIN_FAIL",
                        f"margin_required={margin_required_now:.6f},margin_limit={margin_limit_now:.6f},max_notional={max_notional:.6f}",
                    )
                    break

                prev = float(notional)
                notional = float(notional * 0.7)
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
                        "margin_required": float(margin_required_now),
                        "margin_limit": float(margin_limit_now),
                    }
                )
                if notional <= 0.0:
                    cap_binding = "margin"
                    reject("MARGIN_FAIL", "size_reduced_to_zero")
                    break
            if reject_events and str(reject_events[-1].get("timestamp", "")) == timestamp and str(reject_events[-1].get("symbol", "")) == symbol:
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
            notional = max(float(min_trade_notional), notional * max(0.25, min(1.0, ratio)))
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
        if enforce_min_notional and filled_notional < float(min_trade_notional):
            if bool(allow_size_bump_to_min_notional) and float(min_trade_notional) <= max_notional + 1e-12:
                filled_notional = float(min_trade_notional)
                bumped_to_min_notional = True
            else:
                reject("SIZE_TOO_SMALL", f"filled_notional={filled_notional:.6f}")
                continue

        capped_size = float(min(filled_notional, max_notional) / max(close_px, 1e-12))
        rounded_size_before = float(capped_size)
        if enforce_size_step and float(min_trade_qty) > 0.0 and rounded_size_before < float(min_trade_qty):
            rounded_size_before = float(min_trade_qty)

        rounding_mode_used = str(sizing_cfg.qty_rounding_default)
        if bumped_to_min_notional:
            rounding_mode_used = str(sizing_cfg.qty_rounding_on_min_notional_bump)

        rounded_size_after = round_qty_to_step(rounded_size_before, float(qty_step), rounding_mode_used)

        if rounded_size_after <= 0.0 and rounded_size_before > 0.0:
            if bool(sizing_fix_enabled) and bool(sizing_cfg.allow_single_step_ceil_rescue):
                rescue_qty = round_qty_to_step(rounded_size_before, float(qty_step), "ceil")
                rescue_notional = float(rescue_qty * close_px)
                if float(qty_step) > 0.0:
                    overage_steps = float(max(0.0, (rescue_qty - rounded_size_before) / float(qty_step)))
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
                reject(reason, f"qty_before_round={rounded_size_before:.12f},step={float(qty_step):.12f}")
                continue

        if enforce_size_step and float(min_trade_qty) > 0.0 and rounded_size_after < float(min_trade_qty) - 1e-12:
            reject("SIZE_TOO_SMALL", f"rounded_qty={rounded_size_after:.12f}<min_qty={float(min_trade_qty):.12f}")
            continue

        final_notional = float(rounded_size_after * close_px)
        if enforce_min_notional and final_notional < float(min_trade_notional) - 1e-12:
            if bool(allow_size_bump_to_min_notional):
                bump_qty = round_qty_to_step(
                    float(min_trade_notional) / max(close_px, 1e-12),
                    float(qty_step),
                    str(sizing_cfg.qty_rounding_on_min_notional_bump),
                )
                bump_notional = float(bump_qty * close_px)
                if bump_qty > 0.0 and bump_notional <= max_notional + 1e-12 and bump_notional >= float(min_trade_notional) - 1e-12:
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
                reject("SIZE_TOO_SMALL", f"final_notional={final_notional:.6f}<min_notional={float(min_trade_notional):.6f}")
                continue

        if enforce_margin_caps and final_notional > max_notional + 1e-12:
            cap_binding = "policy_cap"
            reject("POLICY_CAP_HIT", f"final_notional={final_notional:.6f}>max_notional={max_notional:.6f}")
            continue

        shadow_reason = ""
        if is_research_mode:
            shadow_live_summary["research_accepted_count"] = int(shadow_live_summary.get("research_accepted_count", 0)) + 1
            shadow_reason = _shadow_live_reject_reason(
                final_notional=float(final_notional),
                final_qty=float(rounded_size_after),
                live_min_notional=float(live_min_notional),
                live_min_trade_qty=float(live_min_trade_qty),
                live_qty_step=float(live_qty_step),
            )
            if shadow_reason:
                shadow_live_summary["research_accepted_but_live_rejected_count"] = int(
                    shadow_live_summary.get("research_accepted_but_live_rejected_count", 0)
                ) + 1
                _register_shadow_live_reject(shadow_live_summary, shadow_reason)
                research_infeasible_flags.append(
                    {
                        "timestamp": timestamp,
                        "symbol": symbol,
                        "side": side_label,
                        "research_infeasible_live": True,
                        "reason": str(shadow_reason),
                        "final_notional": float(final_notional),
                        "final_qty": float(rounded_size_after),
                        "live_min_notional": float(live_min_notional),
                        "live_min_trade_qty": float(live_min_trade_qty),
                        "live_qty_step": float(live_qty_step),
                    }
                )
            else:
                shadow_live_summary["live_pass_count"] = int(shadow_live_summary.get("live_pass_count", 0)) + 1

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
                "research_infeasible_live": bool(is_research_mode and str(shadow_reason) != ""),
                "fill_ratio": float(fill_ratio),
                "slippage_bps": float(slippage_bps),
                "spread_bps": float(spread_bps),
                "adjustment_events_count": int(
                    sum(1 for event in adjustment_events if event.get("timestamp", "") == timestamp and event.get("symbol") == symbol)
                ),
            }
        )
        if stage24_enabled:
            next_idx = min(idx + 1, len(frame) - 1)
            next_close = float(close[next_idx]) if np.isfinite(close[next_idx]) else close_px
            trade_ret = (next_close - close_px) / max(close_px, 1e-12) * float(direction)
            pnl_proxy = float(final_notional * trade_ret)
            equity_state = float(max(1e-9, equity_state + pnl_proxy))
            peak_equity = float(max(peak_equity, equity_state))
            if pnl_proxy < 0:
                losing_streak += 1
            elif pnl_proxy > 0:
                losing_streak = 0

    research_accepted = int(shadow_live_summary.get("research_accepted_count", 0))
    if research_accepted > 0:
        live_rejected = int(shadow_live_summary.get("research_accepted_but_live_rejected_count", 0))
        shadow_live_summary["live_reject_rate"] = float(live_rejected / max(research_accepted, 1))

    orders_df = pd.DataFrame(order_rows)
    sizing_trace_df = pd.DataFrame(sizing_trace_rows)
    stage24_trace_df = pd.DataFrame(stage24_trace_rows)
    return {
        "accepted_signal": pd.Series(accepted, index=frame.index, dtype=int),
        "orders_df": orders_df,
        "reject_events": reject_events,
        "adjustment_events": adjustment_events,
        "risk_bump_events": risk_bump_events,
        "sizing_trace": sizing_trace_df,
        "sizing_trace_summary": _summarize_sizing_trace(sizing_trace_rows),
        "stage24_sizing_trace": stage24_trace_df,
        "stage24_sizing_summary": _summarize_stage24_sizing_trace(stage24_trace_rows),
        "shadow_live_summary": shadow_live_summary,
        "research_infeasible_flags": research_infeasible_flags,
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
    feasibility: dict[str, Any] | None = None,
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
            "minimum_required_risk_pct": float(feasibility.get("minimum_required_risk_pct", 0.0)) if isinstance(feasibility, dict) else 0.0,
            "minimum_required_equity": float(feasibility.get("minimum_required_equity", 0.0)) if isinstance(feasibility, dict) else 0.0,
            "min_risk_required": float(feasibility.get("min_risk_required", 0.0)) if isinstance(feasibility, dict) else 0.0,
            "min_equity_required": float(feasibility.get("min_equity_required", 0.0)) if isinstance(feasibility, dict) else 0.0,
            "stop_distance": float(feasibility.get("stop_distance", 0.0)) if isinstance(feasibility, dict) else 0.0,
            "fee_rt_pct": float(feasibility.get("fee_rt_pct", 0.0)) if isinstance(feasibility, dict) else 0.0,
            "feasibility_risk_pct_used": float(feasibility.get("risk_pct_used", 0.0)) if isinstance(feasibility, dict) else 0.0,
            "feasibility_stop_dist_pct": float(feasibility.get("stop_dist_pct", 0.0)) if isinstance(feasibility, dict) else 0.0,
            "feasibility_cost_rt_pct": float(feasibility.get("cost_rt_pct", 0.0)) if isinstance(feasibility, dict) else 0.0,
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
            "margin_required_min": 0.0,
            "margin_required_median": 0.0,
            "margin_required_max": 0.0,
            "margin_limit_min": 0.0,
            "margin_limit_median": 0.0,
            "margin_limit_max": 0.0,
            "zero_size_count": 0,
            "rescued_by_ceil_count": 0,
            "bumped_to_min_notional_count": 0,
            "cap_binding_reject_count": 0,
            "reject_reason_counts": {},
        }
    frame = pd.DataFrame(rows)
    raw_size = pd.to_numeric(frame.get("raw_size", 0.0), errors="coerce").fillna(0.0)
    rounded_after = pd.to_numeric(frame.get("rounded_size_after", 0.0), errors="coerce").fillna(0.0)
    margin_required = pd.to_numeric(frame.get("margin_required", 0.0), errors="coerce").fillna(0.0)
    margin_limit = pd.to_numeric(frame.get("margin_limit", 0.0), errors="coerce").fillna(0.0)
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
        "margin_required_min": float(margin_required.min()),
        "margin_required_median": float(margin_required.median()),
        "margin_required_max": float(margin_required.max()),
        "margin_limit_min": float(margin_limit.min()),
        "margin_limit_median": float(margin_limit.median()),
        "margin_limit_max": float(margin_limit.max()),
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


def _summarize_stage24_sizing_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "valid_count": 0,
            "invalid_count": 0,
            "top_invalid_reasons": {},
            "notional_min": 0.0,
            "notional_median": 0.0,
            "notional_max": 0.0,
            "risk_used_min": 0.0,
            "risk_used_median": 0.0,
            "risk_used_max": 0.0,
        }
    frame = pd.DataFrame(rows)
    status = frame.get("status", pd.Series(dtype=str)).astype(str)
    reason = frame.get("reason", pd.Series(dtype=str)).astype(str).replace("", "VALID")
    notional = pd.to_numeric(frame.get("notional", 0.0), errors="coerce").fillna(0.0)
    risk_used = pd.to_numeric(frame.get("risk_used", 0.0), errors="coerce").fillna(0.0)
    invalid = frame.loc[status != "VALID"]
    invalid_counts = (
        invalid.get("reason", pd.Series(dtype=str))
        .astype(str)
        .replace("", "UNKNOWN")
        .value_counts()
        .head(5)
        .to_dict()
    )
    return {
        "valid_count": int((status == "VALID").sum()),
        "invalid_count": int((status != "VALID").sum()),
        "top_invalid_reasons": {str(k): int(v) for k, v in invalid_counts.items()},
        "notional_min": float(notional.min()),
        "notional_median": float(notional.median()),
        "notional_max": float(notional.max()),
        "risk_used_min": float(risk_used.min()),
        "risk_used_median": float(risk_used.median()),
        "risk_used_max": float(risk_used.max()),
    }
