"""Signal-to-order allocation with Stage-4 execution policy and risk controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from buffmini.execution.policy import ExecutionMode, Signal, apply_execution_policy
from buffmini.execution.risk import PortfolioState, compute_position_size, enforce_exposure_caps, killswitch_update_and_decide
from buffmini.risk.confidence_sizing import (
    candidate_confidence,
    confidence_multiplier,
    extract_candidate_metric,
    renormalize_signed_weights,
)


@dataclass(frozen=True)
class Order:
    """Final order intent emitted by Stage-4 allocator."""

    ts: Any
    symbol: str
    direction: int
    notional_fraction_of_equity: float
    leverage: float
    strategy_id: str
    notes: list[str]


def generate_orders(
    signals: list[Signal],
    cfg: dict[str, Any],
    state: PortfolioState,
    chosen_leverage: float,
    method_weights: dict[str, float] | None,
) -> list[Order]:
    """Convert raw signals into policy/risk-compliant orders."""

    if not signals:
        return []

    execution_cfg = cfg["execution"]
    risk_cfg = cfg["risk"]
    mode = ExecutionMode(str(execution_cfg["mode"]))
    leverage = float(cfg.get("_runtime_leverage", chosen_leverage))
    weights = {str(key): float(value) for key, value in (method_weights or {}).items()}
    stage6_cfg = (cfg.get("evaluation", {}) or {}).get("stage6", {})
    stage6_enabled = bool(stage6_cfg.get("enabled", False))
    confidence_cfg = dict(stage6_cfg.get("confidence_sizing", {})) if isinstance(stage6_cfg, dict) else {}
    candidate_metrics = cfg.get("_runtime_candidate_metrics", {}) if isinstance(cfg.get("_runtime_candidate_metrics", {}), dict) else {}
    confidence_scale = float(confidence_cfg.get("scale", 2.0))
    multiplier_min = float(confidence_cfg.get("multiplier_min", 0.5))
    multiplier_max = float(confidence_cfg.get("multiplier_max", 1.5))

    timestamp = signals[0].ts
    decision = killswitch_update_and_decide(
        state=state,
        pnl_change=float(cfg.get("_runtime_pnl_change", 0.0)),
        ts=timestamp,
        bar_index=int(cfg.get("_runtime_bar_index", 0)),
        cfg=risk_cfg,
    )
    cfg["_last_risk_decision"] = decision
    if not decision.allow_new_trades:
        cfg["_last_sizing_records"] = []
        return []

    prepared_signals: list[Signal] = []
    component_signed_weights: dict[str, float] = {}
    component_records: dict[str, dict[str, Any]] = {}
    cfg["_last_sizing_records"] = []
    for signal_idx, signal in enumerate(signals):
        strategy_weight = float(weights.get(str(signal.strategy_id), 1.0))
        signed_strength = float(signal.direction) * float(signal.strength) * strategy_weight
        if signed_strength == 0.0:
            continue
        try:
            base_size = compute_position_size(
                equity=float(state.equity),
                risk_cfg=risk_cfg,
                stop_distance_pct=signal.stop_distance,
            )
        except ValueError:
            continue
        base_signed_weight = float(signed_strength) * float(base_size)
        if base_signed_weight == 0.0:
            continue
        component_id = str(signal.strategy_id)
        confidence = 0.5
        multiplier = 1.0
        if stage6_enabled:
            candidate_meta = candidate_metrics.get(component_id, {})
            exp_lcb = extract_candidate_metric(candidate_meta, "exp_lcb_holdout", fallback=0.0)
            pf_adj = extract_candidate_metric(candidate_meta, "pf_adj_holdout", fallback=1.0)
            confidence = candidate_confidence(
                exp_lcb_holdout=exp_lcb,
                pf_adj_holdout=pf_adj,
                scale=confidence_scale,
            )
            multiplier = confidence_multiplier(
                confidence=confidence,
                lower=multiplier_min,
                upper=multiplier_max,
            )
        desired_signed_weight = float(base_signed_weight) * float(multiplier)
        if desired_signed_weight == 0.0:
            continue
        record_key = f"{component_id}:{signal.symbol}:{signal_idx}"
        component_signed_weights[record_key] = float(desired_signed_weight)
        component_records[record_key] = {
            "ts": signal.ts,
            "symbol": str(signal.symbol),
            "component_id": component_id,
            "confidence": float(confidence),
            "multiplier": float(multiplier),
            "base_weight": float(base_signed_weight),
            "stop_distance": signal.stop_distance,
            "timeframe": signal.timeframe,
        }

    if not component_signed_weights:
        cfg["_last_sizing_records"] = []
        return []

    max_component_abs = float(risk_cfg["max_gross_exposure"]) / float(leverage) if leverage > 0 else 0.0
    renormalized_signed, component_scale = renormalize_signed_weights(
        signed_weights=component_signed_weights,
        max_abs_sum=max_component_abs,
    )

    for record_key, signed_weight in sorted(renormalized_signed.items()):
        component = component_records[record_key]
        prepared_signals.append(
            Signal(
                ts=component["ts"],
                symbol=component["symbol"],
                direction=1 if float(signed_weight) > 0 else -1,
                strength=abs(float(signed_weight)),
                stop_distance=component["stop_distance"],
                strategy_id=component["component_id"],
                timeframe=component["timeframe"],
            )
        )

    if not prepared_signals:
        cfg["_last_sizing_records"] = []
        return []

    targets = apply_execution_policy(
        signals=prepared_signals,
        mode=mode,
        per_symbol_netting=bool(execution_cfg["per_symbol_netting"]),
    )
    desired: list[dict[str, Any]] = []
    for target in targets:
        strategy_id = "NETTED" if mode == ExecutionMode.NET else str(target.components[0].get("strategy_id", "unknown"))
        desired.append(
            {
                "ts": target.ts,
                "symbol": target.symbol,
                "exposure_fraction": float(target.net_exposure),
                "strategy_id": strategy_id,
                "components": target.components,
            }
        )

    scaled, cap_scale, cap_reasons = enforce_exposure_caps(
        desired_exposures=desired,
        leverage=leverage,
        risk_cfg=risk_cfg,
    )
    if not scaled:
        cfg["_last_sizing_records"] = []
        return []

    symbol_net: dict[str, float] = {}
    for item in scaled:
        symbol = str(item["symbol"])
        symbol_net[symbol] = symbol_net.get(symbol, 0.0) + float(item["exposure_fraction"])

    sizing_records: list[dict[str, Any]] = []
    for record_key, record in sorted(component_records.items()):
        renorm_weight = float(renormalized_signed.get(record_key, 0.0))
        applied_weight = float(renorm_weight) * float(cap_scale)
        symbol = str(record["symbol"])
        sizing_records.append(
            {
                "timestamp": str(record["ts"]),
                "component_id": str(record["component_id"]),
                "symbol": symbol,
                "confidence": float(record["confidence"]),
                "multiplier": float(record["multiplier"]),
                "component_renorm_scale": float(component_scale),
                "cap_scale": float(cap_scale),
                "base_weight": float(record["base_weight"]),
                "applied_weight": float(applied_weight),
                "final_net_exposure": float(symbol_net.get(symbol, 0.0) * leverage),
            }
        )
    cfg["_last_sizing_records"] = sizing_records
    cfg["_last_target_exposures"] = scaled

    orders: list[Order] = []
    for item in scaled:
        fraction = float(item["exposure_fraction"])
        if fraction == 0.0:
            continue
        direction = 1 if fraction > 0 else -1
        notes = [
            f"policy_mode={mode.value}",
            f"cap_scale={cap_scale:.6f}",
            f"component_scale={component_scale:.6f}",
        ]
        if cap_reasons:
            notes.append("cap_reasons=" + ",".join(sorted(set(cap_reasons))))
        components = item.get("components", [])
        if components:
            notes.append(f"components={len(components)}")
        orders.append(
            Order(
                ts=item["ts"],
                symbol=str(item["symbol"]),
                direction=direction,
                notional_fraction_of_equity=abs(fraction),
                leverage=leverage,
                strategy_id=str(item["strategy_id"]),
                notes=notes,
            )
        )

    if not orders:
        return []

    max_open_positions = int(risk_cfg["max_open_positions"])
    available_slots = max(0, max_open_positions - int(len(state.open_positions)))
    if available_slots <= 0:
        return []
    if len(orders) > available_slots:
        orders = sorted(
            orders,
            key=lambda order: (-float(order.notional_fraction_of_equity), str(order.symbol), str(order.strategy_id)),
        )[:available_slots]
    return orders
