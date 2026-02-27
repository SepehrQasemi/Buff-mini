"""Signal-to-order allocation with Stage-4 execution policy and risk controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from buffmini.execution.policy import ExecutionMode, Signal, apply_execution_policy
from buffmini.execution.risk import PortfolioState, compute_position_size, enforce_exposure_caps, killswitch_update_and_decide


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
    leverage = float(chosen_leverage)
    weights = {str(key): float(value) for key, value in (method_weights or {}).items()}

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
        return []

    prepared_signals: list[Signal] = []
    for signal in signals:
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
        exposure_strength = abs(signed_strength) * float(base_size)
        if exposure_strength <= 0.0:
            continue
        prepared_signals.append(
            Signal(
                ts=signal.ts,
                symbol=signal.symbol,
                direction=1 if signed_strength > 0 else -1,
                strength=exposure_strength,
                stop_distance=signal.stop_distance,
                strategy_id=signal.strategy_id,
                timeframe=signal.timeframe,
            )
        )

    if not prepared_signals:
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
        return []

    orders: list[Order] = []
    for item in scaled:
        fraction = float(item["exposure_fraction"])
        if fraction == 0.0:
            continue
        direction = 1 if fraction > 0 else -1
        notes = [
            f"policy_mode={mode.value}",
            f"cap_scale={cap_scale:.6f}",
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
