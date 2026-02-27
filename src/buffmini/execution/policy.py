"""Execution policy layer for Stage-4 order intent shaping."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd


class ExecutionMode(str, Enum):
    """Supported execution policies."""

    NET = "net"
    HEDGE = "hedge"
    ISOLATED = "isolated"


@dataclass(frozen=True)
class Signal:
    """One strategy intent at a point in time."""

    ts: pd.Timestamp
    symbol: str
    direction: int
    strength: float = 1.0
    stop_distance: float | None = None
    strategy_id: str = "unknown"
    timeframe: str = "1h"


@dataclass(frozen=True)
class TargetExposure:
    """Policy-normalized exposure target."""

    ts: pd.Timestamp
    symbol: str
    net_exposure: float
    gross_exposure: float
    components: list[dict[str, Any]]


def apply_execution_policy(
    signals: list[Signal],
    mode: ExecutionMode,
    per_symbol_netting: bool,
) -> list[TargetExposure]:
    """Apply NET/HEDGE/ISOLATED policy to raw signals."""

    if not signals:
        return []

    grouped: dict[tuple[pd.Timestamp, str], list[Signal]] = {}
    for signal in signals:
        ts = _ensure_utc(signal.ts)
        key = (ts, str(signal.symbol))
        grouped.setdefault(key, []).append(signal)

    targets: list[TargetExposure] = []
    for (ts, symbol), bucket in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        prepared = []
        for signal in bucket:
            direction = 1 if int(signal.direction) >= 0 else -1
            strength = abs(float(signal.strength)) if signal.strength is not None else 1.0
            prepared.append(
                {
                    "strategy_id": str(signal.strategy_id),
                    "direction": direction,
                    "signed_exposure": direction * strength,
                    "strength": strength,
                    "stop_distance": signal.stop_distance,
                    "timeframe": signal.timeframe,
                }
            )

        if mode == ExecutionMode.NET:
            if per_symbol_netting:
                net_value = float(sum(float(item["signed_exposure"]) for item in prepared))
                gross_value = float(sum(abs(float(item["signed_exposure"])) for item in prepared))
                targets.append(
                    TargetExposure(
                        ts=ts,
                        symbol=symbol,
                        net_exposure=net_value,
                        gross_exposure=gross_value,
                        components=prepared,
                    )
                )
            else:
                for item in prepared:
                    signed = float(item["signed_exposure"])
                    targets.append(
                        TargetExposure(
                            ts=ts,
                            symbol=symbol,
                            net_exposure=signed,
                            gross_exposure=abs(signed),
                            components=[item],
                        )
                    )
        elif mode == ExecutionMode.HEDGE:
            total_net = float(sum(float(item["signed_exposure"]) for item in prepared))
            total_gross = float(sum(abs(float(item["signed_exposure"])) for item in prepared))
            for item in prepared:
                signed = float(item["signed_exposure"])
                targets.append(
                    TargetExposure(
                        ts=ts,
                        symbol=symbol,
                        net_exposure=signed,
                        gross_exposure=abs(signed),
                        components=[
                            {
                                **item,
                                "aggregate_symbol_net_exposure": total_net,
                                "aggregate_symbol_gross_exposure": total_gross,
                            }
                        ],
                    )
                )
        elif mode == ExecutionMode.ISOLATED:
            by_strategy: dict[str, list[dict[str, Any]]] = {}
            for item in prepared:
                by_strategy.setdefault(str(item["strategy_id"]), []).append(item)
            for strategy_id, items in sorted(by_strategy.items(), key=lambda kv: kv[0]):
                net_value = float(sum(float(item["signed_exposure"]) for item in items))
                gross_value = float(sum(abs(float(item["signed_exposure"])) for item in items))
                targets.append(
                    TargetExposure(
                        ts=ts,
                        symbol=symbol,
                        net_exposure=net_value,
                        gross_exposure=gross_value,
                        components=[{**item, "strategy_group": strategy_id} for item in items],
                    )
                )
        else:
            raise ValueError(f"Unsupported execution mode: {mode}")

    return targets


def _ensure_utc(value: pd.Timestamp | str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")
