"""Deterministic planning utilities for CoinAPI incremental backfills."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pandas as pd

from buffmini.utils.hashing import stable_hash


COINAPI_SYMBOL_MAP: dict[str, str] = {
    "BTC/USDT": "BINANCE_PERP_BTC_USDT",
    "ETH/USDT": "BINANCE_PERP_ETH_USDT",
    "BINANCE_PERP_BTC_USDT": "BINANCE_PERP_BTC_USDT",
    "BINANCE_PERP_ETH_USDT": "BINANCE_PERP_ETH_USDT",
}


ENDPOINT_CONFIG: dict[str, dict[str, Any]] = {
    "funding_rates": {"path": "/v1/exchangerate/futures/funding_rate/history"},
    "open_interest": {"path": "/v1/exchangerate/futures/open_interest/history"},
    "liquidations": {"path": "/v1/exchangerate/futures/liquidations/history"},
}


@dataclass(frozen=True, slots=True)
class PlanSlice:
    index: int
    endpoint: str
    endpoint_path: str
    symbol: str
    symbol_id: str
    start_ts: str
    end_ts: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": int(self.index),
            "endpoint": str(self.endpoint),
            "endpoint_path": str(self.endpoint_path),
            "symbol": str(self.symbol),
            "symbol_id": str(self.symbol_id),
            "start_ts": str(self.start_ts),
            "end_ts": str(self.end_ts),
        }


def map_symbol_to_coinapi(symbol: str) -> str:
    key = str(symbol).strip()
    return COINAPI_SYMBOL_MAP.get(key, key)


def _coerce_ts(value: str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.to_datetime(value, utc=True)
    return pd.Timestamp(ts)


def _slice_windows(start: pd.Timestamp, end: pd.Timestamp, increment_days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    inc = max(1, int(increment_days))
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor <= end:
        right = min(end, cursor + timedelta(days=inc) - timedelta(seconds=1))
        windows.append((cursor, right))
        cursor = right + timedelta(seconds=1)
    return windows


def build_backfill_plan(
    *,
    symbols: list[str],
    endpoints: list[str],
    start_ts: str | pd.Timestamp,
    end_ts: str | pd.Timestamp,
    increment_days: int = 7,
    max_requests: int = 2000,
) -> dict[str, Any]:
    start = _coerce_ts(start_ts)
    end = _coerce_ts(end_ts)
    if end < start:
        raise ValueError("end_ts must be >= start_ts")
    endpoint_list = [str(v) for v in endpoints]
    unknown = [name for name in endpoint_list if name not in ENDPOINT_CONFIG]
    if unknown:
        raise ValueError(f"unknown endpoint(s): {unknown}")
    symbol_list = [str(v) for v in symbols]
    windows = _slice_windows(start, end, increment_days=max(1, int(increment_days)))
    items: list[PlanSlice] = []
    idx = 0
    for endpoint in endpoint_list:
        ep_path = str(ENDPOINT_CONFIG[endpoint]["path"])
        for symbol in symbol_list:
            symbol_id = map_symbol_to_coinapi(symbol)
            for left, right in windows:
                items.append(
                    PlanSlice(
                        index=idx,
                        endpoint=endpoint,
                        endpoint_path=ep_path,
                        symbol=symbol,
                        symbol_id=symbol_id,
                        start_ts=left.isoformat(),
                        end_ts=right.isoformat(),
                    )
                )
                idx += 1

    planned_count = int(len(items))
    max_req = max(1, int(max_requests))
    truncated = planned_count > max_req
    selected = items[:max_req]
    plan_seed = {
        "symbols": symbol_list,
        "endpoints": endpoint_list,
        "start_ts": start.isoformat(),
        "end_ts": end.isoformat(),
        "increment_days": int(increment_days),
        "planned_count": planned_count,
        "selected_count": int(len(selected)),
    }
    plan_id = stable_hash(plan_seed, length=12)
    return {
        "plan_id": str(plan_id),
        "start_ts": start.isoformat(),
        "end_ts": end.isoformat(),
        "increment_days": int(increment_days),
        "planned_count": int(planned_count),
        "selected_count": int(len(selected)),
        "truncated": bool(truncated),
        "max_requests": int(max_req),
        "items": [item.to_dict() for item in selected],
    }


def estimate_additional_days_required(*, coverage_years: float, required_years: float) -> float:
    if float(coverage_years) >= float(required_years):
        return 0.0
    missing_years = float(required_years) - float(coverage_years)
    return float(max(0.0, missing_years * 365.25))

