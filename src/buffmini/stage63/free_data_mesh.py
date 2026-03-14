"""Stage-63 free data mesh planning utilities."""

from __future__ import annotations

from typing import Any

from buffmini.utils.hashing import stable_hash


def canonical_source_schema() -> dict[str, list[str]]:
    return {
        "ohlcv": ["timestamp", "symbol", "timeframe", "open", "high", "low", "close", "volume", "source"],
        "funding": ["timestamp", "symbol", "funding_rate", "source", "coverage_window", "validity_mask"],
        "oi": ["timestamp", "symbol", "open_interest", "source", "coverage_window", "validity_mask"],
        "long_short": ["timestamp", "symbol", "long_short_ratio", "source", "coverage_window", "validity_mask"],
        "taker_flow": ["timestamp", "symbol", "buy_sell_ratio", "buy_volume", "sell_volume", "source", "coverage_window", "validity_mask"],
    }


def build_free_data_mesh_plan(config: dict[str, Any]) -> dict[str, Any]:
    source_cfg = dict(config.get("data_sources", {}))
    enabled = [str(v).lower() for v in source_cfg.get("enabled", ["binance", "bybit", "deribit"])]
    default_endpoints = {
        "binance": ["ohlcv", "funding", "oi", "long_short", "taker_flow"],
        "bybit": ["ohlcv", "funding", "oi", "long_short"],
        "deribit": ["ohlcv", "funding", "oi"],
    }
    source_priority = [str(v).lower() for v in source_cfg.get("source_priority", enabled)]
    rate_limits = dict(source_cfg.get("source_rate_limits", {}))
    contracts = []
    for source in source_priority:
        if source not in enabled:
            continue
        contracts.append(
            {
                "source": source,
                "endpoints": list(default_endpoints.get(source, ["ohlcv"])),
                "rate_limit": rate_limits.get(source, {"mode": "respect_provider", "max_requests_per_min": None}),
                "source_cost": 0.0,
            }
        )
    payload = {
        "enabled_sources": enabled,
        "source_priority": source_priority,
        "source_contracts": contracts,
        "schema": canonical_source_schema(),
    }
    payload["summary_hash"] = stable_hash(payload, length=16)
    return payload

