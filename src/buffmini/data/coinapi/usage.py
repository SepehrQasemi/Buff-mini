"""Usage ledger and summaries for CoinAPI requests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _normalize_key(value: str) -> str:
    return str(value).strip().lower().replace("-", "_")


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_coinapi_header_signals(headers: dict[str, Any] | None) -> dict[str, Any]:
    """Extract possible quota/credit signals from HTTP headers."""

    hdrs = headers or {}
    normalized: dict[str, Any] = {_normalize_key(k): v for k, v in hdrs.items()}
    signals: dict[str, Any] = {}
    for key in (
        "x_ratelimit_limit",
        "x_ratelimit_remaining",
        "x_ratelimit_reset",
        "x_rate_limit_limit",
        "x_rate_limit_remaining",
        "x_rate_limit_reset",
        "x_quota_used",
        "x_quota_remaining",
        "x_credits_used",
        "x_credits_remaining",
    ):
        if key in normalized:
            signals[key] = normalized[key]
    return signals


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class CoinAPIUsageLedger:
    """Append-only JSONL ledger for CoinAPI usage records."""

    path: Path

    def append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(record)
        payload.setdefault("ts_utc", _utc_now_iso())
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True, allow_nan=False))
            handle.write("\n")

    def load_records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if not text:
                continue
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                rows.append(parsed)
        return rows


def build_usage_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate basic usage and quota totals from ledger rows."""

    total_requests = int(len(records))
    total_success = int(sum(1 for row in records if 200 <= int(row.get("status_code", 0) or 0) < 300))
    total_fail = int(total_requests - total_success)

    endpoint_stats: dict[str, dict[str, Any]] = {}
    symbol_stats: dict[str, dict[str, Any]] = {}
    quota_signals: dict[str, Any] = {}
    total_bytes = 0
    total_retries = 0
    status_code_counts: dict[str, int] = {}
    retry_count_by_endpoint: dict[str, int] = {}
    rate_limit_sleep_ms_total = 0
    time_start_min: str | None = None
    time_end_max: str | None = None

    for row in records:
        endpoint = str(row.get("endpoint_name", "") or "unknown")
        symbol = str(row.get("symbol", "") or "unknown")
        status = int(row.get("status_code", 0) or 0)
        bytes_count = int(row.get("response_bytes", 0) or 0)
        retries = int(row.get("retry_count", 0) or 0)
        backoff_sleep_ms = int(row.get("backoff_sleep_ms", 0) or 0)
        status_key = str(status)
        status_code_counts[status_key] = int(status_code_counts.get(status_key, 0) + 1)
        total_bytes += bytes_count
        total_retries += retries
        rate_limit_sleep_ms_total += backoff_sleep_ms

        ep = endpoint_stats.setdefault(endpoint, {"requests": 0, "success": 0, "fail": 0, "bytes": 0})
        ep["requests"] = int(ep["requests"] + 1)
        ep["bytes"] = int(ep["bytes"] + bytes_count)
        if 200 <= status < 300:
            ep["success"] = int(ep["success"] + 1)
        else:
            ep["fail"] = int(ep["fail"] + 1)

        sym = symbol_stats.setdefault(symbol, {"requests": 0, "success": 0, "fail": 0, "bytes": 0})
        sym["requests"] = int(sym["requests"] + 1)
        sym["bytes"] = int(sym["bytes"] + bytes_count)
        if 200 <= status < 300:
            sym["success"] = int(sym["success"] + 1)
        else:
            sym["fail"] = int(sym["fail"] + 1)
        retry_count_by_endpoint[endpoint] = int(retry_count_by_endpoint.get(endpoint, 0) + retries)

        row_start = str(row.get("time_start", "") or "")
        row_end = str(row.get("time_end", "") or "")
        if row_start:
            if time_start_min is None or row_start < time_start_min:
                time_start_min = row_start
        if row_end:
            if time_end_max is None or row_end > time_end_max:
                time_end_max = row_end

        header_signals = row.get("header_signals", {})
        if isinstance(header_signals, dict):
            for key, value in header_signals.items():
                quota_signals[key] = value

    credits_used = _to_float(quota_signals.get("x_credits_used"))
    credits_remaining = _to_float(quota_signals.get("x_credits_remaining"))
    quota_used = _to_float(quota_signals.get("x_quota_used"))
    quota_remaining = _to_float(quota_signals.get("x_quota_remaining"))

    return {
        "total_requests": total_requests,
        "total_success": total_success,
        "total_fail": total_fail,
        "total_bytes": int(total_bytes),
        "total_retries": int(total_retries),
        "status_code_counts": status_code_counts,
        "per_endpoint": endpoint_stats,
        "per_symbol": symbol_stats,
        "endpoints_hit": sorted(endpoint_stats.keys()),
        "retry_count_by_endpoint": retry_count_by_endpoint,
        "time_start_min": time_start_min,
        "time_end_max": time_end_max,
        "rate_limit_sleep_ms_total": int(rate_limit_sleep_ms_total),
        "quota_signals": quota_signals,
        "credits_used": credits_used,
        "credits_remaining": credits_remaining,
        "quota_used": quota_used,
        "quota_remaining": quota_remaining,
        "credits_estimation_mode": "explicit_header" if credits_used is not None else "UNKNOWN",
    }
