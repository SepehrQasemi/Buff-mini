"""Incremental CoinAPI downloader for funding/open-interest/liquidations extras."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.data.coinapi.client import CoinAPIClient, CoinAPIRequestError
from buffmini.data.coinapi.endpoints.funding_rates import normalize_funding_rates, write_funding_canonical
from buffmini.data.coinapi.endpoints.liquidations import normalize_liquidations, write_liquidations_canonical
from buffmini.data.coinapi.endpoints.open_interest import normalize_open_interest, write_open_interest_canonical
from buffmini.data.coinapi.planner import build_backfill_plan
from buffmini.data.coinapi.usage import CoinAPIUsageLedger, build_usage_summary
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


ADAPTERS = {
    "funding_rates": (normalize_funding_rates, write_funding_canonical),
    "open_interest": (normalize_open_interest, write_open_interest_canonical),
    "liquidations": (normalize_liquidations, write_liquidations_canonical),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update CoinAPI enriched extras with budget-aware planning")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--endpoints", type=str, default="")
    parser.add_argument("--start", type=str, default="")
    parser.add_argument("--end", type=str, default="")
    parser.add_argument("--last-days", type=int, default=0)
    parser.add_argument("--increment-days", type=int, default=7)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    return parser.parse_args()


def _split_csv(value: str) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def _default_coinapi_config(config: dict[str, Any]) -> dict[str, Any]:
    default = {
        "enabled": False,
        "symbols": ["BINANCE_PERP_BTC_USDT", "BINANCE_PERP_ETH_USDT"],
        "priority_endpoints": ["funding_rates", "open_interest", "liquidations"],
        "max_days_per_run": 30,
        "max_total_requests": 2000,
        "sleep_ms": 120,
        "store_raw": True,
        "raw_compression": "gzip",
        "timeframes_base": ["1m", "1h", "4h"],
        "build_derived_from_1m": True,
        "derived_timeframes": ["5m", "15m", "30m", "2h", "6h", "12h", "1d", "1w", "1mo"],
        "require_min_coverage_years": 2.0,
        "cost_reporting": {"enabled": True},
    }
    merged = dict(default)
    merged.update(dict(config.get("coinapi", {})))
    return merged


def _resolve_range(*, start: str, end: str, last_days: int, default_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    end_ts = pd.Timestamp.utcnow().floor("s") if not str(end).strip() else pd.to_datetime(end, utc=True)
    if str(start).strip():
        start_ts = pd.to_datetime(start, utc=True)
    else:
        days = int(last_days) if int(last_days) > 0 else int(default_days)
        start_ts = end_ts - pd.Timedelta(days=max(1, days))
    if end_ts < start_ts:
        raise ValueError("--end must be >= --start")
    return pd.Timestamp(start_ts), pd.Timestamp(end_ts)


def _extract_payload_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return [row for row in payload["data"] if isinstance(row, dict)]
        for value in payload.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return [row for row in value if isinstance(row, dict)]
    return []


def _raw_path(*, symbol_id: str, endpoint: str, ts: pd.Timestamp) -> Path:
    safe_symbol = str(symbol_id).replace("/", "_").replace(":", "_")
    return (
        Path("data")
        / "coinapi"
        / "raw"
        / "coinapi"
        / safe_symbol
        / str(endpoint)
        / f"{ts.year:04d}"
        / f"{ts.month:02d}"
        / f"{ts.day:02d}.jsonl.gz"
    )


def _write_raw_rows(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return 0
    encoded = [json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n" for row in rows]
    with gzip.open(path, "at", encoding="utf-8") as handle:
        for line in encoded:
            handle.write(line)
    return int(sum(len(line.encode("utf-8")) for line in encoded))


def execute_plan_items(
    *,
    plan: dict[str, Any],
    client: CoinAPIClient,
    store_raw: bool,
    raw_bytes_threshold: int = 2 * 1024 * 1024 * 1024,
) -> dict[str, Any]:
    aggregated: dict[tuple[str, str], list[dict[str, Any]]] = {}
    raw_bytes = 0
    raw_disabled_due_to_size = False
    failures: list[dict[str, Any]] = []
    successes = 0

    for item in plan.get("items", []):
        endpoint = str(item.get("endpoint", ""))
        symbol = str(item.get("symbol", ""))
        symbol_id = str(item.get("symbol_id", ""))
        start_ts = str(item.get("start_ts", ""))
        end_ts = str(item.get("end_ts", ""))
        path = str(item.get("endpoint_path", ""))

        params = {
            "symbol_id": symbol_id,
            "time_start": start_ts,
            "time_end": end_ts,
            "limit": "100000",
        }
        try:
            payload, meta = client.request_json(
                path,
                params=params,
                endpoint_name=endpoint,
                symbol=symbol,
                time_start=start_ts,
                time_end=end_ts,
                plan_id=str(plan.get("plan_id", "")),
            )
            rows = _extract_payload_rows(payload)
            aggregated.setdefault((endpoint, symbol), []).extend(rows)
            if store_raw and not raw_disabled_due_to_size and rows:
                day_path = _raw_path(symbol_id=symbol_id, endpoint=endpoint, ts=pd.to_datetime(start_ts, utc=True))
                wrote = _write_raw_rows(day_path, rows)
                raw_bytes += int(wrote)
                if raw_bytes > int(raw_bytes_threshold):
                    raw_disabled_due_to_size = True
            successes += 1
        except CoinAPIRequestError as exc:
            failures.append(
                {
                    "endpoint": endpoint,
                    "symbol": symbol,
                    "start_ts": start_ts,
                    "end_ts": end_ts,
                    "error": str(exc),
                }
            )

    coverage_rows: list[dict[str, Any]] = []
    for (endpoint, symbol), rows in aggregated.items():
        adapter = ADAPTERS.get(endpoint)
        if adapter is None:
            continue
        normalize_fn, write_fn = adapter
        frame = normalize_fn(rows, symbol=symbol, source="coinapi")
        _, coverage_path = write_fn(frame, symbol=symbol)
        if coverage_path.exists():
            payload = json.loads(coverage_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                coverage_rows.append(payload)

    return {
        "plan_id": str(plan.get("plan_id", "")),
        "planned_count": int(plan.get("planned_count", 0)),
        "selected_count": int(plan.get("selected_count", 0)),
        "success_count": int(successes),
        "failure_count": int(len(failures)),
        "failures": failures,
        "raw_written_bytes": int(raw_bytes),
        "raw_disabled_due_to_size": bool(raw_disabled_due_to_size),
        "coverage_rows": coverage_rows,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    coinapi_cfg = _default_coinapi_config(config)
    enabled = bool(coinapi_cfg.get("enabled", False))
    if bool(args.offline) and enabled:
        raise SystemExit("--offline cannot be used when coinapi.enabled=true")

    symbols = _split_csv(args.symbols) or [str(v) for v in coinapi_cfg.get("symbols", [])]
    endpoints = _split_csv(args.endpoints) or [str(v) for v in coinapi_cfg.get("priority_endpoints", [])]
    start_ts, end_ts = _resolve_range(
        start=str(args.start),
        end=str(args.end),
        last_days=int(args.last_days),
        default_days=int(coinapi_cfg.get("max_days_per_run", 30)),
    )
    run_id = f"{utc_now_compact()}_{stable_hash({'symbols': symbols, 'endpoints': endpoints, 'seed': int(args.seed), 'start': start_ts.isoformat(), 'end': end_ts.isoformat()}, length=12)}_stage35"
    run_dir = Path(args.runs_dir) / run_id / "stage35"
    run_dir.mkdir(parents=True, exist_ok=True)

    cfg_max_requests = int(coinapi_cfg.get("max_total_requests", 2000))
    max_requests = int(args.max_requests) if int(args.max_requests) > 0 else cfg_max_requests
    max_requests = min(max_requests, cfg_max_requests)

    plan = build_backfill_plan(
        symbols=symbols,
        endpoints=endpoints,
        start_ts=start_ts,
        end_ts=end_ts,
        increment_days=max(1, int(args.increment_days)),
        max_requests=max_requests,
    )
    plan_path = run_dir / "coinapi_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, allow_nan=False), encoding="utf-8")

    if bool(args.dry_run):
        print(f"run_id: {run_id}")
        print(f"plan_id: {plan.get('plan_id', '')}")
        print(f"planned_count: {plan.get('planned_count', 0)}")
        print(f"selected_count: {plan.get('selected_count', 0)}")
        print(f"truncated: {plan.get('truncated', False)}")
        print(f"plan_path: {plan_path.as_posix()}")
        return

    if not enabled:
        raise SystemExit("coinapi.enabled=false in config. Enable it for network execution.")

    key = str(os.environ.get("COINAPI_KEY", "")).strip()
    if not key:
        raise SystemExit("COINAPI_KEY is required for CoinAPI execution")
    base_url = str(os.environ.get("COINAPI_BASE_URL", coinapi_cfg.get("base_url", "https://rest.coinapi.io")))
    ledger_path = Path("data") / "coinapi" / "meta" / "usage_ledger.jsonl"
    ledger = CoinAPIUsageLedger(ledger_path)
    client = CoinAPIClient(
        key,
        base_url=base_url,
        sleep_ms=int(coinapi_cfg.get("sleep_ms", 120)),
        max_total_requests=max_requests,
        ledger=ledger,
    )
    result = execute_plan_items(
        plan=plan,
        client=client,
        store_raw=bool(coinapi_cfg.get("store_raw", True)),
        raw_bytes_threshold=2 * 1024 * 1024 * 1024,
    )
    result["run_id"] = run_id
    result_path = run_dir / "coinapi_result.json"
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")

    usage_summary = build_usage_summary(ledger.load_records())
    usage_summary_path = run_dir / "coinapi_usage_summary.json"
    usage_summary_path.write_text(json.dumps(usage_summary, indent=2, allow_nan=False), encoding="utf-8")

    print(f"run_id: {run_id}")
    print(f"plan_id: {plan.get('plan_id', '')}")
    print(f"success_count: {result.get('success_count', 0)}")
    print(f"failure_count: {result.get('failure_count', 0)}")
    print(f"usage_summary_path: {usage_summary_path.as_posix()}")


if __name__ == "__main__":
    main()

