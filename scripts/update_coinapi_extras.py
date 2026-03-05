"""Incremental CoinAPI downloader for funding/open-interest/liquidations extras."""

from __future__ import annotations

import argparse
import gzip
import json
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
from buffmini.data.coinapi.secrets import resolve_coinapi_key
from buffmini.data.coinapi.usage import CoinAPIUsageLedger, build_usage_summary
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


ADAPTERS = {
    "funding_rates": (normalize_funding_rates, write_funding_canonical),
    "open_interest": (normalize_open_interest, write_open_interest_canonical),
    "liquidations": (normalize_liquidations, write_liquidations_canonical),
}

ENDPOINT_ALIASES = {
    "funding": "funding_rates",
    "funding_rates": "funding_rates",
    "oi": "open_interest",
    "open_interest": "open_interest",
    "liquidations": "liquidations",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update CoinAPI enriched extras with budget-aware planning")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--plan", action="store_true")
    actions.add_argument("--download", action="store_true")
    actions.add_argument("--verify", action="store_true")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--endpoints", type=str, default="")
    parser.add_argument("--start", type=str, default="")
    parser.add_argument("--end", type=str, default="")
    parser.add_argument("--years", type=int, default=0)
    parser.add_argument("--last-days", type=int, default=0)
    parser.add_argument("--increment-days", type=int, default=7)
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument("--budget-requests", type=int, default=0)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    return parser.parse_args(argv)


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


def _resolve_range(*, start: str, end: str, years: int, last_days: int, default_days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    end_ts = pd.Timestamp.utcnow().floor("s") if not str(end).strip() else pd.to_datetime(end, utc=True)
    if str(start).strip():
        start_ts = pd.to_datetime(start, utc=True)
    else:
        days = int(last_days) if int(last_days) > 0 else int(default_days)
        if int(years) > 0:
            days = int(round(float(years) * 365.25))
        start_ts = end_ts - pd.Timedelta(days=max(1, days))
    if end_ts < start_ts:
        raise ValueError("--end must be >= --start")
    return pd.Timestamp(start_ts), pd.Timestamp(end_ts)


def normalize_endpoints(endpoints: list[str]) -> list[str]:
    out: list[str] = []
    unknown: list[str] = []
    for endpoint in endpoints:
        key = str(endpoint).strip().lower()
        if not key:
            continue
        mapped = ENDPOINT_ALIASES.get(key)
        if mapped is None:
            unknown.append(str(endpoint))
            continue
        if mapped not in out:
            out.append(mapped)
    if unknown:
        allowed = ", ".join(sorted(ENDPOINT_ALIASES))
        raise SystemExit(f"Unknown endpoint(s): {unknown}. Allowed endpoint names/aliases: {allowed}")
    return out


def resolve_action(args: argparse.Namespace) -> str:
    if bool(args.verify):
        return "verify"
    if bool(args.plan) or bool(args.dry_run):
        return "plan"
    if bool(args.download):
        return "download"
    return "download"


def resolve_max_requests(*, args: argparse.Namespace, cfg_max_requests: int) -> int:
    requested = int(args.max_requests) if int(args.max_requests) > 0 else int(args.budget_requests)
    if requested <= 0:
        requested = int(cfg_max_requests)
    return min(int(requested), int(cfg_max_requests))


def _build_run_id(*, symbols: list[str], endpoints: list[str], seed: int, start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> str:
    return (
        f"{utc_now_compact()}_"
        f"{stable_hash({'symbols': symbols, 'endpoints': endpoints, 'seed': int(seed), 'start': start_ts.isoformat(), 'end': end_ts.isoformat()}, length=12)}"
        "_stage35"
    )


def _print_plan_stdout(*, run_id: str, plan: dict[str, Any], plan_path: Path) -> None:
    print(f"run_id: {run_id}")
    print(f"plan_id: {plan.get('plan_id', '')}")
    print(f"planned_count: {plan.get('planned_count', 0)}")
    print(f"selected_count: {plan.get('selected_count', 0)}")
    print(f"truncated: {plan.get('truncated', False)}")
    print(f"plan_path: {plan_path.as_posix()}")


def _missing_key_error() -> str:
    return "COINAPI_KEY missing; use secrets/coinapi_key.txt (gitignored) or environment variable."


def _write_stage35_7_usage_doc(
    *,
    run_id: str,
    action: str,
    plan: dict[str, Any],
    usage_summary: dict[str, Any],
    run_usage_path: Path,
) -> Path:
    docs_path = Path("docs") / "stage35_7_coinapi_usage.json"
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any]
    if docs_path.exists():
        try:
            existing = json.loads(docs_path.read_text(encoding="utf-8"))
            payload = existing if isinstance(existing, dict) else {}
        except json.JSONDecodeError:
            payload = {}
    else:
        payload = {}
    runs = payload.get("runs", [])
    if not isinstance(runs, list):
        runs = []
    run_row = {
        "run_id": str(run_id),
        "action": str(action),
        "plan_id": str(plan.get("plan_id", "")),
        "total_requests_planned": int(plan.get("planned_count", 0)),
        "total_requests_selected": int(plan.get("selected_count", 0)),
        "total_requests_made": int(usage_summary.get("total_requests", 0)),
        "status_code_counts": dict(usage_summary.get("status_code_counts", {})),
        "endpoints_hit": list(usage_summary.get("endpoints_hit", [])),
        "retry_counts": {
            "total": int(usage_summary.get("total_retries", 0)),
            "by_endpoint": dict(usage_summary.get("retry_count_by_endpoint", {})),
        },
        "time_range": {
            "start": usage_summary.get("time_start_min"),
            "end": usage_summary.get("time_end_max"),
        },
        "rate_limit_sleep_ms_total": int(usage_summary.get("rate_limit_sleep_ms_total", 0)),
        "estimated_credits_used": usage_summary.get("credits_used"),
        "credits_estimation_mode": str(usage_summary.get("credits_estimation_mode", "UNKNOWN")),
        "run_usage_path": run_usage_path.as_posix(),
    }
    runs.append(run_row)
    total_planned = int(sum(int(row.get("total_requests_planned", 0)) for row in runs if isinstance(row, dict)))
    total_made = int(sum(int(row.get("total_requests_made", 0)) for row in runs if isinstance(row, dict)))
    payload["runs"] = runs
    payload["latest"] = run_row
    payload["totals"] = {
        "total_requests_planned": total_planned,
        "total_requests_made": total_made,
        "run_count": int(len(runs)),
    }
    docs_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return docs_path


def _empty_usage_summary(*, start_ts: pd.Timestamp, end_ts: pd.Timestamp, endpoints: list[str]) -> dict[str, Any]:
    return {
        "total_requests": 0,
        "total_success": 0,
        "total_fail": 0,
        "total_bytes": 0,
        "total_retries": 0,
        "status_code_counts": {},
        "per_endpoint": {name: {"requests": 0, "success": 0, "fail": 0, "bytes": 0} for name in endpoints},
        "per_symbol": {},
        "endpoints_hit": [],
        "retry_count_by_endpoint": {name: 0 for name in endpoints},
        "time_start_min": start_ts.isoformat(),
        "time_end_max": end_ts.isoformat(),
        "rate_limit_sleep_ms_total": 0,
        "quota_signals": {},
        "credits_used": None,
        "credits_remaining": None,
        "quota_used": None,
        "quota_remaining": None,
        "credits_estimation_mode": "UNKNOWN",
    }


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
    budget_exhausted = False

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
            if "max_total_requests_exceeded" in str(exc):
                budget_exhausted = True
                break

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
        "budget_exhausted": bool(budget_exhausted),
        "raw_written_bytes": int(raw_bytes),
        "raw_disabled_due_to_size": bool(raw_disabled_due_to_size),
        "coverage_rows": coverage_rows,
    }


def main() -> None:
    args = parse_args()
    action = resolve_action(args)
    config = load_config(args.config)
    coinapi_cfg = _default_coinapi_config(config)
    enabled = bool(coinapi_cfg.get("enabled", False))
    if bool(args.offline) and enabled:
        raise SystemExit("--offline cannot be used when coinapi.enabled=true")

    symbols = _split_csv(args.symbols) or [str(v) for v in coinapi_cfg.get("symbols", [])]
    raw_endpoints = _split_csv(args.endpoints) or [str(v) for v in coinapi_cfg.get("priority_endpoints", [])]
    endpoints = normalize_endpoints(raw_endpoints)
    start_ts, end_ts = _resolve_range(
        start=str(args.start),
        end=str(args.end),
        years=int(args.years),
        last_days=int(args.last_days),
        default_days=int(coinapi_cfg.get("max_days_per_run", 30)),
    )
    run_id = _build_run_id(
        symbols=symbols,
        endpoints=endpoints,
        seed=int(args.seed),
        start_ts=start_ts,
        end_ts=end_ts,
    )
    run_root = Path(args.runs_dir) / run_id
    run_stage35_dir = run_root / "stage35"
    run_coinapi_dir = run_root / "coinapi"
    run_stage35_dir.mkdir(parents=True, exist_ok=True)
    run_coinapi_dir.mkdir(parents=True, exist_ok=True)

    max_requests = resolve_max_requests(args=args, cfg_max_requests=int(coinapi_cfg.get("max_total_requests", 2000)))

    plan = build_backfill_plan(
        symbols=symbols,
        endpoints=endpoints,
        start_ts=start_ts,
        end_ts=end_ts,
        increment_days=max(1, int(args.increment_days)),
        max_requests=max_requests,
    )
    plan_path = run_coinapi_dir / "plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, allow_nan=False), encoding="utf-8")
    (run_stage35_dir / "coinapi_plan.json").write_text(json.dumps(plan, indent=2, allow_nan=False), encoding="utf-8")

    if action == "plan":
        usage_summary = _empty_usage_summary(start_ts=start_ts, end_ts=end_ts, endpoints=endpoints)
        run_usage_path = run_root / "coinapi_usage.json"
        run_usage_path.write_text(json.dumps(usage_summary, indent=2, allow_nan=False), encoding="utf-8")
        _write_stage35_7_usage_doc(
            run_id=run_id,
            action=action,
            plan=plan,
            usage_summary=usage_summary,
            run_usage_path=run_usage_path,
        )
        _print_plan_stdout(run_id=run_id, plan=plan, plan_path=plan_path)
        return

    if not enabled:
        raise SystemExit("coinapi.enabled=false in config. Enable it for network execution.")

    if bool(plan.get("truncated", False)):
        usage_summary = _empty_usage_summary(start_ts=start_ts, end_ts=end_ts, endpoints=endpoints)
        usage_summary["download_refused"] = "planned_requests_exceed_max_requests"
        run_usage_path = run_root / "coinapi_usage.json"
        run_usage_path.write_text(json.dumps(usage_summary, indent=2, allow_nan=False), encoding="utf-8")
        _write_stage35_7_usage_doc(
            run_id=run_id,
            action=action,
            plan=plan,
            usage_summary=usage_summary,
            run_usage_path=run_usage_path,
        )
        raise SystemExit(
            "Planned request count exceeds --max-requests. Increase --increment-days or reduce range before downloading."
        )

    key = str(resolve_coinapi_key(repo_root=Path.cwd()) or "").strip()
    if not key:
        raise SystemExit(_missing_key_error())
    base_url = str(coinapi_cfg.get("base_url", "https://rest.coinapi.io"))
    ledger_path = Path("data") / "coinapi" / "meta" / "usage_ledger.jsonl"
    ledger = CoinAPIUsageLedger(ledger_path)
    client = CoinAPIClient(
        key,
        base_url=base_url,
        sleep_ms=int(coinapi_cfg.get("sleep_ms", 120)),
        max_total_requests=max_requests,
        ledger=ledger,
    )
    if action == "verify":
        try:
            client.request_json(
                "/v1/exchanges",
                endpoint_name="verify_auth",
                symbol="verify",
                time_start=start_ts.isoformat(),
                time_end=end_ts.isoformat(),
                plan_id=str(plan.get("plan_id", "")),
            )
        except CoinAPIRequestError as exc:
            usage_summary = build_usage_summary(ledger.load_records())
            usage_summary["verify_error"] = str(exc)
            usage_summary_path = run_coinapi_dir / "usage_summary.json"
            usage_summary_path.write_text(json.dumps(usage_summary, indent=2, allow_nan=False), encoding="utf-8")
            run_usage_path = run_root / "coinapi_usage.json"
            run_usage_path.write_text(json.dumps(usage_summary, indent=2, allow_nan=False), encoding="utf-8")
            _write_stage35_7_usage_doc(
                run_id=run_id,
                action=action,
                plan=plan,
                usage_summary=usage_summary,
                run_usage_path=run_usage_path,
            )
            message = str(exc)
            if "401" in message or "403" in message:
                raise SystemExit(f"Auth failed during verify endpoint=verify_auth error={message}")
            raise SystemExit(f"Verify failed endpoint=verify_auth error={message}")
        usage_summary = build_usage_summary(ledger.load_records())
        usage_summary_path = run_coinapi_dir / "usage_summary.json"
        usage_summary_path.write_text(json.dumps(usage_summary, indent=2, allow_nan=False), encoding="utf-8")
        run_usage_path = run_root / "coinapi_usage.json"
        run_usage_path.write_text(json.dumps(usage_summary, indent=2, allow_nan=False), encoding="utf-8")
        _write_stage35_7_usage_doc(
            run_id=run_id,
            action=action,
            plan=plan,
            usage_summary=usage_summary,
            run_usage_path=run_usage_path,
        )
        print(f"run_id: {run_id}")
        print("verify_status: OK")
        print(f"usage_summary_path: {usage_summary_path.as_posix()}")
        return

    result = execute_plan_items(
        plan=plan,
        client=client,
        store_raw=bool(coinapi_cfg.get("store_raw", True)),
        raw_bytes_threshold=2 * 1024 * 1024 * 1024,
    )
    result["run_id"] = run_id
    result_path = run_stage35_dir / "coinapi_result.json"
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    (run_coinapi_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")

    usage_summary = build_usage_summary(ledger.load_records())
    usage_summary_path = run_coinapi_dir / "usage_summary.json"
    usage_summary_path.write_text(json.dumps(usage_summary, indent=2, allow_nan=False), encoding="utf-8")
    (run_stage35_dir / "coinapi_usage_summary.json").write_text(json.dumps(usage_summary, indent=2, allow_nan=False), encoding="utf-8")
    run_usage_path = run_root / "coinapi_usage.json"
    run_usage_path.write_text(json.dumps(usage_summary, indent=2, allow_nan=False), encoding="utf-8")
    _write_stage35_7_usage_doc(
        run_id=run_id,
        action=action,
        plan=plan,
        usage_summary=usage_summary,
        run_usage_path=run_usage_path,
    )

    print(f"run_id: {run_id}")
    print(f"plan_id: {plan.get('plan_id', '')}")
    print(f"success_count: {result.get('success_count', 0)}")
    print(f"failure_count: {result.get('failure_count', 0)}")
    print(f"usage_summary_path: {usage_summary_path.as_posix()}")


if __name__ == "__main__":
    main()
