"""Stage-35 orchestrator: CoinAPI extras ingestion, coverage audit, and usage reporting."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.data.coinapi.client import CoinAPIClient
from buffmini.data.coinapi.planner import build_backfill_plan, estimate_additional_days_required
from buffmini.data.coinapi.secrets import resolve_coinapi_key
from buffmini.data.coinapi.usage import CoinAPIUsageLedger, build_usage_summary
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact

try:
    from scripts.update_coinapi_extras import execute_plan_items
except ModuleNotFoundError:  # pragma: no cover - direct script execution path
    from update_coinapi_extras import execute_plan_items


SPOT_TO_COINAPI = {
    "BTC/USDT": "BINANCE_PERP_BTC_USDT",
    "ETH/USDT": "BINANCE_PERP_ETH_USDT",
}
KEY_ENDPOINTS = ("funding_rates", "open_interest")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-35 CoinAPI enrichment and reporting")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--allow-insufficient-coverage", action="store_true")
    return parser.parse_args()


def _coinapi_defaults(config: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "enabled": False,
        "symbols": ["BINANCE_PERP_BTC_USDT", "BINANCE_PERP_ETH_USDT"],
        "priority_endpoints": ["funding_rates", "open_interest", "liquidations"],
        "max_days_per_run": 30,
        "max_total_requests": 2000,
        "sleep_ms": 120,
        "store_raw": True,
        "raw_compression": "gzip",
        "require_min_coverage_years": 2.0,
        "cost_reporting": {"enabled": True},
    }
    merged = dict(defaults)
    merged.update(dict(config.get("coinapi", {})))
    return merged


def _safe_symbol(symbol: str) -> str:
    return str(symbol).replace("/", "_").replace(":", "_").replace("-", "_")


def _detect_endpoint_frame(symbol: str, endpoint: str, root: Path) -> pd.DataFrame:
    candidates = [symbol, SPOT_TO_COINAPI.get(symbol, symbol), symbol.replace("/", "_"), symbol.replace("/", "-")]
    for cand in candidates:
        path = root / _safe_symbol(cand) / f"{endpoint}.parquet"
        if path.exists():
            frame = pd.read_parquet(path)
            if "ts" in frame.columns:
                frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
            return frame
    return pd.DataFrame(columns=["ts"])


def _coverage_years(frame: pd.DataFrame) -> float:
    if frame.empty or "ts" not in frame.columns:
        return 0.0
    ts = pd.to_datetime(frame["ts"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return 0.0
    span_days = float((ts.iloc[-1] - ts.iloc[0]) / pd.Timedelta(days=1))
    return float(max(0.0, span_days / 365.25))


def _coverage_summary(config: dict[str, Any], coinapi_cfg: dict[str, Any]) -> dict[str, Any]:
    symbols = [str(v) for v in config.get("universe", {}).get("symbols", ["BTC/USDT", "ETH/USDT"])]
    endpoints = [str(v) for v in coinapi_cfg.get("priority_endpoints", KEY_ENDPOINTS)]
    canonical_root = Path("data") / "coinapi" / "canonical"
    out: dict[str, Any] = {"symbols": {}, "required_years": float(coinapi_cfg.get("require_min_coverage_years", 2.0))}
    for symbol in symbols:
        per_ep: dict[str, Any] = {}
        for endpoint in endpoints:
            frame = _detect_endpoint_frame(symbol, endpoint, canonical_root)
            years = _coverage_years(frame)
            ts = pd.to_datetime(frame.get("ts"), utc=True, errors="coerce").dropna() if "ts" in frame.columns else pd.Series(dtype="datetime64[ns, UTC]")
            per_ep[endpoint] = {
                "coverage_years": float(years),
                "sample_count": int(frame.shape[0]),
                "start_ts": ts.iloc[0].isoformat() if not ts.empty else None,
                "end_ts": ts.iloc[-1].isoformat() if not ts.empty else None,
            }
        out["symbols"][symbol] = per_ep
    return out


def _check_min_coverage(coverage: dict[str, Any], *, required_years: float) -> tuple[bool, list[dict[str, Any]]]:
    missing: list[dict[str, Any]] = []
    symbols = coverage.get("symbols", {})
    for symbol, endpoint_rows in symbols.items():
        if not isinstance(endpoint_rows, dict):
            continue
        for endpoint in KEY_ENDPOINTS:
            row = endpoint_rows.get(endpoint, {})
            years = float(row.get("coverage_years", 0.0))
            if years < float(required_years):
                missing.append(
                    {
                        "symbol": str(symbol),
                        "endpoint": str(endpoint),
                        "coverage_years": float(years),
                        "missing_days_estimate": float(estimate_additional_days_required(coverage_years=years, required_years=float(required_years))),
                    }
                )
    return (len(missing) == 0), missing


def _storage_footprint() -> dict[str, float]:
    base = Path("data") / "coinapi"
    out: dict[str, float] = {}
    for name in ("raw", "canonical", "meta"):
        root = base / name
        total = 0
        if root.exists():
            for path in root.rglob("*"):
                if path.is_file():
                    total += int(path.stat().st_size)
        out[f"{name}_mb"] = float(total / (1024 * 1024))
    out["total_mb"] = float(sum(out.values()))
    return out


def _usage_docs(summary: dict[str, Any]) -> tuple[Path, Path]:
    md_path = Path("docs/stage35_coinapi_usage_report.md")
    json_path = Path("docs/stage35_coinapi_usage_summary.json")
    lines = [
        "# Stage-35 CoinAPI Usage Report",
        "",
        f"- total_requests: `{int(summary.get('total_requests', 0))}`",
        f"- total_success: `{int(summary.get('total_success', 0))}`",
        f"- total_fail: `{int(summary.get('total_fail', 0))}`",
        f"- total_bytes: `{int(summary.get('total_bytes', 0))}`",
        f"- credits_estimation_mode: `{summary.get('credits_estimation_mode', 'UNKNOWN')}`",
        "",
        "## Per Endpoint",
        "| endpoint | requests | success | fail | bytes |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for endpoint, row in sorted((summary.get("per_endpoint", {}) or {}).items()):
        lines.append(
            f"| {endpoint} | {int(row.get('requests', 0))} | {int(row.get('success', 0))} | {int(row.get('fail', 0))} | {int(row.get('bytes', 0))} |"
        )
    lines.extend(
        [
            "",
            "## Per Symbol",
            "| symbol | requests | success | fail | bytes |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for symbol, row in sorted((summary.get("per_symbol", {}) or {}).items()):
        lines.append(
            f"| {symbol} | {int(row.get('requests', 0))} | {int(row.get('success', 0))} | {int(row.get('fail', 0))} | {int(row.get('bytes', 0))} |"
        )
    md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return md_path, json_path


def _render_stage35_docs(payload: dict[str, Any]) -> tuple[Path, Path]:
    md_path = Path("docs/stage35_report.md")
    json_path = Path("docs/stage35_report_summary.json")
    lines = [
        "# Stage-35 Report",
        "",
        "## Summary",
        f"- head_commit: `{payload.get('head_commit', '')}`",
        f"- run_id: `{payload.get('run_id', '')}`",
        f"- coinapi_enabled: `{payload.get('coinapi_enabled', False)}`",
        f"- dry_run: `{payload.get('dry_run', False)}`",
        f"- status: `{payload.get('status', '')}`",
        "",
        "## Endpoints",
        f"- requested: `{payload.get('requested_endpoints', [])}`",
        f"- download_attempted: `{payload.get('download_attempted', False)}`",
        "",
        "## Coverage",
        f"- required_years: `{payload.get('coverage', {}).get('required_years', 0.0)}`",
    ]
    for symbol, endpoint_rows in sorted((payload.get("coverage", {}).get("symbols", {}) or {}).items()):
        lines.append(f"- {symbol}:")
        for endpoint, row in sorted((endpoint_rows or {}).items()):
            lines.append(
                "  - {ep}: years={years:.3f}, samples={samples}, range={start}..{end}".format(
                    ep=endpoint,
                    years=float(row.get("coverage_years", 0.0)),
                    samples=int(row.get("sample_count", 0)),
                    start=str(row.get("start_ts", "None")),
                    end=str(row.get("end_ts", "None")),
                )
            )
    lines.extend(
        [
            "",
            "## Usage",
            f"- total_requests: `{int(payload.get('usage', {}).get('total_requests', 0))}`",
            f"- total_success: `{int(payload.get('usage', {}).get('total_success', 0))}`",
            f"- total_fail: `{int(payload.get('usage', {}).get('total_fail', 0))}`",
            "",
            "## ML Trigger",
            f"- ml_executed: `{payload.get('ml', {}).get('executed', False)}`",
            f"- reason: `{payload.get('ml', {}).get('reason', '')}`",
            "",
            "## Blocking Reasons",
        ]
    )
    blockers = payload.get("missing_coverage", []) or []
    if blockers:
        lines.extend([f"- {item}" for item in blockers])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Storage",
            f"- raw_mb: `{payload.get('storage', {}).get('raw_mb', 0.0):.3f}`",
            f"- canonical_mb: `{payload.get('storage', {}).get('canonical_mb', 0.0):.3f}`",
            f"- meta_mb: `{payload.get('storage', {}).get('meta_mb', 0.0):.3f}`",
            f"- total_mb: `{payload.get('storage', {}).get('total_mb', 0.0):.3f}`",
        ]
    )
    md_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return md_path, json_path


def run_stage35_pipeline(
    *,
    config_path: Path,
    seed: int,
    dry_run: bool,
    runs_dir: Path,
    allow_insufficient_coverage: bool,
) -> dict[str, Any]:
    config = load_config(config_path)
    coinapi_cfg = _coinapi_defaults(config)
    run_id = f"{utc_now_compact()}_{stable_hash({'seed': int(seed), 'cfg': str(config_path), 'dry_run': bool(dry_run)}, length=12)}_stage35"
    run_dir = Path(runs_dir) / run_id / "stage35"
    run_dir.mkdir(parents=True, exist_ok=True)

    requested_endpoints = [str(v) for v in coinapi_cfg.get("priority_endpoints", [])]
    symbols = [str(v) for v in coinapi_cfg.get("symbols", [])]
    now_utc = pd.Timestamp.utcnow().floor("s")
    start_utc = now_utc - pd.Timedelta(days=int(coinapi_cfg.get("max_days_per_run", 30)))

    plan = build_backfill_plan(
        symbols=symbols,
        endpoints=requested_endpoints,
        start_ts=start_utc,
        end_ts=now_utc,
        increment_days=7,
        max_requests=int(coinapi_cfg.get("max_total_requests", 2000)),
    )
    (run_dir / "coinapi_plan.json").write_text(json.dumps(plan, indent=2, allow_nan=False), encoding="utf-8")

    download_attempted = False
    result: dict[str, Any] = {"success_count": 0, "failure_count": 0, "failures": []}
    if bool(coinapi_cfg.get("enabled", False)) and not bool(dry_run):
        key = str(resolve_coinapi_key(repo_root=Path.cwd()) or "").strip()
        if not key:
            raise SystemExit("COINAPI_KEY is required when coinapi.enabled=true and not --dry-run (.secrets supported)")
        base_url = str(coinapi_cfg.get("base_url", "https://rest.coinapi.io")).strip()
        ledger = CoinAPIUsageLedger(Path("data") / "coinapi" / "meta" / "usage_ledger.jsonl")
        client = CoinAPIClient(
            key,
            base_url=base_url,
            sleep_ms=int(coinapi_cfg.get("sleep_ms", 120)),
            max_total_requests=int(coinapi_cfg.get("max_total_requests", 2000)),
            ledger=ledger,
        )
        result = execute_plan_items(
            plan=plan,
            client=client,
            store_raw=bool(coinapi_cfg.get("store_raw", True)),
            raw_bytes_threshold=2 * 1024 * 1024 * 1024,
        )
        (run_dir / "coinapi_result.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        download_attempted = True

    coverage = _coverage_summary(config, coinapi_cfg)
    meta_dir = Path("data") / "coinapi" / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / "coverage_summary.json").write_text(json.dumps(coverage, indent=2, allow_nan=False), encoding="utf-8")

    ledger = CoinAPIUsageLedger(meta_dir / "usage_ledger.jsonl")
    usage_summary = build_usage_summary(ledger.load_records())
    _usage_docs(usage_summary)

    min_required = float(coinapi_cfg.get("require_min_coverage_years", 2.0))
    coverage_ok, missing = _check_min_coverage(coverage, required_years=min_required)
    missing_lines = [
        "{symbol}/{endpoint}: years={years:.3f}, missing_days~{days:.1f}".format(
            symbol=row["symbol"],
            endpoint=row["endpoint"],
            years=float(row["coverage_years"]),
            days=float(row["missing_days_estimate"]),
        )
        for row in missing
    ]

    ml_payload: dict[str, Any]
    if coverage_ok and not bool(dry_run):
        # Stage-35 keeps ML run optional/manual-heavy; this branch is explicit and truthful.
        ml_payload = {
            "executed": False,
            "reason": "coverage_met_manual_ml_run_recommended",
            "baseline": {},
            "after": {},
            "delta": {},
        }
        status = "READY_FOR_ML"
    else:
        ml_payload = {
            "executed": False,
            "reason": "insufficient_coverage" if not coverage_ok else "dry_run",
            "baseline": {},
            "after": {},
            "delta": {},
        }
        if not coverage_ok and not bool(allow_insufficient_coverage):
            status = "INSUFFICIENT_COVERAGE"
        elif not coverage_ok:
            status = "INSUFFICIENT_COVERAGE_ALLOWED"
        else:
            status = "DRY_RUN"

    head_commit = os.popen("git rev-parse --short HEAD").read().strip()
    summary = {
        "head_commit": head_commit,
        "run_id": run_id,
        "seed": int(seed),
        "coinapi_enabled": bool(coinapi_cfg.get("enabled", False)),
        "dry_run": bool(dry_run),
        "download_attempted": bool(download_attempted),
        "requested_endpoints": requested_endpoints,
        "plan_id": str(plan.get("plan_id", "")),
        "plan_counts": {
            "planned_count": int(plan.get("planned_count", 0)),
            "selected_count": int(plan.get("selected_count", 0)),
            "truncated": bool(plan.get("truncated", False)),
        },
        "download_result": result,
        "usage": usage_summary,
        "coverage": coverage,
        "missing_coverage": missing_lines,
        "status": status,
        "ml": ml_payload,
        "storage": _storage_footprint(),
    }
    _render_stage35_docs(summary)
    return summary


def main() -> None:
    args = parse_args()
    payload = run_stage35_pipeline(
        config_path=Path(args.config),
        seed=int(args.seed),
        dry_run=bool(args.dry_run),
        runs_dir=Path(args.runs_dir),
        allow_insufficient_coverage=bool(args.allow_insufficient_coverage),
    )
    print(f"run_id: {payload.get('run_id', '')}")
    print(f"status: {payload.get('status', '')}")
    print("docs: docs/stage35_report.md")
    print("summary: docs/stage35_report_summary.json")


if __name__ == "__main__":
    main()
