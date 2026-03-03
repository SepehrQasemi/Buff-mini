"""Stage-26.9.1 raw data audit for canonical raw 1m series."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.constants import RAW_DATA_DIR
from buffmini.data.canonical_raw import detect_gaps_minutes, resolve_raw_meta_path, resolve_raw_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit canonical raw 1m data coverage/integrity")
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--exchange", type=str, default="binance")
    parser.add_argument("--timeframe", type=str, default="1m")
    parser.add_argument("--required-years", type=float, default=4.0)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def _audit_one(*, symbol: str, exchange: str, timeframe: str, data_dir: Path) -> dict[str, Any]:
    path = resolve_raw_path(data_dir=data_dir, exchange=exchange, symbol=symbol, timeframe=timeframe)
    meta_path = resolve_raw_meta_path(data_dir=data_dir, exchange=exchange, symbol=symbol, timeframe=timeframe)
    if not path.exists():
        return {
            "symbol": str(symbol),
            "exchange": str(exchange),
            "timeframe": str(timeframe),
            "exists": False,
            "path": str(path),
            "meta_path": str(meta_path),
            "start_ts": None,
            "end_ts": None,
            "coverage_years": 0.0,
            "rows": 0,
            "duplicates": 0,
            "non_monotonic": False,
            "gaps_detected": {"count": 0, "max_gap_minutes": 0},
        }
    frame = pd.read_parquet(path)
    ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce").dropna()
    ts_sorted = ts.sort_values().reset_index(drop=True)
    coverage_days = 0.0
    if not ts_sorted.empty:
        step_minutes = 1 if str(timeframe).strip().lower().endswith("m") else 60
        coverage_days = float(((ts_sorted.iloc[-1] - ts_sorted.iloc[0]).total_seconds() + (step_minutes * 60.0)) / 86400.0)
    gaps = detect_gaps_minutes(pd.Series(ts_sorted), expected_minutes=1)
    meta = {}
    if meta_path.exists():
        try:
            meta = dict(json.loads(meta_path.read_text(encoding="utf-8")))
        except Exception:
            meta = {}
    return {
        "symbol": str(symbol),
        "exchange": str(exchange),
        "timeframe": str(timeframe),
        "exists": True,
        "path": str(path),
        "meta_path": str(meta_path),
        "start_ts": ts_sorted.iloc[0].isoformat() if not ts_sorted.empty else None,
        "end_ts": ts_sorted.iloc[-1].isoformat() if not ts_sorted.empty else None,
        "coverage_years": float(coverage_days / 365.25),
        "rows": int(ts_sorted.shape[0]),
        "duplicates": int(ts_sorted.duplicated().sum()),
        "non_monotonic": bool((ts.diff().dropna() < pd.Timedelta(0)).any()) if not ts.empty else False,
        "gaps_detected": {"count": int(gaps.gaps_detected), "max_gap_minutes": int(gaps.max_gap_minutes)},
        "meta_sha256": str(meta.get("sha256", "")),
    }


def _render_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage-26.9 Raw Data Audit",
        "",
        f"- required_years: `{float(payload.get('required_years', 4.0)):.2f}`",
        f"- coverage_ok_all_symbols: `{bool(payload.get('coverage_ok_all_symbols', False))}`",
        "",
        "| symbol | exists | start_ts | end_ts | coverage_years | rows | dup | gaps | max_gap_min |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload.get("rows", []):
        gap = dict(row.get("gaps_detected", {}))
        lines.append(
            f"| {row.get('symbol','')} | {row.get('exists',False)} | {row.get('start_ts','')} | {row.get('end_ts','')} | {float(row.get('coverage_years',0.0)):.6f} | {int(row.get('rows',0))} | {int(row.get('duplicates',0))} | {int(gap.get('count',0))} | {int(gap.get('max_gap_minutes',0))} |"
        )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    symbols = [item.strip() for item in str(args.symbols).split(",") if item.strip()]
    rows = [_audit_one(symbol=s, exchange=str(args.exchange), timeframe=str(args.timeframe), data_dir=args.data_dir) for s in symbols]
    required = float(args.required_years)
    coverage_ok = all(bool(row.get("exists", False)) and float(row.get("coverage_years", 0.0)) >= required for row in rows)
    payload = {
        "stage": "26.9.1",
        "required_years": required,
        "rows": rows,
        "coverage_ok_all_symbols": bool(coverage_ok),
    }
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    md_path = docs_dir / "stage26_9_raw_data_audit.md"
    json_path = docs_dir / "stage26_9_raw_data_audit.json"
    md_path.write_text(_render_md(payload), encoding="utf-8")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"audit_md: {md_path}")
    print(f"audit_json: {json_path}")
    raise SystemExit(0 if coverage_ok else 2)


if __name__ == "__main__":
    main()
