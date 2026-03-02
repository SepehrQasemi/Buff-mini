"""Audit Stage-26 data coverage for required 4-year span."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from buffmini.constants import RAW_DATA_DIR
from buffmini.stage26.coverage import CoverageResult, audit_symbol_coverage
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit local OHLCV coverage")
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--base-timeframe", type=str, default="1m")
    parser.add_argument("--required-years", type=int, default=4)
    parser.add_argument("--end-mode", type=str, default="latest")
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def _csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def render_md(payload: dict) -> str:
    lines = [
        "# Stage-26 Data Coverage (4-Year Audit)",
        "",
        f"- generated_at: `{payload.get('generated_at', '')}`",
        f"- required_years: `{payload.get('required_years', 0)}`",
        f"- coverage_ok_all_symbols: `{payload.get('coverage_ok_all_symbols', False)}`",
        "",
        "| symbol | timeframe | exists | start_ts | end_ts | coverage_years | missing_bars_estimate | gap_days_estimate |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in payload.get("rows", []):
        lines.append(
            "| {symbol} | {timeframe} | {exists} | {start_ts} | {end_ts} | {coverage_years:.4f} | {missing_bars_estimate} | {gap_days_estimate:.2f} |".format(
                symbol=row.get("symbol", ""),
                timeframe=row.get("timeframe", ""),
                exists=row.get("exists", False),
                start_ts=row.get("start_ts", ""),
                end_ts=row.get("end_ts", ""),
                coverage_years=float(row.get("coverage_years", 0.0)),
                missing_bars_estimate=int(row.get("missing_bars_estimate", 0)),
                gap_days_estimate=float(row.get("gap_days_estimate", 0.0)),
            )
        )
    if not payload.get("coverage_ok_all_symbols", False):
        lines.extend(
            [
                "",
                "## Warning",
                "- Coverage is below required years for at least one symbol.",
                "- To update local data manually (optional), run existing update scripts (no auto-download in this audit).",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    symbols = _csv(args.symbols)
    rows: list[CoverageResult] = []
    for symbol in symbols:
        rows.append(
            audit_symbol_coverage(
                symbol=symbol,
                timeframe=str(args.base_timeframe),
                data_dir=args.data_dir,
                end_mode=str(args.end_mode),
            )
        )
    required_years = float(args.required_years)
    rows_dict = [row.to_dict() for row in rows]
    coverage_ok = all(bool(row.exists) and float(row.coverage_years) >= required_years for row in rows)
    payload = {
        "stage": "26.1",
        "generated_at": utc_now_compact(),
        "required_years": float(required_years),
        "timeframe": str(args.base_timeframe),
        "rows": rows_dict,
        "coverage_ok_all_symbols": bool(coverage_ok),
    }
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    docs_json = docs_dir / "stage26_data_coverage_4y.json"
    docs_md = docs_dir / "stage26_data_coverage_4y.md"
    docs_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    docs_md.write_text(render_md(payload), encoding="utf-8")

    print(f"coverage_ok_all_symbols: {coverage_ok}")
    print(f"report_json: {docs_json}")
    print(f"report_md: {docs_md}")
    raise SystemExit(0 if coverage_ok else 2)


if __name__ == "__main__":
    main()
