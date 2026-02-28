"""Generate Stage-9 OI backfill coverage audit report from local artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.data.derived_store import read_meta_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-9 OI backfill audit")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--derived-dir", type=Path, default=Path("data") / "derived")
    parser.add_argument("--out-md", type=Path, default=Path("docs") / "stage9_oi_backfill_report.md")
    parser.add_argument("--out-json", type=Path, default=Path("docs") / "stage9_oi_backfill_summary.json")
    return parser.parse_args()


def _symbol_payload(meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "start_ts": meta.get("start_ts"),
        "end_ts": meta.get("end_ts"),
        "row_count": int(meta.get("row_count", 0) or 0),
        "expected_rows": int(meta.get("total_expected_rows", 0) or 0),
        "coverage_ratio": float(meta.get("coverage_ratio", 0.0) or 0.0),
        "gap_count": int(meta.get("gaps_count", 0) or 0),
        "largest_gap_hours": float(meta.get("largest_gap_hours", 0.0) or 0.0),
        "warnings": list(meta.get("warnings", [])) if isinstance(meta.get("warnings", []), list) else [],
        "stop_reason": str(meta.get("stop_reason", "")),
        "previous_start_ts": meta.get("previous_start_ts"),
        "previous_row_count": meta.get("previous_row_count"),
        "previous_coverage_ratio": meta.get("previous_coverage_ratio"),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    symbols = list(config.get("data", {}).get("futures_extras", {}).get("symbols", ["BTC/USDT", "ETH/USDT"]))
    timeframe = str(config.get("data", {}).get("futures_extras", {}).get("timeframe", "1h"))

    summary: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        meta = read_meta_json(
            kind="open_interest",
            symbol=symbol,
            timeframe=timeframe,
            data_dir=args.derived_dir,
        )
        summary[symbol] = _symbol_payload(meta)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Stage-9 OI Backfill Report")
    lines.append("")
    lines.append("| symbol | old_start | new_start | row_count | expected_rows | coverage_ratio | gap_count | largest_gap_hours | stop_reason |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |")

    for symbol in symbols:
        payload = summary[symbol]
        lines.append(
            f"| {symbol} | {payload.get('previous_start_ts')} | {payload.get('start_ts')} | "
            f"{int(payload['row_count'])} | {int(payload['expected_rows'])} | "
            f"{float(payload['coverage_ratio']):.6f} | {int(payload['gap_count'])} | "
            f"{float(payload['largest_gap_hours']):.3f} | {payload.get('stop_reason')} |"
        )

    lines.append("")
    lines.append("## Improvement vs Previous Meta")
    for symbol in symbols:
        payload = summary[symbol]
        old_cov = payload.get("previous_coverage_ratio")
        if old_cov is None:
            lines.append(f"- {symbol}: no previous coverage metadata available for comparison.")
            continue
        old_cov_f = float(old_cov)
        new_cov_f = float(payload["coverage_ratio"])
        delta = new_cov_f - old_cov_f
        lines.append(
            f"- {symbol}: old_coverage={old_cov_f:.6f}, new_coverage={new_cov_f:.6f}, delta={delta:.6f}"
        )
        old_rows = payload.get("previous_row_count")
        if old_rows is not None:
            lines.append(f"- {symbol}: old_rows={int(old_rows)}, new_rows={int(payload['row_count'])}")

    lines.append("")
    lines.append("## Notes")
    lines.append("- Coverage may remain low if Binance OI API history retention is limited.")
    lines.append("- No forward-looking alignment is used; OI merge is latest `ts <= candle_close` only.")
    lines.append("")
    lines.append("## Warnings")
    for symbol in symbols:
        warnings = summary[symbol].get("warnings", [])
        if warnings:
            for warning in warnings:
                lines.append(f"- {symbol}: {warning}")
        else:
            lines.append(f"- {symbol}: none")

    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    print(f"wrote: {args.out_md}")
    print(f"wrote: {args.out_json}")


if __name__ == "__main__":
    main()

