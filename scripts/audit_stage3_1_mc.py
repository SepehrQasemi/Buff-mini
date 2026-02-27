"""Audit Stage-3.1 Monte Carlo outputs against Stage-2 trade reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.constants import RAW_DATA_DIR, RUNS_DIR
from buffmini.portfolio.monte_carlo import load_portfolio_trades


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-check Stage-3.1 Monte Carlo outputs")
    parser.add_argument("--stage2-run-id", type=str, required=True)
    parser.add_argument("--stage3-run-id", type=str, required=True)
    parser.add_argument("--initial-equity", type=float, default=10000.0)
    parser.add_argument("--output", type=Path, default=Path("docs") / "stage3_1_mc_crosscheck.md")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stage2_summary = _load_json(args.runs_dir / args.stage2_run_id / "portfolio_summary.json")
    stage3_summary = _load_json(args.runs_dir / args.stage3_run_id / "mc_summary.json")

    rows: list[dict[str, Any]] = []
    detail_lines: list[str] = []
    for method in ["equal", "vol", "corr-min"]:
        trades = load_portfolio_trades(
            stage2_run_id=args.stage2_run_id,
            method=method,
            runs_dir=args.runs_dir,
            data_dir=args.data_dir,
        )
        baseline = _baseline_metrics(trades, initial_equity=float(args.initial_equity))
        summary_payload = stage3_summary["methods"][method]
        mc = summary_payload["summary"]
        expected_count = int(summary_payload["trade_count_source"])

        trade_count_match = int(len(trades)) == expected_count
        return_in_band = float(mc["return_pct"]["p05"]) <= float(baseline["total_return_pct"]) <= float(mc["return_pct"]["p95"])
        maxdd_within_p95 = float(baseline["max_drawdown"]) <= float(mc["max_drawdown"]["p95"])

        holdout_range = str(stage2_summary["portfolio_methods"][method]["holdout"]["date_range"])
        holdout_start, holdout_end = holdout_range.split("..", 1)
        holdout_start_ts = pd.Timestamp(holdout_start, tz="UTC")
        holdout_end_ts = pd.Timestamp(holdout_end, tz="UTC")
        min_entry = pd.to_datetime(trades["entry_ts"], utc=True).min() if not trades.empty else pd.NaT
        max_exit = pd.to_datetime(trades["exit_ts"], utc=True).max() if not trades.empty else pd.NaT
        timestamp_inside = bool(pd.notna(min_entry) and pd.notna(max_exit) and min_entry >= holdout_start_ts and max_exit <= holdout_end_ts)

        status = "PASS" if all([trade_count_match, return_in_band, maxdd_within_p95, timestamp_inside]) else "WARN"
        rows.append(
            {
                "method": method,
                "status": status,
                "trade_count_reconstructed": int(len(trades)),
                "trade_count_stage3": expected_count,
                "baseline_return_pct": float(baseline["total_return_pct"]),
                "mc_return_p05": float(mc["return_pct"]["p05"]),
                "mc_return_p95": float(mc["return_pct"]["p95"]),
                "baseline_max_dd": float(baseline["max_drawdown"]),
                "mc_maxdd_p95": float(mc["max_drawdown"]["p95"]),
                "trade_range": f"{min_entry} .. {max_exit}",
                "holdout_range": holdout_range,
                "trade_count_match": trade_count_match,
                "return_in_band": return_in_band,
                "maxdd_within_p95": maxdd_within_p95,
                "timestamp_inside_holdout": timestamp_inside,
            }
        )
        detail_lines.append(
            f"- {method}: status={status}, trade_count_match={trade_count_match}, return_in_band={return_in_band}, "
            f"maxdd_within_p95={maxdd_within_p95}, timestamp_inside_holdout={timestamp_inside}"
        )

    report_lines: list[str] = []
    report_lines.append("# Stage-3.1 Monte Carlo Cross-Check")
    report_lines.append("")
    report_lines.append(f"- stage2_run_id: `{args.stage2_run_id}`")
    report_lines.append(f"- stage3_run_id: `{args.stage3_run_id}`")
    report_lines.append(f"- initial_equity: `{float(args.initial_equity)}`")
    report_lines.append("")
    report_lines.append("## Summary")
    for line in detail_lines:
        report_lines.append(line)
    report_lines.append("")
    report_lines.append("## Method Table")
    report_lines.append("| method | status | trade_count_reconstructed | trade_count_stage3 | baseline_return_pct | mc_return_p05 | mc_return_p95 | baseline_max_dd | mc_maxdd_p95 | trade_count_match | return_in_band | maxdd_within_p95 | timestamp_inside_holdout |")
    report_lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |")
    for row in rows:
        report_lines.append(
            f"| {row['method']} | {row['status']} | {row['trade_count_reconstructed']} | {row['trade_count_stage3']} | "
            f"{row['baseline_return_pct']:.4f} | {row['mc_return_p05']:.4f} | {row['mc_return_p95']:.4f} | "
            f"{row['baseline_max_dd']:.4f} | {row['mc_maxdd_p95']:.4f} | {row['trade_count_match']} | "
            f"{row['return_in_band']} | {row['maxdd_within_p95']} | {row['timestamp_inside_holdout']} |"
        )
    report_lines.append("")
    report_lines.append("## Timestamp Ranges")
    for row in rows:
        report_lines.append(f"- {row['method']}: trades `{row['trade_range']}` vs holdout `{row['holdout_range']}`")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(report_lines).strip() + "\n", encoding="utf-8")
    print(f"wrote cross-check report: {args.output}")


def _baseline_metrics(trades: pd.DataFrame, initial_equity: float) -> dict[str, float]:
    if trades.empty:
        return {
            "trade_count": 0.0,
            "total_return_pct": 0.0,
            "max_drawdown": 0.0,
        }
    pnls = trades["pnl"].astype(float).to_numpy()
    equity = float(initial_equity) + pnls.cumsum()
    equity_with_initial = [float(initial_equity), *equity.tolist()]
    peaks = []
    running_peak = float(initial_equity)
    for value in equity_with_initial:
        running_peak = max(running_peak, float(value))
        peaks.append(running_peak)
    drawdowns = [((peak - value) / peak) if peak > 0 else 0.0 for peak, value in zip(peaks, equity_with_initial, strict=False)]
    max_dd = max(drawdowns) if drawdowns else 0.0
    final_equity = equity_with_initial[-1]
    return {
        "trade_count": float(len(pnls)),
        "total_return_pct": float((final_equity / float(initial_equity)) - 1.0),
        "max_drawdown": float(max_dd),
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
