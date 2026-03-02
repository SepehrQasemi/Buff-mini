"""Stage-15.9 signal-flow trace runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.forensics.signal_flow import (
    DEFAULT_COMPOSERS,
    DEFAULT_TIMEFRAMES,
    parse_csv_list,
    parse_stage_arg,
    run_signal_flow_trace,
    write_stage15_9_report,
)
from buffmini.signals.registry import family_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-15.9 full signal-flow trace")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--base-timeframe", type=str, default="1m")
    parser.add_argument("--timeframes", type=str, default="15m,30m,1h,2h,4h")
    parser.add_argument("--mode", type=str, choices=["classic", "v2", "both"], default="both")
    parser.add_argument("--stages", type=str, default="15..22")
    parser.add_argument("--families", type=str, default="")
    parser.add_argument("--composer", type=str, default="none,vote,weighted_sum,gated")
    parser.add_argument("--max-combos", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    parser.add_argument("--before-summary", type=Path, default=None, help="Optional previous trace summary json for before/after report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    symbols = parse_csv_list(args.symbols, default=["BTC/USDT", "ETH/USDT"])
    timeframes = parse_csv_list(args.timeframes, default=DEFAULT_TIMEFRAMES)
    stages = parse_stage_arg(args.stages)
    stages = ["classic"] + [s for s in stages if s != "classic"]
    families = parse_csv_list(args.families if str(args.families).strip() else None, default=family_names())
    composers = parse_csv_list(args.composer, default=DEFAULT_COMPOSERS)

    run = run_signal_flow_trace(
        config=cfg,
        seed=int(args.seed),
        symbols=symbols,
        timeframes=timeframes,
        mode=str(args.mode),
        stages=stages,
        families=families,
        composers=composers,
        max_combos=int(args.max_combos),
        dry_run=bool(args.dry_run),
        runs_root=args.runs_dir,
        data_dir=args.data_dir,
        derived_dir=args.derived_dir,
    )
    post_summary = dict(run["summary"])
    pre_summary = _load_json(args.before_summary) if args.before_summary else dict(post_summary)
    pre_run_id = str(pre_summary.get("run_id", "baseline_unknown"))
    post_run_id = str(post_summary.get("run_id", "trace_unknown"))

    defects_fixed = []
    if args.before_summary is not None:
        pre_ctx = float(pre_summary.get("overall_gate_means", {}).get("death_context", 0.0))
        post_ctx = float(post_summary.get("overall_gate_means", {}).get("death_context", 0.0))
        if pre_ctx < 0.0 and post_ctx >= 0.0:
            defects_fixed.append("fixed_context_gate_counting_and_death_rate_clamp")
        if float(pre_summary.get("invalid_pct", 0.0)) > float(post_summary.get("invalid_pct", 0.0)):
            defects_fixed.append("invalid_pct_reduced_after_fix")
        if float(pre_summary.get("walkforward_executed_true_pct", 0.0)) < float(post_summary.get("walkforward_executed_true_pct", 0.0)):
            defects_fixed.append("walkforward_execution_rate_improved_after_fix")
        if float(pre_summary.get("mc_trigger_rate", 0.0)) < float(post_summary.get("mc_trigger_rate", 0.0)):
            defects_fixed.append("mc_trigger_rate_improved_after_fix")

    report_md = Path("docs/stage15_9_signal_flow_bottleneck_report.md")
    report_json = Path("docs/stage15_9_signal_flow_bottleneck_summary.json")
    write_stage15_9_report(
        report_md=report_md,
        report_json=report_json,
        pre_summary=pre_summary,
        post_summary=post_summary,
        pre_run_id=pre_run_id,
        post_run_id=post_run_id,
        defects_fixed=defects_fixed,
    )
    print(f"run_id: {run['run_id']}")
    print(f"trace_dir: {run['trace_dir']}")
    print(f"trace_summary: {Path(run['trace_dir']) / 'trace_summary.json'}")
    print(f"report_md: {report_md}")
    print(f"report_json: {report_json}")


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
