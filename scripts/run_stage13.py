"""Run Stage-13 family engine evaluations."""

from __future__ import annotations

import argparse
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.stage13.evaluate import (
    run_stage13,
    run_stage13_combined_matrix,
    run_stage13_family_sweep,
    run_stage13_multihorizon,
    run_stage13_robustness,
    validate_stage13_summary_schema,
)
from buffmini.stage14.evaluate import (
    run_stage13_14_master_report,
    run_stage14_meta_family,
    run_stage14_nested_walkforward,
    run_stage14_threshold_calibration,
    run_stage14_weighting,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-13 signal family engine")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbols", type=str, default=None, help="Optional comma-separated symbols")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--families", type=str, default=None, help="Optional comma-separated families")
    parser.add_argument("--composer", type=str, default="none", choices=["none", "vote", "weighted_sum", "gated"])
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    parser.add_argument("--stage-tag", type=str, default="13.1")
    parser.add_argument("--report-name", type=str, default="stage13_1_architecture")
    parser.add_argument("--substage", type=str, default="13.1", choices=["13.1", "13.2", "13.3", "13.4", "13.5", "13.6", "13.7", "14.1", "14.2", "14.3", "14.4"])
    parser.add_argument("--family", type=str, default="price", choices=["price", "volatility", "flow"])
    parser.add_argument("--enable", action="store_true", help="Force enable evaluation.stage13.enabled")
    parser.add_argument("--disable", action="store_true", help="Force disable evaluation.stage13.enabled")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if bool(args.enable):
        cfg.setdefault("evaluation", {}).setdefault("stage13", {})["enabled"] = True
    if bool(args.disable):
        cfg.setdefault("evaluation", {}).setdefault("stage13", {})["enabled"] = False
    symbols = [item.strip() for item in str(args.symbols).split(",") if item.strip()] if args.symbols else None
    families = [item.strip() for item in str(args.families).split(",") if item.strip()] if args.families else None
    substage = str(args.substage)
    if substage in {"13.2", "13.3", "13.4"}:
        stage_map = {
            "13.2": ("stage13_2_price_family", "price"),
            "13.3": ("stage13_3_volatility_family", "volatility"),
            "13.4": ("stage13_4_flow_family", "flow"),
        }
        report_name, default_family = stage_map[substage]
        family = str(args.family or default_family)
        result = run_stage13_family_sweep(
            config=cfg,
            family=family,
            seed=int(args.seed),
            dry_run=bool(args.dry_run),
            symbols=symbols,
            timeframe=str(args.timeframe),
            runs_root=args.runs_dir,
            docs_dir=Path("docs"),
            data_dir=args.data_dir,
            derived_dir=args.derived_dir,
            stage_tag=substage,
            report_name=report_name,
        )
        summary = dict(result["summary"])
        print(f"classification: {summary.get('classification','')}")
        print(f"trade_count_ratio_vs_baseline: {float(summary.get('trade_count_ratio_vs_baseline',0.0)):.6f}")
        print(f"report_md: {result['report_md']}")
        print(f"report_json: {result['report_json']}")
        return
    if substage == "13.5":
        result = run_stage13_combined_matrix(
            config=cfg,
            seed=int(args.seed),
            dry_run=bool(args.dry_run),
            symbols=symbols,
            timeframe=str(args.timeframe),
            runs_root=args.runs_dir,
            docs_dir=Path("docs"),
            data_dir=args.data_dir,
            derived_dir=args.derived_dir,
            stage_tag="13.5",
            report_name="stage13_5_combined",
        )
        summary = dict(result["summary"])
        print(f"classification: {summary.get('classification','')}")
        print(f"gate_pass: {bool(summary.get('gate_pass', False))}")
        print(f"report_md: {result['report_md']}")
        print(f"report_json: {result['report_json']}")
        return
    if substage == "13.6":
        result = run_stage13_robustness(
            config=cfg,
            seed=int(args.seed),
            dry_run=bool(args.dry_run),
            symbols=symbols,
            timeframe=str(args.timeframe),
            runs_root=args.runs_dir,
            docs_dir=Path("docs"),
            data_dir=args.data_dir,
            derived_dir=args.derived_dir,
            stage_tag="13.6",
            report_name="stage13_6_robustness",
        )
        summary = dict(result["summary"])
        print(f"classification: {summary.get('classification','')}")
        print(f"report_md: {result['report_md']}")
        print(f"report_json: {result['report_json']}")
        return
    if substage == "13.7":
        result = run_stage13_multihorizon(
            config=cfg,
            seed=int(args.seed),
            dry_run=bool(args.dry_run),
            symbols=symbols,
            timeframe=str(args.timeframe),
            runs_root=args.runs_dir,
            docs_dir=Path("docs"),
            data_dir=args.data_dir,
            derived_dir=args.derived_dir,
            stage_tag="13.7",
            report_name="stage13_7_multihorizon",
        )
        summary = dict(result["summary"])
        print(f"classification: {summary.get('classification','')}")
        print(f"horizon_consistency_score: {float(summary.get('horizon_consistency_score', 0.0)):.6f}")
        print(f"report_md: {result['report_md']}")
        print(f"report_json: {result['report_json']}")
        return
    if substage == "14.1":
        result = run_stage14_weighting(
            config=cfg,
            seed=int(args.seed),
            dry_run=bool(args.dry_run),
            symbols=symbols,
            timeframe=str(args.timeframe),
            runs_root=args.runs_dir,
            docs_dir=Path("docs"),
            data_dir=args.data_dir,
            derived_dir=args.derived_dir,
            stage_tag="14.1",
            report_name="stage14_1_weighting",
        )
        print(f"classification: {result['summary']['classification']}")
        print(f"report_md: {result['report_md']}")
        print(f"report_json: {result['report_json']}")
        return
    if substage == "14.2":
        result = run_stage14_threshold_calibration(
            config=cfg,
            seed=int(args.seed),
            dry_run=bool(args.dry_run),
            symbols=symbols,
            timeframe=str(args.timeframe),
            runs_root=args.runs_dir,
            docs_dir=Path("docs"),
            data_dir=args.data_dir,
            derived_dir=args.derived_dir,
            stage_tag="14.2",
            report_name="stage14_2_threshold",
        )
        print(f"classification: {result['summary']['classification']}")
        print(f"report_md: {result['report_md']}")
        print(f"report_json: {result['report_json']}")
        return
    if substage == "14.3":
        result = run_stage14_nested_walkforward(
            config=cfg,
            seed=int(args.seed),
            dry_run=bool(args.dry_run),
            symbols=symbols,
            timeframe=str(args.timeframe),
            runs_root=args.runs_dir,
            docs_dir=Path("docs"),
            data_dir=args.data_dir,
            derived_dir=args.derived_dir,
            stage_tag="14.3",
            report_name="stage14_3_nested_wf",
        )
        print(f"classification: {result['summary']['classification']}")
        print(f"report_md: {result['report_md']}")
        print(f"report_json: {result['report_json']}")
        return
    if substage == "14.4":
        result = run_stage14_meta_family(
            config=cfg,
            seed=int(args.seed),
            dry_run=bool(args.dry_run),
            symbols=symbols,
            timeframe=str(args.timeframe),
            runs_root=args.runs_dir,
            docs_dir=Path("docs"),
            data_dir=args.data_dir,
            derived_dir=args.derived_dir,
            stage_tag="14.4",
            report_name="stage14_4_meta_family",
        )
        master = run_stage13_14_master_report(docs_dir=Path("docs"))
        print(f"classification: {result['summary']['classification']}")
        print(f"master_final_verdict: {master.get('final_verdict','')}")
        print(f"report_md: {result['report_md']}")
        print(f"report_json: {result['report_json']}")
        return

    result = run_stage13(
        config=cfg,
        seed=int(args.seed),
        dry_run=bool(args.dry_run),
        symbols=symbols,
        timeframe=str(args.timeframe),
        families=families,
        composer_mode=str(args.composer),
        runs_root=args.runs_dir,
        docs_dir=Path("docs"),
        data_dir=args.data_dir,
        derived_dir=args.derived_dir,
        stage_tag=str(args.stage_tag),
        report_name=str(args.report_name),
    )
    summary = dict(result["summary"])
    validate_stage13_summary_schema(summary)
    metrics = dict(summary.get("metrics", {}))
    print(f"run_id: {summary['run_id']}")
    print(f"classification: {summary['classification']}")
    print(f"zero_trade_pct: {float(metrics.get('zero_trade_pct', 0.0)):.6f}")
    print(f"invalid_pct: {float(metrics.get('invalid_pct', 0.0)):.6f}")
    print(f"walkforward_executed_true_pct: {float(metrics.get('walkforward_executed_true_pct', 0.0)):.6f}")
    print(f"mc_trigger_rate: {float(metrics.get('mc_trigger_rate', 0.0)):.6f}")
    print(f"report_md: {result['report_md']}")
    print(f"report_json: {result['report_json']}")


if __name__ == "__main__":
    main()
