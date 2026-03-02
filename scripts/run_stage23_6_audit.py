"""CLI wrapper for Stage-23.6 sizing integrity baseline-vs-after audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.signals.registry import family_names
from buffmini.stage23.sizing_audit import run_stage23_6_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-23.6 sizing integrity audit")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--timeframes", type=str, default="15m,30m,1h,2h,4h")
    parser.add_argument("--mode", type=str, choices=["classic", "v2", "both"], default="both")
    parser.add_argument("--stages", type=str, default="15,16,17,18,19,20,21,22")
    parser.add_argument("--families", type=str, default="")
    parser.add_argument("--composer", type=str, default="none,vote,weighted_sum,gated")
    parser.add_argument("--max-combos", type=int, default=0)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    return parser.parse_args()


def _csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    families = _csv(args.families) if str(args.families).strip() else family_names()
    result = run_stage23_6_audit(
        config=config,
        seed=int(args.seed),
        dry_run=bool(args.dry_run),
        symbols=_csv(args.symbols),
        timeframes=_csv(args.timeframes),
        mode=str(args.mode),
        stages=_csv(args.stages),
        families=families,
        composers=_csv(args.composer),
        max_combos=int(args.max_combos),
        runs_root=args.runs_dir,
        data_dir=args.data_dir,
        derived_dir=args.derived_dir,
    )
    payload = dict(result["payload"])
    print(f"baseline_run_id: {result['baseline_run_id']}")
    print(f"after_run_id: {result['after_run_id']}")
    print(f"report_md: {result['report_md']}")
    print(f"report_json: {result['report_json']}")
    print(f"diff_json: {result['diff_json']}")
    print(f"zero_trade_pct: {float(payload['baseline']['metrics']['zero_trade_pct']):.6f} -> {float(payload['after']['metrics']['zero_trade_pct']):.6f}")
    print(
        "size_zero_share: "
        f"{float(payload['criteria']['size_zero_share_baseline']):.6f} -> "
        f"{float(payload['criteria']['size_zero_share_after']):.6f}"
    )
    print(f"improvement_sufficient: {bool(payload['criteria']['improvement_sufficient'])}")


if __name__ == "__main__":
    main()
