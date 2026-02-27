"""Run Stage-1 auto-optimization discovery funnel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RAW_DATA_DIR, RUNS_DIR
from buffmini.discovery.funnel import run_stage1_optimization
from buffmini.utils.logging import get_logger


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-1 auto-optimization")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic offline data")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)

    parser.add_argument("--candidate-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--cost-pct", type=float, default=None)
    parser.add_argument("--stage-a-months", type=int, default=None)
    parser.add_argument("--stage-b-months", type=int, default=None)
    parser.add_argument("--holdout-months", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    run_dir = run_stage1_optimization(
        config=config,
        config_path=args.config,
        dry_run=bool(args.dry_run),
        run_id=args.run_id,
        runs_dir=args.runs_dir,
        data_dir=args.data_dir,
        candidate_count=args.candidate_count,
        seed=args.seed,
        cost_pct=args.cost_pct,
        stage_a_months=args.stage_a_months,
        stage_b_months=args.stage_b_months,
        holdout_months=args.holdout_months,
    )

    strategies_path = run_dir / "strategies.json"
    if not strategies_path.exists():
        logger.warning("Stage-1 completed but strategies.json not found at %s", strategies_path)
        return

    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        logger.warning("Stage-1 completed but summary.json not found at %s", summary_path)
        return

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best_tier_a = summary.get("best_tier_A")

    print("Stage-1 Result Tiers")
    print(f"Tier A: {summary.get('tier_A_count', 0)}")
    print(f"Tier B: {summary.get('tier_B_count', 0)}")
    print(f"Near Miss: {summary.get('near_miss_count', 0)}")
    if best_tier_a is not None:
        metrics = best_tier_a["metrics_holdout"]
        print(
            "Best Tier A | "
            f"{best_tier_a['strategy_name']} | "
            f"edge={metrics['effective_edge']:.4f} | "
            f"exp_lcb={metrics['exp_lcb']:.4f} | "
            f"tpm={metrics['trades_per_month']:.4f}"
        )
    else:
        print("Best Tier A | none")


if __name__ == "__main__":
    main()
