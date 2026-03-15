"""Run Stage-99 candidate quality acceleration."""

from __future__ import annotations

import argparse
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.research.quality import evaluate_candidate_quality_acceleration
from buffmini.research.reporting import markdown_kv, markdown_rows, write_stage_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-99 candidate quality acceleration")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--candidate-limit", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    summary = evaluate_candidate_quality_acceleration(
        cfg,
        symbol=str(args.symbol),
        timeframe=str(args.timeframe),
        candidate_limit=int(args.candidate_limit),
    )
    summary.update(
        {
            "stage": "99",
            "status": "SUCCESS",
            "execution_status": "EXECUTED",
            "stage_role": "real_validation",
            "validation_state": "CANDIDATE_QUALITY_ACCELERATION_READY",
        }
    )
    lines = [
        "# Stage-99 Report",
        "",
        f"- status: `{summary['status']}`",
        f"- validation_state: `{summary['validation_state']}`",
        f"- symbol: `{summary['symbol']}`",
        f"- timeframe: `{summary['timeframe']}`",
        f"- stage99b_required: `{summary['stage99b_required']}`",
        f"- stage99b_applied: `{summary['stage99b_applied']}`",
        "",
    ]
    lines.extend(markdown_kv("Before Counts", dict(summary.get("before_counts", {}))))
    lines.extend([""] + markdown_kv("After Counts", dict(summary.get("after_counts", {}))))
    lines.extend([""] + markdown_rows("Transition Rows", list(summary.get("transition_rows", [])), limit=24))
    lines.extend([""] + markdown_rows("Gate Heatmap", list(summary.get("gate_heatmap", [])), limit=24))
    lines.extend([""] + markdown_rows("Near Miss Inventory", list(summary.get("near_miss_inventory", [])), limit=24))
    lines.extend([""] + markdown_rows("Top-K Truth Review", list(summary.get("top_k_truth_review", [])), limit=24))
    lines.extend(["", f"- summary_hash: `{summary['summary_hash']}`"])
    write_stage_artifacts(docs_dir=Path(args.docs_dir), stage="99", summary=summary, report_lines=lines)
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
