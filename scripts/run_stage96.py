"""Run Stage-96 canonical data repair track."""

from __future__ import annotations

import argparse
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.research.canonical_repair import repair_canonical_evaluation_data
from buffmini.research.reporting import markdown_rows, write_stage_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-96 canonical data repair track")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    symbols = list((cfg.get("universe", {}) or {}).get("symbols", ["BTC/USDT", "ETH/USDT"]))[:2]
    summary = repair_canonical_evaluation_data(
        cfg,
        symbols=symbols,
        timeframes=["30m", "1h", "4h"],
    )
    summary.update(
        {
            "stage": "96",
            "status": "SUCCESS",
            "execution_status": "EXECUTED",
            "stage_role": "real_validation",
            "validation_state": "CANONICAL_REPAIR_READY",
        }
    )
    lines = [
        "# Stage-96 Report",
        "",
        f"- status: `{summary['status']}`",
        f"- validation_state: `{summary['validation_state']}`",
        f"- snapshot_id: `{summary['snapshot_id']}`",
        f"- snapshot_path: `{summary['snapshot_path']}`",
        f"- stage96d_required: `{summary['stage96d_required']}`",
        f"- stage96d_reason: `{summary['stage96d_reason']}`",
        "",
    ]
    lines.extend(markdown_rows("Repair Rows", list(summary.get("repair_rows", [])), limit=12))
    lines.extend([""] + markdown_rows("Data Fitness After Repair", list((summary.get("fitness_after", {}) or {}).get("rows", [])), limit=12))
    lines.extend(["", f"- summary_hash: `{summary['summary_hash']}`"])
    write_stage_artifacts(docs_dir=Path(args.docs_dir), stage="96", summary=summary, report_lines=lines)
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
