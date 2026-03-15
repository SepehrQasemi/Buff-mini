"""Run Stage-89 data fitness and canonical comparison."""

from __future__ import annotations

import argparse
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.research.data_fitness import evaluate_data_fitness
from buffmini.research.reporting import markdown_rows, write_stage_artifacts
from buffmini.research.transfer import discover_transfer_symbols
from buffmini.stage51.scope import resolve_research_scope
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-89 data fitness and canonical comparison")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    scope = resolve_research_scope(cfg)
    symbols = discover_transfer_symbols(cfg)[: max(2, len(scope.get("primary_symbols", [])))]
    timeframes = [tf for tf in ["15m", "30m", "1h", "4h"] if tf in set(scope.get("discovery_timeframes", [])) or tf == "1h"]
    fitness = evaluate_data_fitness(cfg, symbols=symbols, timeframes=timeframes)
    rows = list(fitness.get("rows", []))
    class_counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("evaluation_usable_class", "unknown"))
        class_counts[key] = class_counts.get(key, 0) + 1
    summary = {
        "stage": "89",
        "status": "SUCCESS" if rows else "PARTIAL",
        "execution_status": "EXECUTED",
        "stage_role": "reporting_only",
        "validation_state": "DATA_FITNESS_READY" if rows else "DATA_FITNESS_INCOMPLETE",
        **fitness,
        "evaluation_usable_class_counts": class_counts,
    }
    summary["summary_hash"] = stable_hash(summary, length=16)
    lines = [
        "# Stage-89 Report",
        "",
        f"- status: `{summary['status']}`",
        f"- validation_state: `{summary['validation_state']}`",
        f"- row_count: `{len(rows)}`",
    ]
    lines.extend([""] + markdown_rows("Evaluation Usable Class Counts", [{"class": key, "count": value} for key, value in class_counts.items()]))
    lines.extend([""] + markdown_rows("Data Fitness Rows", rows, limit=12))
    lines.extend(["", f"- summary_hash: `{summary['summary_hash']}`"])
    write_stage_artifacts(docs_dir=Path(args.docs_dir), stage="89", summary=summary, report_lines=lines)
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
