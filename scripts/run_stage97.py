"""Run Stage-97 relaxed-to-strict bridge."""

from __future__ import annotations

import argparse
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.research.bridge import build_relaxed_to_strict_bridge
from buffmini.research.reporting import markdown_kv, markdown_rows, write_stage_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-97 relaxed-to-strict bridge")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--candidate-limit-per-scope", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    summary = build_relaxed_to_strict_bridge(
        cfg,
        candidate_limit_per_scope=int(args.candidate_limit_per_scope),
    )
    summary.update(
        {
            "stage": "97",
            "status": "SUCCESS" if summary.get("bridge_rows") else "PARTIAL",
            "execution_status": "EXECUTED",
            "stage_role": "real_validation",
            "validation_state": "RELAXED_TO_STRICT_BRIDGE_READY" if summary.get("bridge_rows") else "RELAXED_TO_STRICT_BRIDGE_EMPTY",
        }
    )
    lines = [
        "# Stage-97 Report",
        "",
        f"- status: `{summary['status']}`",
        f"- validation_state: `{summary['validation_state']}`",
        f"- symbols: `{summary['symbols']}`",
        f"- timeframes: `{summary['timeframes']}`",
        f"- relaxed_candidate_count: `{summary['relaxed_candidate_count']}`",
        "",
    ]
    lines.extend(markdown_kv("Bridge Classification Counts", dict(summary.get("classification_counts", {}))))
    lines.extend([""] + markdown_rows("Bridge Rows", list(summary.get("bridge_rows", [])), limit=24))
    lines.extend(["", f"- summary_hash: `{summary['summary_hash']}`"])
    write_stage_artifacts(docs_dir=Path(args.docs_dir), stage="97", summary=summary, report_lines=lines)
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
