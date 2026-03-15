"""Run Stage-100 multi-scope truth campaign."""

from __future__ import annotations

import argparse
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.research.reporting import markdown_kv, markdown_rows, write_stage_artifacts
from buffmini.research.truth_table import run_multiscope_truth_campaign


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-100 multi-scope truth campaign")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--candidate-limit-per-scope", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    summary = run_multiscope_truth_campaign(
        cfg,
        candidate_limit_per_scope=int(args.candidate_limit_per_scope),
    )
    summary.update(
        {
            "stage": "100",
            "status": "SUCCESS",
            "execution_status": "EXECUTED",
            "stage_role": "real_validation",
            "validation_state": "MULTISCOPE_TRUTH_CAMPAIGN_READY",
        }
    )
    lines = [
        "# Stage-100 Report",
        "",
        f"- status: `{summary['status']}`",
        f"- validation_state: `{summary['validation_state']}`",
        f"- candidate_limit_per_scope: `{summary['candidate_limit_per_scope']}`",
        f"- tier1_symbols: `{summary['tier1_symbols']}`",
        f"- tier2_symbols: `{summary['tier2_symbols']}`",
        "",
    ]
    lines.extend(markdown_kv("Truth Counts", dict(summary.get("truth_counts", {}))))
    lines.extend([""] + markdown_rows("Scope Truth Rows", list(summary.get("scope_truth_rows", [])), limit=36))
    lines.extend([""] + markdown_rows("Regime Truth Rows", list(summary.get("regime_truth_rows", [])), limit=48))
    lines.extend(["", f"- summary_hash: `{summary['summary_hash']}`"])
    write_stage_artifacts(docs_dir=Path(args.docs_dir), stage="100", summary=summary, report_lines=lines)
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
