"""Run Stage-102 robustness rescue attempts."""

from __future__ import annotations

import argparse
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.research.reporting import markdown_kv, markdown_rows, write_stage_artifacts
from buffmini.research.rescue import run_rescue_attempts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-102 robustness rescue attempts")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--candidate-limit", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    summary = run_rescue_attempts(
        cfg,
        symbol=str(args.symbol),
        timeframe=str(args.timeframe),
        candidate_limit=int(args.candidate_limit),
    )
    summary.update(
        {
            "stage": "102",
            "status": "SUCCESS",
            "execution_status": "EXECUTED",
            "stage_role": "real_validation",
            "validation_state": "ROBUSTNESS_RESCUE_ATTEMPTS_READY",
        }
    )
    lines = [
        "# Stage-102 Report",
        "",
        f"- status: `{summary['status']}`",
        f"- validation_state: `{summary['validation_state']}`",
        f"- symbol: `{summary['symbol']}`",
        f"- timeframe: `{summary['timeframe']}`",
        f"- candidate_limit_reviewed: `{summary['candidate_limit_reviewed']}`",
        f"- blocked: `{summary['blocked']}`",
        f"- blocked_reason: `{summary['blocked_reason']}`",
        "",
    ]
    lines.extend(markdown_kv("Classification Counts", dict(summary.get("classification_counts", {}))))
    lines.extend([""] + markdown_rows("Candidate Rows", list(summary.get("candidate_rows", [])), limit=12))
    lines.extend(["", f"- summary_hash: `{summary['summary_hash']}`"])
    write_stage_artifacts(docs_dir=Path(args.docs_dir), stage="102", summary=summary, report_lines=lines)
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
