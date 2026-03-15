"""Run Stage-85 reality isolation and diagnosis."""

from __future__ import annotations

import argparse
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.research.reality import evaluate_reality_matrix
from buffmini.research.reporting import markdown_kv, markdown_rows, write_stage_artifacts
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-85 reality isolation and diagnosis")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    summary = evaluate_reality_matrix(cfg)
    envs = dict(summary.get("environments", {}))
    summary.update(
        {
            "stage": "85",
            "status": "SUCCESS" if envs else "PARTIAL",
            "execution_status": "EXECUTED",
            "stage_role": "real_validation",
            "validation_state": "REALITY_MATRIX_READY" if envs else "REALITY_MATRIX_INCOMPLETE",
        }
    )
    summary["summary_hash"] = stable_hash(summary, length=16)
    lines = [
        "# Stage-85 Report",
        "",
        f"- status: `{summary['status']}`",
        f"- execution_status: `{summary['execution_status']}`",
        f"- stage_role: `{summary['stage_role']}`",
        f"- validation_state: `{summary['validation_state']}`",
        "",
        "## Reality Matrix",
    ]
    for env, payload in envs.items():
        lines.append(
            f"- {env}: candidate_count=`{payload.get('candidate_count', 0)}` "
            f"promising_count=`{payload.get('promising_count', 0)}` "
            f"validated_count=`{payload.get('validated_count', 0)}` "
            f"robust_count=`{payload.get('robust_count', 0)}` "
            f"blocked_count=`{payload.get('blocked_count', 0)}`"
        )
    lines.extend([""] + markdown_kv("Dominant Blockers", {row['blocker']: row['count'] for row in summary.get('dominant_blockers', [])}))
    lines.extend([""] + markdown_rows("Gate Sensitivity", [{"gate": gate, **payload} for gate, payload in dict(summary.get("gate_sensitivity", {})).items()], limit=5))
    lines.extend(["", f"- summary_hash: `{summary['summary_hash']}`"])
    write_stage_artifacts(docs_dir=Path(args.docs_dir), stage="85", summary=summary, report_lines=lines)
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
