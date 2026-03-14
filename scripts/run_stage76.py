"""Run Stage-76 signal detectability proof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.research.synthetic_lab import evaluate_detectability_suite


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-76 signal detectability proof")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    summary = evaluate_detectability_suite(cfg, seed=int(args.seed))
    summary.update(
        {
            "stage": "76",
            "execution_status": "EXECUTED",
            "stage_role": "real_validation",
            "validation_state": "SIGNAL_DETECTABILITY_PROVEN" if summary["status"] == "SUCCESS" else "SIGNAL_DETECTABILITY_WEAK",
            "seed": int(args.seed),
        }
    )
    (docs_dir / "stage76_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        "# Stage-76 Report",
        "",
        f"- status: `{summary['status']}`",
        f"- execution_status: `{summary['execution_status']}`",
        f"- stage_role: `{summary['stage_role']}`",
        f"- validation_state: `{summary['validation_state']}`",
        f"- candidate_count: `{summary['candidate_count']}`",
        f"- signal_detection_rate: `{summary['signal_detection_rate']}`",
        f"- bad_control_rejection_rate: `{summary['bad_control_rejection_rate']}`",
        f"- synthetic_winner_recall: `{summary['synthetic_winner_recall']}`",
        f"- false_negative_rate_on_known_good: `{summary['false_negative_rate_on_known_good']}`",
        "",
        "## Promotion Rate By Regime",
    ]
    for regime, value in sorted(dict(summary.get("promotion_rate_by_regime", {})).items()):
        lines.append(f"- {regime}: `{value}`")
    lines.extend(["", "## Candidate Classes"])
    for label, value in sorted(dict(summary.get("candidate_classes", {})).items()):
        lines.append(f"- {label}: `{value}`")
    lines.extend(["", f"- summary_hash: `{summary['summary_hash']}`"])
    (docs_dir / "stage76_report.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
