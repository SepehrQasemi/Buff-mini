"""Run Stage-77 canonical/evaluation mode hardening."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.research.modes import build_mode_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-77 evaluation-mode hardening")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--mode", type=str, default="evaluation")
    parser.add_argument("--auto-pin-resolved-end", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    _, summary = build_mode_context(cfg, requested_mode=str(args.mode), auto_pin_resolved_end=bool(args.auto_pin_resolved_end))
    summary.update(
        {
            "stage": "77",
            "status": "SUCCESS" if summary["interpretation_allowed"] else "PARTIAL",
            "execution_status": "EXECUTED",
            "stage_role": "real_validation",
        }
    )
    (docs_dir / "stage77_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        "# Stage-77 Report",
        "",
        f"- status: `{summary['status']}`",
        f"- execution_status: `{summary['execution_status']}`",
        f"- stage_role: `{summary['stage_role']}`",
        f"- mode: `{summary['mode']}`",
        f"- evaluation_mode: `{summary['evaluation_mode']}`",
        f"- interpretation_allowed: `{summary['interpretation_allowed']}`",
        f"- validation_state: `{summary['validation_state']}`",
        f"- canonical_status: `{summary['canonical_status']}`",
        f"- resolved_end_ts: `{summary['resolved_end_ts']}`",
        f"- resolved_end_ts_status: `{summary['resolved_end_ts_status']}`",
        f"- data_scope_hash: `{summary['data_scope_hash']}`",
        "",
        "## Effective Values",
    ]
    for key, value in sorted(dict(summary.get("effective_values", {})).items()):
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Blocked Reasons"])
    if summary.get("blocked_reasons"):
        for reason in summary["blocked_reasons"]:
            lines.append(f"- {reason}")
    else:
        lines.append("- (none)")
    lines.extend(["", f"- summary_hash: `{summary['summary_hash']}`"])
    (docs_dir / "stage77_report.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
