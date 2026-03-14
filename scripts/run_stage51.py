"""Run Stage-51 research scope freeze and budget framework."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.stage51 import build_stage51_summary, render_stage51_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-51 research scope freeze")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    summary = build_stage51_summary(cfg)
    report = render_stage51_report(summary)
    summary_path = docs_dir / "stage51_summary.json"
    report_path = docs_dir / "stage51_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(report, encoding="utf-8")
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
