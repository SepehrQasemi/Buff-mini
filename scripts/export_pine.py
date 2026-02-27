"""Export Stage-5.6 Pine scripts from run artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from buffmini.constants import RUNS_DIR
from buffmini.spec.pine_export import export_pine_scripts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Pine scripts for one Buff-mini run")
    parser.add_argument("--run-id", type=str, required=True, help="Run id used as export source")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = export_pine_scripts(run_id=str(args.run_id), runs_dir=Path(args.runs_dir))
    print(f"run_id: {index['run_id']}")
    print(f"component_count: {index['component_count']}")
    print(f"deterministic_export: {index['deterministic_export']}")
    print(f"all_files_valid: {index['validation']['all_files_valid']}")
    print(json.dumps(index.get("validation", {}), indent=2))
    print(f"export_dir: {Path(args.runs_dir) / str(args.run_id) / 'exports' / 'pine'}")


if __name__ == "__main__":
    main()
