"""Export a run's compact strategy metadata/spec into Strategy Library."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from buffmini.ui.components.library import export_run_to_library


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export run artifacts to library/")
    parser.add_argument("--run-id", type=str, required=True)
    parser.add_argument("--display-name", type=str, default=None)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--library-dir", type=Path, default=Path("library"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    card = export_run_to_library(
        run_id=args.run_id,
        display_name=args.display_name,
        runs_dir=args.runs_dir,
        library_dir=args.library_dir,
    )
    print(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()

