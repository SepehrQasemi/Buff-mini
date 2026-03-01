"""Run Stage-13/14 forensic root-cause audit."""

from __future__ import annotations

import argparse
from pathlib import Path

from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.forensics.stage13_14_root_cause import run_forensic_stage13_14_root_cause


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-13/14 forensic root-cause audit")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = run_forensic_stage13_14_root_cause(
        config_path=args.config,
        seed=int(args.seed),
        docs_dir=args.docs_dir,
    )
    print(f"raw_json: {out['raw_path']}")
    print(f"summary_json: {out['summary_path']}")
    print(f"report_md: {out['report_path']}")
    print(f"final_conclusion: {out['summary'].get('final_conclusion','OTHER')}")


if __name__ == "__main__":
    main()
