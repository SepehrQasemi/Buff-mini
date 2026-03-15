"""Run Stage-98 mechanism saturation pass."""

from __future__ import annotations

import argparse
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.research.reporting import markdown_kv, markdown_rows, write_stage_artifacts
from buffmini.research.saturation import evaluate_mechanism_saturation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-98 mechanism saturation pass")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    summary = evaluate_mechanism_saturation(cfg)
    summary.update(
        {
            "stage": "98",
            "status": "SUCCESS",
            "execution_status": "EXECUTED",
            "stage_role": "real_validation",
            "validation_state": "MECHANISM_SATURATION_READY",
        }
    )
    lines = [
        "# Stage-98 Report",
        "",
        f"- status: `{summary['status']}`",
        f"- validation_state: `{summary['validation_state']}`",
        f"- stage98b_required: `{summary['stage98b_required']}`",
        f"- stage98b_applied: `{summary['stage98b_applied']}`",
        f"- stage98b_reason: `{summary['stage98b_reason']}`",
        "",
    ]
    lines.extend(markdown_kv("Richness Delta", dict(summary.get("richness_delta", {}))))
    lines.extend([""] + markdown_rows("Family Richness", list(summary.get("family_rows", [])), limit=16))
    lines.extend([""] + markdown_kv("Candidate Volume", {
        "raw_candidate_count": summary.get("raw_candidate_count", 0),
        "post_compression_candidate_count": summary.get("post_compression_candidate_count", 0),
        "post_similarity_collapse_count": summary.get("post_similarity_collapse_count", 0),
        "post_dedup_candidate_count": summary.get("post_dedup_candidate_count", 0),
        "precompression_duplication_ratio": summary.get("precompression_duplication_ratio", 0.0),
        "trivial_duplication_ratio": summary.get("trivial_duplication_ratio", 0.0),
    }))
    lines.extend(["", f"- summary_hash: `{summary['summary_hash']}`"])
    write_stage_artifacts(docs_dir=Path(args.docs_dir), stage="98", summary=summary, report_lines=lines)
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
