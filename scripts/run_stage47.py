"""Stage-47 signal genesis 2.0 runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.constants import RUNS_DIR
from buffmini.stage47.genesis import (
    beam_search_setups,
    generate_setup_candidates,
    summarize_stage47_candidates,
)
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-47 signal genesis 2.0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--stage28-run-id", type=str, default="")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _resolve_stage28_run_id(args: argparse.Namespace, docs_dir: Path) -> str:
    if str(args.stage28_run_id).strip():
        return str(args.stage28_run_id).strip()
    stage39 = _load_json(docs_dir / "stage39_signal_generation_summary.json")
    if str(stage39.get("stage28_run_id", "")).strip():
        return str(stage39["stage28_run_id"]).strip()
    stage38 = _load_json(docs_dir / "stage38_master_summary.json")
    return str(stage38.get("stage28_run_id", "")).strip()


def _render(payload: dict[str, Any], *, notes: list[str]) -> str:
    lines = [
        "# Stage-47 Signal Genesis 2.0 Report",
        "",
        f"- status: `{payload.get('status', '')}`",
        f"- baseline_raw_candidate_count: `{int(payload.get('baseline_raw_candidate_count', 0))}`",
        f"- upgraded_raw_candidate_count: `{int(payload.get('upgraded_raw_candidate_count', 0))}`",
        f"- shortlisted_count: `{int(payload.get('shortlisted_count', 0))}`",
        f"- lineage_examples_present: `{bool(payload.get('lineage_examples_present', False))}`",
        "",
        "## Setup Family Counts",
    ]
    setup_counts = dict(payload.get("setup_family_counts", {}))
    for key, value in sorted(setup_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
        lines.append(f"- {key}: {int(value)}")
    lines.extend(["", "## Context Counts"])
    context_counts = dict(payload.get("context_counts", {}))
    for key, value in sorted(context_counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
        lines.append(f"- {key}: {int(value)}")
    if notes:
        lines.extend(["", "## Partial Notes"])
        lines.extend([f"- {note}" for note in notes])
    lines.extend(["", f"- summary_hash: `{payload.get('summary_hash', '')}`"])
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    stage39 = _load_json(docs_dir / "stage39_signal_generation_summary.json")
    stage28_run_id = _resolve_stage28_run_id(args, docs_dir)
    if not stage28_run_id:
        raise SystemExit("unable to resolve stage28_run_id for Stage-47")

    layer_a_path = Path(args.runs_dir) / stage28_run_id / "stage39" / "layer_a_candidates.csv"
    if not layer_a_path.exists():
        raise SystemExit(f"missing Stage-39 layer_a file: {layer_a_path}")
    layer_a = pd.read_csv(layer_a_path)

    setups = generate_setup_candidates(layer_a, seed=int(args.seed))
    shortlist = beam_search_setups(setups, beam_width=96, per_family_max=18)
    core = summarize_stage47_candidates(
        baseline_raw_candidate_count=int(stage39.get("raw_candidate_count", 0)),
        setups=setups,
        shortlist=shortlist,
    )

    status = "SUCCESS"
    notes: list[str] = []
    if int(core.get("upgraded_raw_candidate_count", 0)) <= 0:
        status = "PARTIAL"
        notes.append("No setup candidates generated from Stage-39 Layer-A input.")
    if not bool(core.get("lineage_examples_present", False)):
        status = "PARTIAL"
        notes.append("Candidate lineage not present in shortlist.")

    payload = {
        "stage": "47",
        "status": status,
        **core,
    }
    payload["summary_hash"] = stable_hash(
        {
            "stage": payload["stage"],
            "status": payload["status"],
            "baseline_raw_candidate_count": payload["baseline_raw_candidate_count"],
            "upgraded_raw_candidate_count": payload["upgraded_raw_candidate_count"],
            "shortlisted_count": payload["shortlisted_count"],
            "setup_family_counts": payload["setup_family_counts"],
            "context_counts": payload["context_counts"],
            "lineage_examples_present": payload["lineage_examples_present"],
        },
        length=16,
    )

    out_dir = Path(args.runs_dir) / stage28_run_id / "stage47"
    out_dir.mkdir(parents=True, exist_ok=True)
    setups.to_csv(out_dir / "setup_candidates.csv", index=False)
    shortlist.to_csv(out_dir / "setup_shortlist.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    summary_path = docs_dir / "stage47_signal_gen2_summary.json"
    report_path = docs_dir / "stage47_signal_gen2_report.md"
    summary_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(_render(payload, notes=notes), encoding="utf-8")

    print(f"status: {payload['status']}")
    print(f"summary_hash: {payload['summary_hash']}")
    print(f"report: {report_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()

