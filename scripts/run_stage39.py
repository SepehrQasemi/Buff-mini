"""Stage-39 upstream signal generation upgrade runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.constants import RUNS_DIR
from buffmini.stage39.signal_generation import build_layered_candidates, summarize_layered_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-39 upstream signal generation upgrade")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--stage28-run-id", type=str, default="")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _resolve_stage28_run_id(args: argparse.Namespace) -> str:
    direct = str(args.stage28_run_id).strip()
    if direct:
        return direct
    stage38 = _load_json(Path(args.docs_dir) / "stage38_master_summary.json")
    stage28_run_id = str(stage38.get("stage28_run_id", "")).strip()
    if stage28_run_id:
        return stage28_run_id
    stage37 = _load_json(Path(args.docs_dir) / "stage37_activation_hunt_summary.json")
    return str(stage37.get("stage28_run_id", "")).strip()


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage-39 Signal Generation Report",
        "",
        "## Run Context",
        f"- stage28_run_id: `{payload.get('stage28_run_id', '')}`",
        f"- seed: `{int(payload.get('seed', 0))}`",
        "",
        "## Layered Flow",
        f"- raw_candidate_count: `{int(payload.get('raw_candidate_count', 0))}`",
        f"- light_pruned_count: `{int(payload.get('light_pruned_count', 0))}`",
        f"- shortlisted_count: `{int(payload.get('shortlisted_count', 0))}`",
        "",
        "## Before vs After",
        f"- baseline_engine_raw_signal_count: `{int(payload.get('baseline_engine_raw_signal_count', 0))}`",
        f"- upgraded_raw_candidate_count: `{int(payload.get('raw_candidate_count', 0))}`",
        "",
        "## Non-Zero Grammar Branches",
    ]
    branches = list(payload.get("nonzero_branches", []))
    if branches:
        lines.extend([f"- {item}" for item in branches])
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Family Counts (Layer C)",
        ]
    )
    family = dict(payload.get("family_counts_layer_c", {}))
    if family:
        for key, value in sorted(family.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
            lines.append(f"- {key}: {int(value)}")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Context Counts (Layer C)",
        ]
    )
    contexts = dict(payload.get("context_counts_layer_c", {}))
    if contexts:
        for key, value in sorted(contexts.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
            lines.append(f"- {key}: {int(value)}")
    else:
        lines.append("- none")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    stage28_run_id = _resolve_stage28_run_id(args)
    if not stage28_run_id:
        raise SystemExit("unable to resolve stage28_run_id for Stage-39")

    stage28_dir = Path(args.runs_dir) / stage28_run_id / "stage28"
    finalists_path = stage28_dir / "finalists_stageC.csv"
    if not finalists_path.exists():
        raise SystemExit(f"missing finalists file: {finalists_path}")
    finalists = pd.read_csv(finalists_path)

    output = build_layered_candidates(finalists, seed=int(args.seed))
    summary = summarize_layered_candidates(output)
    stage38 = _load_json(docs_dir / "stage38_master_summary.json")
    baseline_raw = int(((stage38.get("lineage_table", {}) or {}).get("engine_raw_signal_count", 0)))
    summary.update(
        {
            "stage": "39",
            "seed": int(args.seed),
            "stage28_run_id": stage28_run_id,
            "baseline_engine_raw_signal_count": baseline_raw,
        }
    )

    out_dir = Path(args.runs_dir) / stage28_run_id / "stage39"
    out_dir.mkdir(parents=True, exist_ok=True)
    output.layer_a.to_csv(out_dir / "layer_a_candidates.csv", index=False)
    output.layer_b.to_csv(out_dir / "layer_b_candidates.csv", index=False)
    output.layer_c.to_csv(out_dir / "layer_c_candidates.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    report_md = docs_dir / "stage39_signal_generation_report.md"
    report_json = docs_dir / "stage39_signal_generation_summary.json"
    report_md.write_text(_render_report(summary), encoding="utf-8")
    report_json.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    print(f"stage28_run_id: {stage28_run_id}")
    print(f"raw_candidate_count: {summary['raw_candidate_count']}")
    print(f"shortlisted_count: {summary['shortlisted_count']}")
    print(f"report_md: {report_md}")
    print(f"report_json: {report_json}")


if __name__ == "__main__":
    main()
