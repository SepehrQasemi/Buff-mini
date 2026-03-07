"""Stage-49 self-learning 3.0 runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from buffmini.constants import RUNS_DIR
from buffmini.stage49.self_learning3 import (
    deterministic_elites,
    expand_registry_rows_v3,
    failure_aware_mutation,
    family_module_downweighting,
    learning_depth_assessment,
)
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-49 self-learning 3.0")
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


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [dict(item) for item in payload if isinstance(item, dict)]


def _resolve_stage28_run_id(args: argparse.Namespace, docs_dir: Path) -> str:
    if str(args.stage28_run_id).strip():
        return str(args.stage28_run_id).strip()
    stage39 = _load_json(docs_dir / "stage39_signal_generation_summary.json")
    return str(stage39.get("stage28_run_id", "")).strip()


def _render(payload: dict[str, Any], *, notes: list[str]) -> str:
    lines = [
        "# Stage-49 Self-Learning 3.0 Report",
        "",
        f"- status: `{payload.get('status', '')}`",
        f"- registry_rows: `{int(payload.get('registry_rows', 0))}`",
        f"- elites_count: `{int(payload.get('elites_count', 0))}`",
        f"- family_weights_present: `{bool(payload.get('family_weights_present', False))}`",
        f"- mutate_next: `{payload.get('mutate_next', '')}`",
        f"- learning_depth_assessment: `{payload.get('learning_depth_assessment', '')}`",
        "",
        "## Recurring Motifs",
    ]
    motifs = dict(payload.get("recurring_motifs", {}))
    if motifs:
        for key, value in sorted(motifs.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
            lines.append(f"- {key}: {int(value)}")
    else:
        lines.append("- none")
    if notes:
        lines.extend(["", "## Partial Notes"])
        lines.extend([f"- {note}" for note in notes])
    lines.extend(["", f"- summary_hash: `{payload.get('summary_hash', '')}`"])
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    stage28_run_id = _resolve_stage28_run_id(args, docs_dir)
    if not stage28_run_id:
        raise SystemExit("unable to resolve stage28_run_id for Stage-49")

    stage47 = _load_json(docs_dir / "stage47_signal_gen2_summary.json")
    stage48 = _load_json(docs_dir / "stage48_tradability_learning_summary.json")
    base_rows = _load_json_list(Path(args.runs_dir) / stage28_run_id / "stage37" / "learning_registry.json")
    stage47_counts = {str(k): int(v) for k, v in dict(stage47.get("setup_family_counts", {})).items()}
    rows = expand_registry_rows_v3(
        base_rows=base_rows,
        seed=int(args.seed),
        run_id=stage28_run_id,
        stage47_counts=stage47_counts,
        stage48=stage48,
    )
    for row in rows:
        row["mutation_guidance"] = failure_aware_mutation(row)

    elites = deterministic_elites(rows, top_k=5)
    family_weights = family_module_downweighting(rows)
    motifs: dict[str, int] = {}
    for row in rows:
        for motif in row.get("failure_motif_tags", []):
            key = str(motif)
            motifs[key] = int(motifs.get(key, 0) + 1)
    mutate_next = ""
    if rows:
        mutate_next = str(rows[0].get("mutation_guidance", "explore_local_variants"))
    depth = learning_depth_assessment(rows, family_weights=family_weights)

    status = "SUCCESS"
    notes: list[str] = []
    if not rows:
        status = "PARTIAL"
        notes.append("No registry rows were produced.")
    if not motifs:
        status = "PARTIAL"
        notes.append("No recurring motifs were captured.")

    payload = {
        "stage": "49",
        "status": status,
        "registry_rows": int(len(rows)),
        "elites_count": int(len(elites)),
        "recurring_motifs": motifs,
        "family_weights_present": bool(family_weights),
        "mutate_next": mutate_next,
        "learning_depth_assessment": depth,
    }
    payload["summary_hash"] = stable_hash(
        {
            "stage": payload["stage"],
            "status": payload["status"],
            "registry_rows": payload["registry_rows"],
            "elites_count": payload["elites_count"],
            "recurring_motifs": payload["recurring_motifs"],
            "family_weights_present": payload["family_weights_present"],
            "mutate_next": payload["mutate_next"],
            "learning_depth_assessment": payload["learning_depth_assessment"],
        },
        length=16,
    )

    out_dir = Path(args.runs_dir) / stage28_run_id / "stage49"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "learning_registry_v3.json").write_text(json.dumps(rows, indent=2, allow_nan=False), encoding="utf-8")
    (out_dir / "learning_elites_v3.json").write_text(json.dumps(elites, indent=2, allow_nan=False), encoding="utf-8")
    (out_dir / "family_weights_v3.json").write_text(json.dumps(family_weights, indent=2, allow_nan=False), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    summary_path = docs_dir / "stage49_self_learning3_summary.json"
    report_path = docs_dir / "stage49_self_learning3_report.md"
    summary_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(_render(payload, notes=notes), encoding="utf-8")

    print(f"status: {payload['status']}")
    print(f"summary_hash: {payload['summary_hash']}")
    print(f"report: {report_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()

