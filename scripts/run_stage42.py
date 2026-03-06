"""Stage-42 self-learning 2.0 runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from buffmini.constants import RUNS_DIR
from buffmini.stage42.self_learning2 import (
    build_self_diagnosis,
    expand_registry_rows,
    family_memory_summary,
    stability_aware_feature_pruning,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-42 self-learning 2.0")
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


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [dict(item) for item in payload if isinstance(item, dict)]


def _resolve_stage28_run_id(args: argparse.Namespace, docs_dir: Path) -> str:
    if str(args.stage28_run_id).strip():
        return str(args.stage28_run_id).strip()
    stage41 = _load_json(docs_dir / "stage41_derivatives_completion_summary.json")
    if str(stage41.get("stage28_run_id", "")).strip():
        return str(stage41["stage28_run_id"]).strip()
    stage40 = _load_json(docs_dir / "stage40_tradability_objective_summary.json")
    return str(stage40.get("stage28_run_id", "")).strip()


def _render(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage-42 Self-Learning 2.0 Report",
        "",
        "## Registry Expansion",
        f"- stage28_run_id: `{payload.get('stage28_run_id', '')}`",
        f"- registry_rows_v2: `{int(payload.get('registry_rows_v2', 0))}`",
        f"- elite_count: `{int(payload.get('elite_count', 0))}`",
        "",
        "## Family Memory",
        "### Family Weights",
    ]
    weights = dict((payload.get("family_memory", {}) or {}).get("family_weights", {})
)
    if weights:
        for key, value in sorted(weights.items(), key=lambda kv: str(kv[0])):
            lines.append(f"- {key}: {float(value):.6f}")
    else:
        lines.append("- none")
    lines.extend(["", "### Recurring Failure Motifs"])
    motifs = dict((payload.get("family_memory", {}) or {}).get("recurring_failure_motifs", {})
)
    if motifs:
        for key, value in list(motifs.items())[:12]:
            lines.append(f"- {key}: {int(value)}")
    else:
        lines.append("- none")

    prune = dict(payload.get("stability_pruning", {}))
    lines.extend(
        [
            "",
            "## Stability-Aware Pruning",
            f"- kept_features: `{len(prune.get('kept_features', []))}`",
            f"- dropped_features: `{len(prune.get('dropped_features', []))}`",
            "",
            "## Self Diagnosis",
            f"- improved: `{payload.get('self_diagnosis', {}).get('improved', [])}`",
            f"- regressed: `{payload.get('self_diagnosis', {}).get('regressed', [])}`",
            f"- mutate_next: `{payload.get('self_diagnosis', {}).get('mutate_next', '')}`",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    stage28_run_id = _resolve_stage28_run_id(args, docs_dir)
    if not stage28_run_id:
        raise SystemExit("unable to resolve stage28_run_id for Stage-42")

    registry_path = Path(args.runs_dir) / stage28_run_id / "stage37" / "learning_registry.json"
    base_rows = _load_json_list(registry_path)
    stage39 = _load_json(docs_dir / "stage39_signal_generation_summary.json")
    stage40 = _load_json(docs_dir / "stage40_tradability_objective_summary.json")
    stage41 = _load_json(docs_dir / "stage41_derivatives_completion_summary.json")

    expanded = expand_registry_rows(
        base_rows,
        seed=int(args.seed),
        raw_candidate_count=int(stage39.get("raw_candidate_count", 0)),
        shortlisted_count=int(stage39.get("shortlisted_count", 0)),
        mutation_origin="stage42_self_learning_v2",
    )
    family_memory = family_memory_summary(expanded, top_k=5)

    history_rows: list[dict[str, Any]] = []
    for row in stage41.get("family_contributions", []):
        history_rows.append(
            {
                "run_id": stage28_run_id,
                "feature": str(row.get("family", "")),
                "gain": float(row.get("final_policy_share", 0.0)) - float(row.get("candidate_lift", 0.0)),
            }
        )
    stability = stability_aware_feature_pruning(history_rows, min_runs=1, min_mean_contribution=0.0)

    global_mutation_guidance = "widen_context_and_expand_grammar"
    if expanded:
        guides = [str(row.get("mutation_guidance", "")) for row in expanded if str(row.get("mutation_guidance", "")).strip()]
        if guides:
            global_mutation_guidance = sorted(guides)[0]

    current_state = {
        "raw_candidate_count": int(stage39.get("raw_candidate_count", 0)),
        "counts": dict(stage40.get("counts", {})),
        "family_memory": family_memory,
        "global_mutation_guidance": global_mutation_guidance,
    }
    self_diag = build_self_diagnosis(previous={}, current=current_state)

    payload = {
        "stage": "42",
        "seed": int(args.seed),
        "stage28_run_id": stage28_run_id,
        "registry_rows_v2": int(len(expanded)),
        "elite_count": int(sum(1 for row in expanded if bool(row.get("elite", False)))),
        "family_memory": family_memory,
        "stability_pruning": stability,
        "self_diagnosis": self_diag,
    }

    out_dir = Path(args.runs_dir) / stage28_run_id / "stage42"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "learning_registry_v2.json").write_text(json.dumps(expanded, indent=2, allow_nan=False), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    report_md = docs_dir / "stage42_self_learning_2_report.md"
    report_json = docs_dir / "stage42_self_learning_2_summary.json"
    report_md.write_text(_render(payload), encoding="utf-8")
    report_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    print(f"stage28_run_id: {stage28_run_id}")
    print(f"registry_rows_v2: {payload['registry_rows_v2']}")
    print(f"report_md: {report_md}")
    print(f"report_json: {report_json}")


if __name__ == "__main__":
    main()

