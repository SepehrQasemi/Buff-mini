"""Stage-37.3 failure-aware self-learning registry updater."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.stage37.self_learning import (
    apply_elite_flags,
    compute_family_exploration_weights,
    load_learning_registry,
    prune_features_by_contribution,
    save_learning_registry,
    select_elites_deterministic,
    upsert_learning_registry_entry,
)
from buffmini.stage38.audit import build_failure_aware_registry_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-37 failure-aware self-learning upgrade")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--stage28-run-id", type=str, default="")
    parser.add_argument("--activation-summary", type=Path, default=Path("docs/stage37_activation_hunt_summary.json"))
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_finalists(runs_dir: Path, run_id: str) -> pd.DataFrame:
    path = Path(runs_dir) / str(run_id) / "stage28" / "finalists_stageC.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage-37 Self-Learning Upgrade",
        "",
        "## Registry",
        f"- registry_rows: `{int(payload.get('registry_rows', 0))}`",
        f"- new_rows_added: `{int(payload.get('new_rows_added', 0))}`",
        f"- stage28_run_id: `{payload.get('stage28_run_id', '')}`",
        "",
        "## Family Exploration Weights",
    ]
    weights = dict(payload.get("family_weights", {}))
    if weights:
        for family, weight in sorted(weights.items(), key=lambda kv: str(kv[0])):
            lines.append(f"- {family}: {float(weight):.6f}")
    else:
        lines.append("- none")

    lines.extend(
        [
            "",
            "## Elites",
            "| family | exp_lcb | activation_rate | final_trade_count | top_reject_reason |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
    )
    for row in payload.get("elites", []):
        lines.append(
            "| {family} | {exp:.6f} | {act:.6f} | {trades} | {reason} |".format(
                family=str(row.get("family", "")),
                exp=float(row.get("exp_lcb", 0.0)),
                act=float(row.get("activation_rate", 0.0)),
                trades=int(row.get("final_trade_count", 0)),
                reason=str(row.get("top_reject_reason", "")),
            )
        )

    lines.extend(
        [
            "",
            "## Failure Motifs",
        ]
    )
    motifs = dict(payload.get("failure_motifs", {}))
    if motifs:
        for key, value in sorted(motifs.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
            lines.append(f"- {key}: {int(value)}")
    else:
        lines.append("- none")

    prune = dict(payload.get("feature_pruning", {}))
    lines.extend(
        [
            "",
            "## Feature Pruning",
            f"- kept_features: `{len(prune.get('kept_features', []))}`",
            f"- dropped_features: `{len(prune.get('dropped_features', []))}`",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    stage37_cfg = dict(((cfg.get("evaluation", {}) or {}).get("stage37", {})))
    self_cfg = dict(stage37_cfg.get("self_learning", {}))
    elite_count = int(self_cfg.get("elite_count", 5))

    activation_payload = _load_json(Path(args.activation_summary))
    if not activation_payload:
        raise SystemExit(f"missing activation summary: {args.activation_summary}")

    stage28_run_id = str(args.stage28_run_id).strip() or str(activation_payload.get("stage28_run_id", "")).strip()
    if not stage28_run_id:
        raise SystemExit("stage28_run_id could not be resolved")

    run_id = f"stage37_seed{int(args.seed)}_{stage28_run_id}"
    registry_path = Path(args.runs_dir) / stage28_run_id / "stage37" / "learning_registry.json"
    registry_before = load_learning_registry(registry_path)
    before_rows = len(registry_before)

    generated_rows = build_failure_aware_registry_rows(activation_payload=activation_payload, run_id=run_id)
    for row in generated_rows:
        upsert_learning_registry_entry(registry_path, row)

    registry = load_learning_registry(registry_path)
    family_weights = compute_family_exploration_weights(registry)
    elite_rows = select_elites_deterministic(registry, top_k=elite_count)
    registry = apply_elite_flags(registry, elite_rows)
    save_learning_registry(registry_path, registry)
    elites = [dict(row) for row in registry if bool(row.get("elite", False))]
    if not elites:
        elites = [dict(row) for row in elite_rows]

    failure_motifs: dict[str, int] = {}
    for row in registry:
        tags = list(row.get("failure_motif_tags", []))
        if not tags:
            tags = [f"REJECT::{str(row.get('top_reject_reason', 'unknown')).upper()}"]
        for tag in tags:
            key = str(tag)
            failure_motifs[key] = int(failure_motifs.get(key, 0) + 1)

    finalists = _load_finalists(Path(args.runs_dir), stage28_run_id)
    contrib_rows: list[dict[str, Any]] = []
    if not finalists.empty:
        for item in finalists.to_dict(orient="records"):
            contrib_rows.append({"feature": str(item.get("candidate", "")), "gain": float(item.get("exp_lcb", 0.0))})
    pruning = prune_features_by_contribution(contrib_rows, min_mean_gain=0.0, keep_top=64)

    payload = {
        "stage": "37.3",
        "seed": int(args.seed),
        "stage28_run_id": stage28_run_id,
        "registry_path": str(registry_path),
        "registry_rows": int(len(registry)),
        "new_rows_added": int(len(registry) - before_rows),
        "family_weights": family_weights,
        "elites": elites,
        "dead_family_count": int(sum(1 for row in registry if str(row.get("status", "")) == "dead_end")),
        "failure_motifs": failure_motifs,
        "feature_pruning": pruning,
    }

    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_md = docs_dir / "stage37_self_learning_upgrade.md"
    report_json = docs_dir / "stage37_self_learning_upgrade_summary.json"
    report_md.write_text(_render_report(payload), encoding="utf-8")
    report_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    print(f"registry_path: {registry_path}")
    print(f"report_md: {report_md}")
    print(f"report_json: {report_json}")


if __name__ == "__main__":
    main()
