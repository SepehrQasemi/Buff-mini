"""Run Stage-80 layered robustness funnel summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.research.robustness import summarize_layered_robustness
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-80 layered robustness summary")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(Path(args.config))

    stage67 = _load_json(docs_dir / "stage67_summary.json")
    replay = _load_json(Path(stage67.get("replay_artifact_path", ""))) if str(stage67.get("replay_artifact_path", "")).strip() else {}
    walkforward = _load_json(Path(stage67.get("walkforward_artifact_path", ""))) if str(stage67.get("walkforward_artifact_path", "")).strip() else {}
    monte_carlo = _load_json(Path(stage67.get("monte_carlo_artifact_path", ""))) if str(stage67.get("monte_carlo_artifact_path", "")).strip() else {}
    perturb = _load_json(Path(stage67.get("cross_perturbation_artifact_path", ""))) if str(stage67.get("cross_perturbation_artifact_path", "")).strip() else {}
    split = _load_json(Path(stage67.get("split_perturbation_artifact_path", ""))) if str(stage67.get("split_perturbation_artifact_path", "")).strip() else {}

    layered = summarize_layered_robustness(
        replay_metrics=replay,
        walkforward_summary=dict(walkforward),
        monte_carlo=monte_carlo,
        perturbation=perturb,
        split_perturbation=split,
        config=cfg,
    )
    status = "SUCCESS" if int(layered.get("level_reached", 0)) >= 1 else "PARTIAL"
    summary = {
        "stage": "80",
        "status": status,
        "execution_status": "EXECUTED",
        "stage_role": "real_validation",
        "validation_state": "LAYERED_ROBUSTNESS_READY" if int(layered.get("level_reached", 0)) >= 1 else "ROBUSTNESS_NOT_ESTABLISHED",
        **layered,
    }
    summary["summary_hash"] = stable_hash(summary, length=16)
    (docs_dir / "stage80_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        "# Stage-80 Report",
        "",
        f"- status: `{summary['status']}`",
        f"- execution_status: `{summary['execution_status']}`",
        f"- stage_role: `{summary['stage_role']}`",
        f"- validation_state: `{summary['validation_state']}`",
        f"- level_reached: `{summary['level_reached']}`",
        f"- level_name: `{summary['level_name']}`",
        f"- stop_reason: `{summary['stop_reason']}`",
        "",
        "## Levels",
    ]
    for key, value in sorted((summary.get("levels") or {}).items()):
        lines.append(f"- {key}: `{bool(value)}`")
    lines.extend(["", "## Forward Metrics"])
    for key, value in sorted((summary.get("forward_metrics") or {}).items()):
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Stress Results"])
    stress = summary.get("stress_results") or {}
    lines.append(f"- monte_carlo_passed: `{bool(stress.get('monte_carlo_passed', False))}`")
    lines.append(f"- cross_perturbation_passed: `{bool(stress.get('cross_perturbation_passed', False))}`")
    lines.append(f"- split_perturbation_passed: `{bool(stress.get('split_perturbation_passed', False))}`")
    lines.append("")
    lines.append(f"- summary_hash: `{summary['summary_hash']}`")
    (docs_dir / "stage80_report.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
