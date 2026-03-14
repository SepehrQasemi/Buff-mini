"""Run Stage-69 self-learning v5 campaign memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.stage69 import build_campaign_memory_rows_v5, derive_campaign_priors_v5
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-69 self-learning v5")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--stage28-run-id", type=str, default="")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _deduplicate_memory_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for row in sorted(
        [dict(item) for item in rows if isinstance(item, dict)],
        key=lambda item: (
            str(item.get("run_id", "")),
            str(item.get("candidate_id", "")),
            str(item.get("mutation_guidance", "")),
        ),
    ):
        key = (
            str(row.get("run_id", "")),
            str(row.get("candidate_id", "")),
            str(row.get("mutation_guidance", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    stage62 = _load_json(docs_dir / "stage62_summary.json")
    stage28_run_id = str(args.stage28_run_id).strip() or str(stage62.get("stage28_run_id", "")).strip()
    if not stage28_run_id:
        raise SystemExit("unable to resolve stage28_run_id for Stage-69")
    base = Path(args.runs_dir) / stage28_run_id
    outcomes = _read_csv(base / "stage62" / "candidate_outcomes_v3.csv")
    gated = _read_csv(base / "stage68" / "gated_candidates.csv")
    rows = build_campaign_memory_rows_v5(
        outcomes=outcomes,
        gated_candidates=gated,
        stage28_run_id=stage28_run_id,
    )
    priors = derive_campaign_priors_v5(rows)
    memory_cfg = dict(cfg.get("campaign_memory", {}))
    memory_path = Path(str(memory_cfg.get("store_path", docs_dir / "stage69_campaign_memory.json")))
    if not memory_path.is_absolute():
        memory_path = Path(memory_path)
    frozen_mode = bool(cfg.get("reproducibility", {}).get("frozen_research_mode", False))
    cold_start_each_run = bool(memory_cfg.get("cold_start_each_run", True)) or frozen_mode
    prior_payload = []
    if memory_path.exists() and not cold_start_each_run:
        loaded = json.loads(memory_path.read_text(encoding="utf-8"))
        prior_payload = loaded if isinstance(loaded, list) else []
    merged_rows = _deduplicate_memory_rows(prior_payload + rows)
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text(json.dumps(merged_rows, indent=2, allow_nan=False), encoding="utf-8")

    status = "SUCCESS" if rows else "PARTIAL"
    summary = {
        "stage": "69",
        "status": status,
        "execution_status": "EXECUTED",
        "stage_role": "reporting_only",
        "validation_state": "REPORTING_ONLY",
        "stage28_run_id": stage28_run_id,
        "new_memory_rows": int(len(rows)),
        "campaign_memory_rows_total": int(len(merged_rows)),
        "frozen_research_mode": bool(frozen_mode),
        "cold_start_each_run_effective": bool(cold_start_each_run),
        "memory_path": str(memory_path),
        "memory_source_rows": int(len(prior_payload)),
        "priors": priors,
        "blocker_reason": "" if status == "SUCCESS" else "no_rows_generated",
    }
    summary["summary_hash"] = stable_hash(summary, length=16)
    (docs_dir / "stage69_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (docs_dir / "stage69_report.md").write_text(
        "\n".join(
            [
                "# Stage-69 Report",
                "",
                f"- status: `{summary['status']}`",
                f"- execution_status: `{summary['execution_status']}`",
                f"- stage_role: `{summary['stage_role']}`",
                f"- validation_state: `{summary['validation_state']}`",
                f"- stage28_run_id: `{summary['stage28_run_id']}`",
                f"- new_memory_rows: `{summary['new_memory_rows']}`",
                f"- campaign_memory_rows_total: `{summary['campaign_memory_rows_total']}`",
                f"- frozen_research_mode: `{summary['frozen_research_mode']}`",
                f"- cold_start_each_run_effective: `{summary['cold_start_each_run_effective']}`",
                f"- memory_path: `{summary['memory_path']}`",
                f"- memory_source_rows: `{summary['memory_source_rows']}`",
                f"- priors: `{summary['priors']}`",
                f"- blocker_reason: `{summary['blocker_reason']}`",
                f"- summary_hash: `{summary['summary_hash']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
