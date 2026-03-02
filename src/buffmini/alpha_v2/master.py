"""Master summary/report builder for Stage-15..22."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from buffmini.utils.hashing import stable_hash


def build_master_summary(
    *,
    summaries: dict[str, dict[str, Any]],
    seed: int,
    git_head: str | None = None,
    commit_hashes: list[str] | None = None,
) -> dict[str, Any]:
    milestones = {}
    for stage in ("15", "17", "19", "22"):
        payload = dict(summaries.get(stage, {}))
        milestones[stage] = {
            "run_id": payload.get("run_id"),
            "status": payload.get("status", "UNKNOWN"),
            "trade_count": payload.get("trade_count", 0.0),
            "exp_lcb": payload.get("exp_lcb", 0.0),
            "max_drawdown": payload.get("max_drawdown", 0.0),
            "zero_trade_pct": payload.get("zero_trade_pct", 0.0),
            "invalid_pct": payload.get("invalid_pct", 0.0),
        }
    best_stage = max(
        milestones.items(),
        key=lambda kv: float(kv[1].get("exp_lcb", 0.0)),
        default=("15", {"exp_lcb": 0.0}),
    )[0]
    stage_statuses = {stage: str(payload.get("status", "UNKNOWN")) for stage, payload in summaries.items()}
    pass_count = sum(1 for x in stage_statuses.values() if x == "PASS")
    if float(milestones.get(best_stage, {}).get("exp_lcb", 0.0)) > 0.0 and pass_count >= 4:
        verdict = "WEAK_EDGE"
    elif pass_count == 0:
        verdict = "INSUFFICIENT_DATA"
    else:
        verdict = "NO_EDGE"
    payload = {
        "stage": "15_22",
        "seed": int(seed),
        "git_head": str(git_head or ""),
        "commit_hashes": list(commit_hashes or []),
        "run_ids": {stage: summaries.get(stage, {}).get("run_id") for stage in ("15", "16", "17", "18", "19", "20", "21", "22")},
        "stages": summaries,
        "ab_milestones": milestones,
        "best_stage": best_stage,
        "multi_horizon_consistency": {
            "available": "20" in summaries and "horizon_consistency" in summaries.get("20", {}),
            "score": summaries.get("20", {}).get("horizon_consistency"),
        },
        "drag_cost_sensitivity": {
            "stage17_delta_exp_lcb": milestones.get("17", {}).get("exp_lcb", 0.0),
            "stage22_delta_exp_lcb": milestones.get("22", {}).get("exp_lcb", 0.0),
        },
        "final_verdict": verdict,
        "runtime_hash": stable_hash(summaries, length=16),
    }
    return payload


def write_master_report(payload: dict[str, Any]) -> None:
    summary_path = Path("docs/stage15_22_master_summary.json")
    report_path = Path("docs/stage15_22_master_report.md")
    summary_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        "# Stage-15..22 Master Report",
        "",
        "## 1) Per-stage summary",
    ]
    for stage in ("15", "16", "17", "18", "19", "20", "21", "22"):
        s = payload["stages"].get(stage, {})
        lines.append(
            f"- Stage-{stage}: status=`{s.get('status','UNKNOWN')}` run_id=`{s.get('run_id','')}` exp_lcb=`{s.get('exp_lcb',0.0)}`"
        )
    lines.extend(
        [
            "",
            "## 2) A/B milestones",
        ]
    )
    for stage, data in payload["ab_milestones"].items():
        lines.append(
            f"- Stage-{stage}: trade_count=`{data.get('trade_count',0.0)}` exp_lcb=`{data.get('exp_lcb',0.0)}` maxDD=`{data.get('max_drawdown',0.0)}`"
        )
    lines.extend(
        [
            "",
            "## 3) Best configuration",
            f"- best_stage: `{payload.get('best_stage')}`",
            "",
            "## 4) Multi-horizon consistency",
            f"- {payload.get('multi_horizon_consistency')}",
            "",
            "## 5) Drag/cost sensitivity",
            f"- {payload.get('drag_cost_sensitivity')}",
            "",
            "## 6) Final verdict",
            f"- `{payload.get('final_verdict')}`",
        ]
    )
    report_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
