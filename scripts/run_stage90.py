"""Run Stage-90 evaluation campaign rerun comparison."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, PROJECT_ROOT
from buffmini.research.campaign import evaluate_scope_campaign, select_campaign_families
from buffmini.research.reporting import markdown_rows, write_stage_artifacts
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-90 evaluation campaign rerun comparison")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _load_snapshot_end(config: dict[str, Any], *, symbol: str, timeframe: str) -> str:
    snapshot_cfg = dict((config.get("data", {}) or {}).get("snapshot", {}))
    path = Path(snapshot_cfg.get("path", PROJECT_ROOT / "data" / "snapshots" / "DATA_FROZEN_v1.json"))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    row = dict(dict(payload.get("per_symbol_per_tf", {})).get(str(symbol), {})).get(str(timeframe), {})
    return str(dict(row).get("end_ts", ""))


def _scenario_outcome(payload: dict[str, Any]) -> str:
    if bool(payload.get("blocked", False)) or int(payload.get("blocked_count", 0)) > 0:
        return "blocked"
    if int(payload.get("robust_count", 0)) > 0:
        return "robust"
    if int(payload.get("validated_count", 0)) > 0:
        return "validated"
    if int(payload.get("promising_count", 0)) > 0:
        return "promising"
    return "no_edge"


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    docs_dir = Path(args.docs_dir)
    feedback = _load_json(docs_dir / "stage82_search_feedback.json")
    families = select_campaign_families(feedback, limit=5)
    symbol = "BTC/USDT"
    timeframe = "1h"

    live_relaxed = evaluate_scope_campaign(
        config=cfg,
        symbol=symbol,
        timeframe=timeframe,
        families=families,
        candidate_limit=10,
        requested_mode="exploration",
        auto_pin_resolved_end=False,
        relax_continuity=True,
    )
    live_strict = evaluate_scope_campaign(
        config=cfg,
        symbol=symbol,
        timeframe=timeframe,
        families=families,
        candidate_limit=10,
        requested_mode="evaluation",
        auto_pin_resolved_end=True,
        relax_continuity=False,
    )
    canonical_cfg = deepcopy(cfg)
    canonical_cfg.setdefault("research_run", {})["data_source"] = "canonical"
    canonical_end = _load_snapshot_end(cfg, symbol=symbol, timeframe=timeframe)
    if canonical_end:
        canonical_cfg.setdefault("universe", {})["resolved_end_ts"] = canonical_end
    canonical_snapshot = evaluate_scope_campaign(
        config=canonical_cfg,
        symbol=symbol,
        timeframe=timeframe,
        families=families,
        candidate_limit=10,
        requested_mode="evaluation",
        auto_pin_resolved_end=False,
        relax_continuity=False,
    )
    matrix = []
    for name, payload in (
        ("live_relaxed", live_relaxed),
        ("live_strict", live_strict),
        ("canonical_snapshot", canonical_snapshot),
    ):
        matrix.append(
            {
                "environment": name,
                "candidate_count": int(payload.get("candidate_count", 0)),
                "promising_count": int(payload.get("promising_count", 0)),
                "validated_count": int(payload.get("validated_count", 0)),
                "robust_count": int(payload.get("robust_count", 0)),
                "blocked_count": int(payload.get("blocked_count", 0)),
                "blocked_reason": str(payload.get("blocked_reason", "")),
                "dominant_failure_reasons": dict(payload.get("dominant_failure_reasons", {})),
                "campaign_outcome": _scenario_outcome(payload),
            }
        )
    if _scenario_outcome(live_strict) == "blocked" and _scenario_outcome(canonical_snapshot) in {"promising", "validated", "robust", "no_edge"}:
        dominant_culprit = "data_canonicalization"
    elif _scenario_outcome(live_relaxed) == "promising" and _scenario_outcome(canonical_snapshot) in {"no_edge", "blocked"}:
        dominant_culprit = "evaluation_strictness_or_search_weakness"
    else:
        dominant_culprit = "search_or_ranking_limits"
    summary = {
        "stage": "90",
        "status": "SUCCESS",
        "execution_status": "EXECUTED",
        "stage_role": "real_validation",
        "validation_state": "CAMPAIGN_COMPARISON_READY",
        "symbol": symbol,
        "timeframe": timeframe,
        "families": families,
        "comparison_matrix": matrix,
        "dominant_culprit": dominant_culprit,
    }
    summary["summary_hash"] = stable_hash(summary, length=16)
    lines = [
        "# Stage-90 Report",
        "",
        f"- status: `{summary['status']}`",
        f"- symbol: `{summary['symbol']}`",
        f"- timeframe: `{summary['timeframe']}`",
        f"- families: `{summary['families']}`",
        f"- dominant_culprit: `{summary['dominant_culprit']}`",
    ]
    lines.extend([""] + markdown_rows("Campaign Comparison Matrix", matrix, limit=6))
    lines.extend(["", f"- summary_hash: `{summary['summary_hash']}`"])
    write_stage_artifacts(docs_dir=docs_dir, stage="90", summary=summary, report_lines=lines)
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
