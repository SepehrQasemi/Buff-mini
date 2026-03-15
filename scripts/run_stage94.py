"""Run Stage-94 first interpretable edge inventory campaign."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, PROJECT_ROOT
from buffmini.research.campaign import classify_campaign_outcome, evaluate_scope_campaign, select_campaign_families
from buffmini.research.reporting import markdown_rows, write_stage_artifacts
from buffmini.research.transfer import build_transfer_intelligence, discover_transfer_symbols
from buffmini.utils.hashing import stable_hash


TIMEFRAMES = ("1h", "4h")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-94 first interpretable edge inventory campaign")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--max-candidates-per-scope", type=int, default=6)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _snapshot_end(config: dict[str, Any], *, symbol: str, timeframe: str) -> str:
    snapshot_cfg = dict((config.get("data", {}) or {}).get("snapshot", {}))
    path = Path(snapshot_cfg.get("path", PROJECT_ROOT / "data" / "snapshots" / "DATA_FROZEN_v1.json"))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return ""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(dict(dict(payload.get("per_symbol_per_tf", {})).get(str(symbol), {})).get(str(timeframe), {}).get("end_ts", ""))


def _campaign_label(edge_rows: list[dict[str, Any]], blocked_rows: list[dict[str, Any]]) -> str:
    if not edge_rows and blocked_rows:
        return "data_blocked_or_scope_blocked"
    robust = sum(1 for row in edge_rows if str(row.get("final_class", "")) == "robust_candidate")
    promising = sum(1 for row in edge_rows if str(row.get("final_class", "")) == "promising_but_unproven")
    if robust > 0:
        return "robust_candidate_found"
    if promising > 0:
        return "weak_promising_signs"
    return "no_edge_found"


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    cfg = load_config(Path(args.config))
    feedback = _load_json(docs_dir / "stage93_search_feedback.json")
    families = select_campaign_families(feedback, limit=5)
    symbols = discover_transfer_symbols(cfg)[:2]
    edge_rows: list[dict[str, Any]] = []
    blocked_rows: list[dict[str, Any]] = []
    for symbol in symbols:
        for timeframe in TIMEFRAMES:
            scenario_cfg = deepcopy(cfg)
            scenario_cfg.setdefault("research_run", {})["data_source"] = "canonical"
            end_ts = _snapshot_end(cfg, symbol=symbol, timeframe=timeframe)
            if end_ts:
                scenario_cfg.setdefault("universe", {})["resolved_end_ts"] = end_ts
            summary = evaluate_scope_campaign(
                config=scenario_cfg,
                symbol=symbol,
                timeframe=timeframe,
                families=families,
                candidate_limit=int(args.max_candidates_per_scope),
                requested_mode="evaluation",
                auto_pin_resolved_end=False,
                relax_continuity=False,
            )
            if bool(summary.get("blocked", False)) or int(summary.get("blocked_count", 0)) > 0:
                blocked_rows.append(
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "blocked_reason": str(summary.get("blocked_reason", "")),
                        "mode_summary": dict(summary.get("mode_summary", {})),
                    }
                )
                continue
            for row in summary.get("evaluations", []):
                edge_rows.append(
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        **row,
                    }
                )
    transfer_intelligence = build_transfer_intelligence(
        [
            {
                "classification": row.get("transfer_classification", "not_transferable"),
                "diagnostics": row.get("transfer_diagnostics", []),
                "expected_regime": row.get("expected_regime", "unknown"),
            }
            for row in edge_rows
        ]
    )
    candidate_class_counts = {
        "robust_candidate": int(sum(1 for row in edge_rows if str(row.get("final_class", "")) == "robust_candidate")),
        "promising_but_unproven": int(sum(1 for row in edge_rows if str(row.get("final_class", "")) == "promising_but_unproven")),
        "rejected": int(sum(1 for row in edge_rows if str(row.get("final_class", "")) == "rejected")),
    }
    mechanism_inventory = {}
    regime_map = {}
    failure_map = {}
    for row in edge_rows:
        family = str(row.get("family", ""))
        regime = str(row.get("expected_regime", "unknown"))
        mechanism_inventory[family] = mechanism_inventory.get(family, 0) + 1
        regime_map[regime] = regime_map.get(regime, 0) + 1
        reason = str(row.get("death_reason", "")).strip()
        if reason:
            failure_map[reason] = failure_map.get(reason, 0) + 1
    for row in blocked_rows:
        reason = str(row.get("blocked_reason", "")).strip() or "blocked"
        failure_map[f"blocked::{reason}"] = failure_map.get(f"blocked::{reason}", 0) + 1
    campaign_outcome = _campaign_label(edge_rows, blocked_rows)
    summary = {
        "stage": "94",
        "status": "SUCCESS" if edge_rows or blocked_rows else "PARTIAL",
        "execution_status": "EXECUTED",
        "stage_role": "real_validation",
        "validation_state": "EDGE_INVENTORY_READY" if edge_rows or blocked_rows else "EDGE_INVENTORY_INCOMPLETE",
        "symbols": symbols,
        "timeframes": list(TIMEFRAMES),
        "mechanism_families": families,
        "edge_inventory": edge_rows,
        "blocked_scope_rows": blocked_rows,
        "mechanism_inventory": mechanism_inventory,
        "regime_map": regime_map,
        "transfer_map": transfer_intelligence,
        "failure_map": failure_map,
        "candidate_class_counts": candidate_class_counts,
        "campaign_outcome": campaign_outcome,
        "classification": classify_campaign_outcome(
            edge_inventory=edge_rows,
            evaluated_assets=len({str(row.get("symbol", "")) for row in edge_rows}),
            blocked_assets=len({str(row.get("symbol", "")) for row in blocked_rows}),
        ),
    }
    summary["summary_hash"] = stable_hash(summary, length=16)
    lines = [
        "# Stage-94 Report",
        "",
        f"- status: `{summary['status']}`",
        f"- validation_state: `{summary['validation_state']}`",
        f"- symbols: `{summary['symbols']}`",
        f"- timeframes: `{summary['timeframes']}`",
        f"- mechanism_families: `{summary['mechanism_families']}`",
        f"- campaign_outcome: `{summary['campaign_outcome']}`",
        f"- classification: `{summary['classification']}`",
    ]
    lines.extend([""] + markdown_rows("Candidate Class Counts", [{"class": key, "count": value} for key, value in candidate_class_counts.items()], limit=8))
    lines.extend([""] + markdown_rows("Mechanism Inventory", [{"family": key, "count": value} for key, value in mechanism_inventory.items()], limit=12))
    lines.extend([""] + markdown_rows("Regime Map", [{"regime": key, "count": value} for key, value in regime_map.items()], limit=12))
    lines.extend([""] + markdown_rows("Blocked Scope Rows", blocked_rows, limit=12))
    lines.extend([""] + markdown_rows("Transfer Map", [transfer_intelligence], limit=1))
    lines.extend([""] + markdown_rows("Failure Map", [{"failure": key, "count": value} for key, value in failure_map.items()], limit=20))
    lines.extend(["", f"- summary_hash: `{summary['summary_hash']}`"])
    write_stage_artifacts(docs_dir=docs_dir, stage="94", summary=summary, report_lines=lines)
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
