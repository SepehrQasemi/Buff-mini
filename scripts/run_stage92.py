"""Run Stage-92 transfer and regime intelligence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.research.campaign import evaluate_scope_campaign, select_campaign_families
from buffmini.research.reporting import markdown_rows, write_stage_artifacts
from buffmini.research.transfer import build_transfer_intelligence, discover_transfer_symbols
from buffmini.utils.hashing import stable_hash


TIMEFRAMES = ("15m", "30m", "1h", "4h")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-92 transfer and regime intelligence")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--candidate-limit-per-scope", type=int, default=2)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    cfg = load_config(Path(args.config))
    feedback = _load_json(docs_dir / "stage82_search_feedback.json")
    families = select_campaign_families(feedback, limit=6)
    symbols = discover_transfer_symbols(cfg)[:2]
    transfer_rows: list[dict[str, Any]] = []
    for symbol in symbols:
        for timeframe in TIMEFRAMES:
            summary = evaluate_scope_campaign(
                config=cfg,
                symbol=symbol,
                timeframe=timeframe,
                families=families,
                candidate_limit=int(max(1, args.candidate_limit_per_scope)),
                requested_mode="exploration",
                auto_pin_resolved_end=False,
                relax_continuity=True,
            )
            for row in summary.get("evaluations", []):
                transfer_rows.append(
                    {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "candidate_id": str(row.get("candidate_id", "")),
                        "family": str(row.get("family", "")),
                        "expected_regime": str(row.get("expected_regime", "unknown")),
                        "classification": str(row.get("transfer_classification", "not_transferable")),
                        "diagnostics": list(row.get("transfer_diagnostics", [])),
                    }
                )
    intelligence = build_transfer_intelligence(transfer_rows)
    summary = {
        "stage": "92",
        "status": "SUCCESS" if transfer_rows else "PARTIAL",
        "execution_status": "EXECUTED",
        "stage_role": "reporting_only",
        "validation_state": "TRANSFER_INTELLIGENCE_READY" if transfer_rows else "TRANSFER_INTELLIGENCE_INCOMPLETE",
        "symbols": symbols,
        "families": families,
        "candidate_limit_per_scope": int(max(1, args.candidate_limit_per_scope)),
        "transfer_rows": transfer_rows,
        **intelligence,
    }
    summary["summary_hash"] = stable_hash(summary, length=16)
    lines = [
        "# Stage-92 Report",
        "",
        f"- status: `{summary['status']}`",
        f"- symbols: `{summary['symbols']}`",
        f"- families: `{summary['families']}`",
        f"- candidate_limit_per_scope: `{summary['candidate_limit_per_scope']}`",
        f"- transfer_row_count: `{len(transfer_rows)}`",
    ]
    lines.extend([""] + markdown_rows("Transfer Class Counts", [{"class": key, "count": value} for key, value in dict(summary.get("transfer_class_counts", {})).items()]))
    lines.extend([""] + markdown_rows("Regime Portability Map", list(summary.get("regime_portability_map", [])), limit=10))
    lines.extend([""] + markdown_rows("Failure Diagnostics", [{"diagnostic": key, "count": value} for key, value in dict(summary.get("failure_diagnostics", {})).items()], limit=12))
    lines.extend(["", f"- summary_hash: `{summary['summary_hash']}`"])
    write_stage_artifacts(docs_dir=docs_dir, stage="92", summary=summary, report_lines=lines)
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
