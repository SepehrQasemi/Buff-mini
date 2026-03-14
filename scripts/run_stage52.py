"""Run Stage-52 setup candidate v2 bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.stage47.genesis import SETUP_FAMILIES
from buffmini.stage51 import resolve_research_scope
from buffmini.stage52 import build_setup_candidate_v2, evaluate_family_coverage, summarize_stage52_candidates
from buffmini.stage52 import deduplicate_setup_candidates_by_economics
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-52 setup candidate v2 bootstrap")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--stage28-run-id", type=str, default="")
    return parser.parse_args()


def _cost_pct(value: float) -> float:
    return float(value / 100.0) if float(value) > 0.02 else float(value)


def _normalize_family(raw_family: str) -> str:
    family = str(raw_family).strip()
    aliases = {
        "regime_shift_entry": "structure_pullback_continuation",
    }
    return str(aliases.get(family, family))


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _resolve_stage28_run_id(args: argparse.Namespace, docs_dir: Path) -> str:
    if str(args.stage28_run_id).strip():
        return str(args.stage28_run_id).strip()
    stage39 = _load_json(docs_dir / "stage39_signal_generation_summary.json")
    return str(stage39.get("stage28_run_id", "")).strip()


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    scope = resolve_research_scope(cfg)
    stage28_run_id = _resolve_stage28_run_id(args, docs_dir)
    shortlist = pd.DataFrame()
    if stage28_run_id:
        shortlist_path = Path(args.runs_dir) / stage28_run_id / "stage47" / "setup_shortlist.csv"
        if shortlist_path.exists():
            shortlist = pd.read_csv(shortlist_path)
    family_lookup = {str(row["family"]): dict(row) for row in SETUP_FAMILIES}
    rows: list[dict] = []
    input_mode = "bootstrap_templates"
    if not shortlist.empty:
        input_mode = "stage47_shortlist"
        if "family" in shortlist.columns:
            shortlist["family"] = shortlist["family"].astype(str).map(_normalize_family)
            shortlist = shortlist.loc[shortlist["family"].astype(str).isin(scope["active_setup_families"]), :].copy()
        else:
            shortlist = pd.DataFrame()
        for tf_idx, timeframe in enumerate(scope["discovery_timeframes"]):
            for rec in shortlist.to_dict(orient="records"):
                rows.append(
                    {
                        "candidate_id": str(rec.get("candidate_id", "")),
                        "family": str(rec.get("family", "")),
                        "timeframe": str(timeframe),
                        "context": str(rec.get("context", "range")),
                        "trigger": str(rec.get("trigger", "")),
                        "confirmation": str(rec.get("confirmation", "")),
                        "invalidation": str(rec.get("invalidation", "")),
                        "modules": list(rec.get("modules", [])) if isinstance(rec.get("modules", []), list) else [],
                        "beam_score": float(rec.get("beam_score", 0.0)),
                        "source_branch": str(rec.get("source_branch", "stage47_shortlist")),
                    }
                )
    else:
        for tf_idx, timeframe in enumerate(scope["discovery_timeframes"]):
            for fam_idx, family in enumerate(scope["active_setup_families"]):
                meta = family_lookup[str(family)]
                rows.append(
                    {
                        "candidate_id": f"stage52_seed_{family}_{timeframe}",
                        "family": str(family),
                        "timeframe": str(timeframe),
                        "context": str(meta["contexts"][0]),
                        "trigger": str(meta["trigger"]),
                        "confirmation": str(meta["confirmation"]),
                        "invalidation": str(meta["invalidation"]),
                        "modules": [str(v) for v in meta.get("modules", ())],
                        "beam_score": round(0.55 + (fam_idx * 0.08) + (tf_idx * 0.03), 6),
                        "source_branch": "stage52_bootstrap",
                    }
                )
    base = pd.DataFrame(rows)
    upgraded_raw = pd.DataFrame(
        [
            build_setup_candidate_v2(
                row,
                timeframe=str(row["timeframe"]),
                round_trip_cost_pct=_cost_pct(float(cfg.get("costs", {}).get("round_trip_cost_pct", 0.1))),
            )
            for row in base.loc[base["family"].isin(scope["active_setup_families"]), :].to_dict(orient="records")
        ]
    )
    upgraded = deduplicate_setup_candidates_by_economics(upgraded_raw, keep_per_fingerprint=1)
    summary = summarize_stage52_candidates(upgraded)
    coverage = evaluate_family_coverage(summary, active_families=[str(v) for v in scope["active_setup_families"]])
    if not bool(coverage["family_coverage_ok"]):
        summary["status"] = "PARTIAL"
        summary["blocker_reason"] = "missing_active_families"
    summary.update(
        {
            "stage": "52",
            "input_mode": input_mode,
            "stage28_run_id": stage28_run_id,
            "raw_candidate_count": int(len(upgraded_raw)),
            "deduplicated_candidate_count": int(len(upgraded)),
            "dedup_drop_count": int(max(0, len(upgraded_raw) - len(upgraded))),
            "active_setup_families": list(scope["active_setup_families"]),
            "discovery_timeframes": list(scope["discovery_timeframes"]),
            "family_coverage_ok": bool(coverage["family_coverage_ok"]),
            "missing_families": [str(v) for v in coverage["missing_families"]],
            "representative_candidate_id": str(upgraded.iloc[0]["candidate_id"]) if not upgraded.empty else "",
        }
    )
    summary["summary_hash"] = stable_hash(
        {
            "status": summary["status"],
            "candidate_count": summary["candidate_count"],
            "eligible_for_replay_count": summary["eligible_for_replay_count"],
            "family_counts": summary["family_counts"],
            "timeframe_counts": summary["timeframe_counts"],
            "rejection_counts": summary["rejection_counts"],
            "avg_cost_edge_proxy": summary["avg_cost_edge_proxy"],
            "active_setup_families": summary["active_setup_families"],
            "family_coverage_ok": summary["family_coverage_ok"],
            "missing_families": summary["missing_families"],
            "stage28_run_id": summary["stage28_run_id"],
            "input_mode": summary["input_mode"],
            "raw_candidate_count": summary["raw_candidate_count"],
            "deduplicated_candidate_count": summary["deduplicated_candidate_count"],
            "dedup_drop_count": summary["dedup_drop_count"],
            "representative_candidate_id": summary["representative_candidate_id"],
            "blocker_reason": str(summary.get("blocker_reason", "")),
        },
        length=16,
    )
    if stage28_run_id:
        out_dir = Path(args.runs_dir) / stage28_run_id / "stage52"
        out_dir.mkdir(parents=True, exist_ok=True)
        upgraded.to_csv(out_dir / "setup_candidates_v2.csv", index=False)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    summary_path = docs_dir / "stage52_summary.json"
    report_path = docs_dir / "stage52_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(
        "\n".join(
            [
                "# Stage-52 Report",
                "",
                f"- status: `{summary['status']}`",
                f"- input_mode: `{summary['input_mode']}`",
                f"- candidate_count: `{summary['candidate_count']}`",
                f"- raw_candidate_count: `{summary['raw_candidate_count']}`",
                f"- deduplicated_candidate_count: `{summary['deduplicated_candidate_count']}`",
                f"- dedup_drop_count: `{summary['dedup_drop_count']}`",
                f"- eligible_for_replay_count: `{summary['eligible_for_replay_count']}`",
                f"- economic_fingerprint_count: `{summary.get('economic_fingerprint_count', 0)}`",
                f"- representative_candidate_id: `{summary['representative_candidate_id']}`",
                f"- family_coverage_ok: `{summary['family_coverage_ok']}`",
                f"- missing_families: `{summary['missing_families']}`",
                f"- avg_cost_edge_proxy: `{summary['avg_cost_edge_proxy']}`",
                f"- blocker_reason: `{summary.get('blocker_reason', '')}`",
                f"- summary_hash: `{summary['summary_hash']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
