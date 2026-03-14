"""Run Stage-73 integrated repair evidence report generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.utils.hashing import stable_hash


PROBLEMS: tuple[tuple[int, str], ...] = (
    (1, "Validation semantics mixed with proxies"),
    (2, "Late-stage orchestration acting like validation theater"),
    (3, "Discovery generator too template-based / weak hypothesis generation"),
    (4, "Economically identical candidates not deduplicated"),
    (5, "Ranking too proxy-heavy and not candidate-specific enough"),
    (6, "Walk-forward validation not sufficiently real / artifact-backed"),
    (7, "Monte Carlo not sufficiently real / artifact-backed"),
    (8, "Cross-seed weak and should become cross-perturbation robustness"),
    (9, "Transfer validation can be synthetic / defaulted"),
    (10, "Important config blocks not fully wired into actual stage behavior"),
    (11, "Funding realism incomplete in core backtest flow"),
    (12, "Position sizing too simplistic / unrealistic"),
    (13, "Reproducibility incomplete"),
    (14, "Campaign memory can break reproducibility"),
    (15, "Data continuity / missing-candle handling not strict enough"),
    (16, "Docs / status semantics misleading"),
    (17, "Performance claims partly projected rather than measured"),
    (18, "UI can create false confidence by hiding provenance / evidence quality"),
    (19, "Tests do not sufficiently validate scientific correctness"),
    (20, "Search space too narrow / regime-limited"),
    (21, "Architecture has parallel stage ecosystems with semantic drift"),
    (22, "SUCCESS status semantics misleading"),
    (23, "Late-stage ML stack can create false confidence"),
    (24, "Metrics edge cases / pathological states can pollute ranking and reports"),
    (25, "Runtime truth not sufficiently proven in a clean verification path"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-73 integrated evidence report")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--compileall-status", type=str, default="NOT_VERIFIED")
    parser.add_argument("--pytest-status", type=str, default="NOT_VERIFIED")
    parser.add_argument("--integration-status", type=str, default="NOT_VERIFIED")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _status_entry(
    *,
    number: int,
    title: str,
    status: str,
    evidence: list[str],
    remaining: str,
) -> dict[str, Any]:
    return {
        "problem_number": int(number),
        "title": str(title),
        "status": str(status),
        "evidence": [str(v) for v in evidence],
        "remaining_limitations": str(remaining),
    }


def _build_problem_statuses(
    *,
    cfg: dict[str, Any],
    docs_dir: Path,
    runs_dir: Path,
) -> list[dict[str, Any]]:
    stage57 = _load_json(docs_dir / "stage57_summary.json")
    stage58 = _load_json(docs_dir / "stage58_summary.json")
    stage55 = _load_json(docs_dir / "stage55_summary.json")
    stage65 = _load_json(docs_dir / "stage65_summary.json")
    stage66 = _load_json(docs_dir / "stage66_summary.json")
    stage70 = _load_json(docs_dir / "stage70_summary.json")
    stage71 = _load_json(docs_dir / "stage71_summary.json")
    stage72 = _load_json(docs_dir / "stage72_summary.json")
    stage60 = _load_json(docs_dir / "stage60_summary.json")
    stage28_run_id = str(stage60.get("stage28_run_id", "")).strip()
    base = runs_dir / stage28_run_id if stage28_run_id else Path("__missing__")
    real_walkforward = bool((base / "stage67" / "walkforward_metrics_real.json").exists())
    real_replay = bool((base / "stage53" / "replay_metrics_real.json").exists())
    real_mc = bool((base / "stage57" / "monte_carlo_metrics_real.json").exists())
    real_cross = bool((base / "stage57" / "cross_perturbation_metrics_real.json").exists())
    decision_allowed = bool(stage57.get("decision_evidence", {}).get("allowed", False))
    strict_real = bool(cfg.get("promotion_gates", {}).get("strict_real_evidence", True))
    frozen_mode = bool(cfg.get("reproducibility", {}).get("frozen_research_mode", False))
    transfer_real = bool(stage58.get("transfer_artifact_exists", False))
    projection_only = bool(stage55.get("projection_only", True))
    continuity_state = str(stage65.get("validation_state", ""))
    ml_reporting_only = bool(stage66.get("decision_use_allowed", True) is False)
    stage70_diversity = bool(stage70.get("diversity_ok", False))
    runtime_measured = bool(stage71.get("projected_only", True) is False)

    statuses: dict[int, dict[str, Any]] = {
        1: _status_entry(
            number=1,
            title=PROBLEMS[0][1],
            status="FULLY_FIXED" if decision_allowed and strict_real else "PARTIALLY_FIXED",
            evidence=[f"stage57 decision_evidence.allowed={decision_allowed}", f"promotion_gates.strict_real_evidence={strict_real}"],
            remaining="Still dependent on upstream real artifacts availability per run.",
        ),
        2: _status_entry(
            number=2,
            title=PROBLEMS[1][1],
            status="PARTIALLY_FIXED",
            evidence=["scripts/run_stage60_72.py reorders decision stages after real-validation artifacts"],
            remaining="Legacy stage scripts still exist and can be executed out of order manually.",
        ),
        3: _status_entry(
            number=3,
            title=PROBLEMS[2][1],
            status="PARTIALLY_FIXED",
            evidence=[f"stage70 diversity_ok={stage70_diversity}", "stage70 uses context/trigger/confirmation/invalidation/exit/time_stop structure"],
            remaining="Generator remains bounded and still heuristic, not open-ended hypothesis synthesis.",
        ),
        4: _status_entry(
            number=4,
            title=PROBLEMS[3][1],
            status="FULLY_FIXED",
            evidence=["stage52 economic_fingerprint + deduplicate_setup_candidates_by_economics", "stage70 deduplicate_economic_candidates"],
            remaining="Near-identical but non-identical fingerprints may still pass.",
        ),
        5: _status_entry(
            number=5,
            title=PROBLEMS[4][1],
            status="PARTIALLY_FIXED",
            evidence=["stage48 ranking now includes candidate-specific rr/cost/exp_lcb/reject penalties"],
            remaining="Some global label priors remain in stage48 score composition.",
        ),
        6: _status_entry(
            number=6,
            title=PROBLEMS[5][1],
            status="FULLY_FIXED" if real_walkforward else "PARTIALLY_FIXED",
            evidence=[f"walkforward_metrics_real_exists={real_walkforward}", "scripts/run_stage67.py writes walkforward_windows_real.csv + metrics JSON"],
            remaining="If stage67 cannot run, fallback evidence remains insufficient.",
        ),
        7: _status_entry(
            number=7,
            title=PROBLEMS[6][1],
            status="PARTIALLY_FIXED" if real_mc else "NOT_FIXED",
            evidence=[f"monte_carlo_metrics_real_exists={real_mc}", "scripts/run_stage67.py writes stage57/monte_carlo_metrics_real.json"],
            remaining="Monte Carlo is still simplified and not a full trade-path simulator.",
        ),
        8: _status_entry(
            number=8,
            title=PROBLEMS[7][1],
            status="PARTIALLY_FIXED" if real_cross else "NOT_FIXED",
            evidence=[f"cross_perturbation_metrics_real_exists={real_cross}", "scripts/run_stage67.py writes cross_perturbation metrics"],
            remaining="Perturbation design is still basic and not exhaustive across market regimes.",
        ),
        9: _status_entry(
            number=9,
            title=PROBLEMS[8][1],
            status="PARTIALLY_FIXED" if transfer_real else "NOT_FIXED",
            evidence=[f"stage58 transfer_artifact_exists={transfer_real}", "stage58 blocks proxy/synthetic transfer evidence"],
            remaining="No guaranteed real transfer dataset is produced in all environments.",
        ),
        10: _status_entry(
            number=10,
            title=PROBLEMS[9][1],
            status="PARTIALLY_FIXED",
            evidence=["stage61/stage67/stage68 now emit used_config_keys/effective_values"],
            remaining="Not all legacy config sections are fully wired end-to-end.",
        ),
        11: _status_entry(
            number=11,
            title=PROBLEMS[10][1],
            status="PARTIALLY_FIXED",
            evidence=["backtest engine now applies funding_cost_for_trade with funding_pct_per_day"],
            remaining="Funding model is simplified and not venue-specific signed funding.",
        ),
        12: _status_entry(
            number=12,
            title=PROBLEMS[11][1],
            status="PARTIALLY_FIXED",
            evidence=["backtest engine now supports full_equity/fixed_fraction/risk_budget sizing modes"],
            remaining="Sizing still single-position and lacks richer portfolio/risk interactions.",
        ),
        13: _status_entry(
            number=13,
            title=PROBLEMS[12][1],
            status="PARTIALLY_FIXED",
            evidence=[f"reproducibility.frozen_research_mode={frozen_mode}", "stage71 stable hash-based seed mapping replaces Python hash() nondeterminism"],
            remaining="Full deterministic replay still depends on full pipeline input immutability.",
        ),
        14: _status_entry(
            number=14,
            title=PROBLEMS[13][1],
            status="PARTIALLY_FIXED",
            evidence=["stage69 cold_start_each_run + deduplicated sorted memory rows + frozen mode override"],
            remaining="Historical memory remains mutable when cold_start_each_run is disabled.",
        ),
        15: _status_entry(
            number=15,
            title=PROBLEMS[14][1],
            status="PARTIALLY_FIXED",
            evidence=[f"stage65 continuity validation_state={continuity_state}", "data.continuity report artifact added"],
            remaining="Strict continuity mode is configurable and can still be disabled.",
        ),
        16: _status_entry(
            number=16,
            title=PROBLEMS[15][1],
            status="PARTIALLY_FIXED",
            evidence=["stage reports now include execution_status/validation_state/stage_role fields"],
            remaining="Older docs outside stage chain may still contain legacy semantics.",
        ),
        17: _status_entry(
            number=17,
            title=PROBLEMS[16][1],
            status="PARTIALLY_FIXED" if runtime_measured else "NOT_FIXED",
            evidence=[f"stage55 projection_only={projection_only}", f"stage71 projected_only={stage71.get('projected_only', True)}"],
            remaining="Stage55 still depends on external runtime probe files for FULL measured status.",
        ),
        18: _status_entry(
            number=18,
            title=PROBLEMS[17][1],
            status="PARTIALLY_FIXED",
            evidence=["UI pages now surface decision_evidence_allowed/missing_real_sources/final verdict evidence flags"],
            remaining="UI still allows reading legacy summaries that may predate semantic fixes.",
        ),
        19: _status_entry(
            number=19,
            title=PROBLEMS[18][1],
            status="PARTIALLY_FIXED",
            evidence=["New tests added: validation evidence, continuity, backtest realism, stage57 evidence semantics, stage67 real artifacts, stage53 replay artifact"],
            remaining="Not all scientific assumptions are covered by deterministic end-to-end tests.",
        ),
        20: _status_entry(
            number=20,
            title=PROBLEMS[19][1],
            status="PARTIALLY_FIXED",
            evidence=["stage70 expanded families + structured hypothesis components + diversity checks"],
            remaining="Search space remains constrained and handcrafted.",
        ),
        21: _status_entry(
            number=21,
            title=PROBLEMS[20][1],
            status="PARTIALLY_FIXED",
            evidence=["stage60_72 orchestration now centralizes final decision ordering with real evidence prerequisites"],
            remaining="Parallel legacy stage ecosystems still exist in repository.",
        ),
        22: _status_entry(
            number=22,
            title=PROBLEMS[21][1],
            status="PARTIALLY_FIXED",
            evidence=["stage summaries now include execution_status/validation_state and downgrade status on insufficient decision evidence"],
            remaining="Legacy SUCCESS/PARTIAL contract remains for backward compatibility.",
        ),
        23: _status_entry(
            number=23,
            title=PROBLEMS[22][1],
            status="PARTIALLY_FIXED" if ml_reporting_only else "NOT_FIXED",
            evidence=[f"stage66 decision_use_allowed={stage66.get('decision_use_allowed')} stage_role={stage66.get('stage_role')}"],
            remaining="ML stack still present; misuse outside declared reporting-only role remains possible.",
        ),
        24: _status_entry(
            number=24,
            title=PROBLEMS[23][1],
            status="FULLY_FIXED",
            evidence=["backtest metrics sanitize inf/nan and cap pathological profit_factor", "metrics_sanitized flag added"],
            remaining="Metric clipping threshold choices remain heuristic.",
        ),
        25: _status_entry(
            number=25,
            title=PROBLEMS[24][1],
            status="PARTIALLY_FIXED",
            evidence=["full_trace report now includes evidence_quality/source_types and stage sequence hashes"],
            remaining="Clean runtime truth still depends on successful offline chain execution in current environment.",
        ),
    }
    return [statuses[number] for number, _ in PROBLEMS]


def _final_verdict(*, compile_status: str, pytest_status: str, integration_status: str, full_count: int, blocked_count: int) -> str:
    if blocked_count > 0:
        return "REPAIR_BLOCKED"
    all_ok = str(compile_status).upper() == "PASS" and str(pytest_status).upper() == "PASS" and str(integration_status).upper() == "PASS"
    if all_ok and full_count >= 6:
        return "MAJOR_REPAIR_COMPLETE"
    if all_ok:
        return "PARTIAL_REPAIR_MEANINGFUL"
    return "LIMITED_REPAIR_ONLY"


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = Path(args.runs_dir)

    problems = _build_problem_statuses(cfg=cfg, docs_dir=docs_dir, runs_dir=runs_dir)
    counts = {
        "FULLY_FIXED": int(sum(1 for row in problems if row["status"] == "FULLY_FIXED")),
        "PARTIALLY_FIXED": int(sum(1 for row in problems if row["status"] == "PARTIALLY_FIXED")),
        "NOT_FIXED": int(sum(1 for row in problems if row["status"] == "NOT_FIXED")),
        "NOT_APPLICABLE": int(sum(1 for row in problems if row["status"] == "NOT_APPLICABLE")),
        "BLOCKED": int(sum(1 for row in problems if row["status"] == "BLOCKED")),
    }
    verdict = _final_verdict(
        compile_status=str(args.compileall_status),
        pytest_status=str(args.pytest_status),
        integration_status=str(args.integration_status),
        full_count=counts["FULLY_FIXED"],
        blocked_count=counts["BLOCKED"],
    )

    summary = {
        "stage": "73",
        "status": "SUCCESS",
        "compileall_status": str(args.compileall_status),
        "pytest_status": str(args.pytest_status),
        "integration_status": str(args.integration_status),
        "problem_counts": counts,
        "problems": problems,
        "final_verdict": verdict,
    }
    summary["summary_hash"] = stable_hash(summary, length=16)
    (docs_dir / "stage73_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    lines: list[str] = [
        "# Stage-73 Repair Report",
        "",
        "## Verification Inputs",
        f"- compileall_status: `{summary['compileall_status']}`",
        f"- pytest_status: `{summary['pytest_status']}`",
        f"- integration_status: `{summary['integration_status']}`",
        "",
        "## Problem Status Counts",
        f"- FULLY_FIXED: `{counts['FULLY_FIXED']}`",
        f"- PARTIALLY_FIXED: `{counts['PARTIALLY_FIXED']}`",
        f"- NOT_FIXED: `{counts['NOT_FIXED']}`",
        f"- NOT_APPLICABLE: `{counts['NOT_APPLICABLE']}`",
        f"- BLOCKED: `{counts['BLOCKED']}`",
        "",
        "## Per-Problem Resolution",
    ]
    for row in problems:
        lines.extend(
            [
                f"### {row['problem_number']}. {row['title']}",
                f"- status: `{row['status']}`",
                f"- evidence: `{row['evidence']}`",
                f"- remaining_limitations: `{row['remaining_limitations']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Final Verdict",
            f"- final_verdict: `{verdict}`",
            f"- summary_hash: `{summary['summary_hash']}`",
            "",
        ]
    )
    (docs_dir / "stage73_report.md").write_text("\n".join(lines), encoding="utf-8")

    stage60 = _load_json(docs_dir / "stage60_summary.json")
    run_id = str(stage60.get("stage28_run_id", "")).strip()
    if run_id:
        out_dir = runs_dir / run_id / "stage73"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
        (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"status: {summary['status']}")
    print(f"final_verdict: {verdict}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
