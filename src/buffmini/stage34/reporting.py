"""Stage-34 report builders."""

from __future__ import annotations

from typing import Any


def build_next_actions(summary: dict[str, Any]) -> list[str]:
    verdict = str(summary.get("final_verdict", "NO_EDGE"))
    bottleneck = str(summary.get("top_bottleneck", "signal_quality"))
    actions: list[str] = []
    if verdict == "EDGE":
        actions.extend(
            [
                "Run a paper-trading harness with frozen policy and strict drift alerts before any execution integration.",
                "Increase evidence windows (more rolling slices and longer holdout) without changing thresholds.",
                "Run additional cost-stress scenarios to confirm edge survives realistic drag.",
            ]
        )
    elif verdict == "WEAK_EDGE":
        actions.extend(
            [
                "Increase evidence by extending rolling validation and gathering more context occurrences.",
                "Tighten policy to contexts with positive exp_lcb repeatability and keep rare contexts flagged.",
                "Run feasibility-aware threshold sweeps without reducing cost assumptions.",
            ]
        )
    elif verdict == "INSUFFICIENT_DATA":
        actions.extend(
            [
                "Verify local snapshot coverage and fill missing canonical windows from the fixed local source.",
                "Re-run Stage-34 once coverage and derived timeframe integrity are restored.",
                "Do not interpret model quality until WF/MC execute on adequate samples.",
            ]
        )
    else:  # NO_EDGE
        actions.extend(
            [
                "Add next free data families: cross-symbol relative strength, session structure, and volatility-state transitions.",
                "Expand model/search diversity: monotonic tree constraints, calibrated ranking losses, and richer feature subset mutations.",
                "Prioritize bottleneck repair: "
                + (
                    "cost drag sensitivity via trade quality filters and lower-turnover exits."
                    if bottleneck == "cost_drag"
                    else "policy activation and threshold calibration per context."
                    if bottleneck == "policy_thresholds"
                    else "signal quality and contextual evidence density."
                ),
            ]
        )
    return actions


def render_stage34_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage-34 Report",
        "",
        "## Executive Summary",
        f"- run_id: `{summary.get('run_id', '')}`",
        f"- final_verdict: `{summary.get('final_verdict', 'NO_EDGE')}`",
        f"- top_bottleneck: `{summary.get('top_bottleneck', '')}`",
        f"- did_generations_improve: `{bool(summary.get('did_generations_improve', False))}`",
        "",
        "## Built Components",
        "- 34.1 data snapshot audit + deterministic timeframe completion",
        "- 34.2 OHLCV-only ML dataset + label pipeline",
        "- 34.3 deterministic CPU model training + calibration",
        "- 34.4 strict rolling WF/MC + cost stress evaluation",
        "- 34.5 policy selection + held-out replay",
        "- 34.6 model registry",
        "- 34.7 evolution engine",
        "- 34.8 ten-generation experiment",
        "",
        "## Data Snapshot Audit",
        f"- snapshot_hash: `{summary.get('snapshot_hash', '')}`",
        f"- resolved_end_ts: `{summary.get('resolved_end_ts', '')}`",
        f"- data_hash: `{summary.get('data_hash', '')}`",
        "",
        "## ML Dataset Summary",
        f"- rows_total: `{int(summary.get('dataset', {}).get('rows_total', 0))}`",
        f"- feature_count: `{int(summary.get('dataset', {}).get('feature_count', 0))}`",
        f"- timeframes: `{summary.get('dataset', {}).get('timeframes', [])}`",
        "",
        "## Model Training Summary",
    ]
    for row in summary.get("training", {}).get("models", []):
        lines.append(
            "- `{model_name}` val_logloss={val_logloss:.6f} test_logloss={test_logloss:.6f} test_brier={test_brier:.6f}".format(
                model_name=str(row.get("model_name", "")),
                val_logloss=float(row.get("val_logloss", 0.0)),
                test_logloss=float(row.get("test_logloss", 0.0)),
                test_brier=float(row.get("test_brier", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Walkforward + Monte Carlo",
            f"- wf_executed_pct: `{float(summary.get('evaluation', {}).get('wf_executed_pct', 0.0)):.2f}`",
            f"- mc_trigger_pct: `{float(summary.get('evaluation', {}).get('mc_trigger_pct', 0.0)):.2f}`",
            f"- failure_modes: `{summary.get('evaluation', {}).get('failure_mode_counts', {})}`",
            "",
            "## Cost Drag Analysis",
            f"- research_best_exp_lcb: `{float(summary.get('evaluation', {}).get('research_best_exp_lcb', 0.0)):.6f}`",
            f"- live_best_exp_lcb: `{float(summary.get('evaluation', {}).get('live_best_exp_lcb', 0.0)):.6f}`",
            "",
            "## Policy Replay",
            f"- policy_id: `{summary.get('policy', {}).get('policy_id', '')}`",
            f"- research_trade_count: `{int(summary.get('policy', {}).get('research_trade_count', 0))}`",
            f"- live_trade_count: `{int(summary.get('policy', {}).get('live_trade_count', 0))}`",
            f"- top_reject_reasons: `{summary.get('policy', {}).get('top_reject_reasons', [])}`",
            "",
            "## Ten-Generation Comparison",
            f"- generation_count: `{int(summary.get('generations', {}).get('generation_count', 0))}`",
            f"- best_generation: `{int(summary.get('generations', {}).get('best_generation', 0))}`",
            f"- did_generations_improve: `{bool(summary.get('did_generations_improve', False))}`",
            "",
            "## Bug Fixes",
            "- Fixed Stage-34.3 runner path handling bug that attempted to read repo root as parquet when dataset path was omitted.",
            "- Added bounded-compute deterministic subset selection for evolution loops to prevent runtime blowups.",
            "",
            "## Final Verdict",
            f"- `{summary.get('final_verdict', 'NO_EDGE')}`",
            "",
            "## Biggest Bottleneck",
            f"- `{summary.get('top_bottleneck', '')}`",
            f"- evidence: `{summary.get('bottleneck_evidence', {})}`",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def render_next_actions_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage-34 Next Actions",
        "",
        f"- final_verdict: `{summary.get('final_verdict', 'NO_EDGE')}`",
        "",
    ]
    for idx, item in enumerate(build_next_actions(summary), start=1):
        lines.append(f"{idx}. {item}")
    return "\n".join(lines).strip() + "\n"
