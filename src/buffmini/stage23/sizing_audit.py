"""Stage-23.6 sizing integrity audit helpers."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.forensics.signal_flow import run_signal_flow_trace
from buffmini.utils.time import utc_now_compact


def run_stage23_6_audit(
    *,
    config: dict[str, Any],
    seed: int,
    dry_run: bool,
    symbols: list[str],
    timeframes: list[str],
    mode: str,
    stages: list[str],
    families: list[str],
    composers: list[str],
    max_combos: int,
    runs_root: Path,
    data_dir: Path,
    derived_dir: Path,
    docs_dir: Path = Path("docs"),
) -> dict[str, Any]:
    cfg_baseline = deepcopy(config)
    stage23_base = cfg_baseline.setdefault("evaluation", {}).setdefault("stage23", {})
    stage23_base["enabled"] = True
    stage23_base["sizing_fix_enabled"] = False
    baseline = run_signal_flow_trace(
        config=cfg_baseline,
        seed=int(seed),
        symbols=list(symbols),
        timeframes=list(timeframes),
        mode=str(mode),
        stages=list(stages),
        families=list(families),
        composers=list(composers),
        max_combos=int(max_combos),
        dry_run=bool(dry_run),
        runs_root=runs_root,
        data_dir=data_dir,
        derived_dir=derived_dir,
    )

    cfg_after = deepcopy(config)
    stage23_after = cfg_after.setdefault("evaluation", {}).setdefault("stage23", {})
    stage23_after["enabled"] = True
    stage23_after["sizing_fix_enabled"] = True
    after = run_signal_flow_trace(
        config=cfg_after,
        seed=int(seed),
        symbols=list(symbols),
        timeframes=list(timeframes),
        mode=str(mode),
        stages=list(stages),
        families=list(families),
        composers=list(composers),
        max_combos=int(max_combos),
        dry_run=bool(dry_run),
        runs_root=runs_root,
        data_dir=data_dir,
        derived_dir=derived_dir,
    )

    baseline_summary = dict(baseline["summary"])
    after_summary = dict(after["summary"])
    baseline_breakdown = _read_json(Path(baseline["trace_dir"]) / "execution_reject_breakdown.json")
    after_breakdown = _read_json(Path(after["trace_dir"]) / "execution_reject_breakdown.json")
    baseline_sizing = _read_json(Path(baseline["trace_dir"]) / "sizing_trace_summary.json")
    after_sizing = _read_json(Path(after["trace_dir"]) / "sizing_trace_summary.json")

    metrics_baseline = _metrics_from_summary(baseline_summary)
    metrics_after = _metrics_from_summary(after_summary)
    death_baseline = _death_from_summary(baseline_summary)
    death_after = _death_from_summary(after_summary)
    top_reasons_baseline = _top_reject_reasons(baseline_breakdown, limit=10)
    top_reasons_after = _top_reject_reasons(after_breakdown, limit=10)

    size_zero_share_baseline = _size_zero_share(baseline_breakdown)
    size_zero_share_after = _size_zero_share(after_breakdown)
    criterion_size_zero = bool(
        size_zero_share_after <= 0.01
        or (
            size_zero_share_baseline > 0
            and (size_zero_share_baseline - size_zero_share_after) / max(size_zero_share_baseline, 1e-12) >= 0.90
        )
    )
    criterion_choke = bool(
        (float(metrics_after["zero_trade_pct"]) - float(metrics_baseline["zero_trade_pct"])) <= -10.0
        or (float(death_after["execution"]) - float(death_baseline["execution"])) <= -0.15
    )
    improvement_sufficient = bool(criterion_size_zero and criterion_choke)

    deltas = {key: float(metrics_after[key] - metrics_baseline[key]) for key in metrics_baseline}
    deltas.update({f"death_{key}": float(death_after[key] - death_baseline[key]) for key in death_baseline})
    sizing_delta = _delta_map(baseline_sizing, after_sizing, keys=("zero_size_count", "rescued_by_ceil_count", "bumped_to_min_notional_count", "cap_binding_reject_count"))
    biggest_bottleneck = (after_summary.get("top_bottlenecks") or [{}])[0]

    deep_why: dict[str, Any] | None = None
    if not improvement_sufficient:
        deep_why = _run_deep_why(after_trace_dir=Path(after["trace_dir"]), summary=after_summary)

    payload = {
        "stage": "23.6",
        "generated_at": utc_now_compact(),
        "seed": int(seed),
        "criteria": {
            "size_zero_share_baseline": float(size_zero_share_baseline),
            "size_zero_share_after": float(size_zero_share_after),
            "criterion_size_zero_pass": criterion_size_zero,
            "criterion_choke_pass": criterion_choke,
            "improvement_sufficient": improvement_sufficient,
        },
        "baseline": {
            "run_id": str(baseline["run_id"]),
            "trace_dir": str(baseline["trace_dir"]),
            "summary": baseline_summary,
            "execution_reject_breakdown": baseline_breakdown,
            "sizing_trace_summary": baseline_sizing,
            "metrics": metrics_baseline,
            "death_rates": death_baseline,
            "top_reject_reasons": top_reasons_baseline,
        },
        "after": {
            "run_id": str(after["run_id"]),
            "trace_dir": str(after["trace_dir"]),
            "summary": after_summary,
            "execution_reject_breakdown": after_breakdown,
            "sizing_trace_summary": after_sizing,
            "metrics": metrics_after,
            "death_rates": death_after,
            "top_reject_reasons": top_reasons_after,
        },
        "deltas": deltas,
        "sizing_delta": sizing_delta,
        "biggest_remaining_bottleneck": biggest_bottleneck,
        "same_seed": bool(baseline_summary.get("seed") == after_summary.get("seed")),
        "same_data_hash": bool(str(baseline_summary.get("data_hash", "")) == str(after_summary.get("data_hash", ""))),
        "deep_why": deep_why,
    }

    docs_dir.mkdir(parents=True, exist_ok=True)
    report_md = docs_dir / "stage23_6_report.md"
    report_json = docs_dir / "stage23_6_report_summary.json"
    diff_json = docs_dir / "stage23_6_report_diff.json"
    report_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    diff_json.write_text(json.dumps({"deltas": payload["deltas"], "sizing_delta": payload["sizing_delta"]}, indent=2, allow_nan=False), encoding="utf-8")
    report_md.write_text(render_stage23_6_md(payload), encoding="utf-8")

    return {
        "baseline_run_id": baseline["run_id"],
        "after_run_id": after["run_id"],
        "baseline_trace_dir": baseline["trace_dir"],
        "after_trace_dir": after["trace_dir"],
        "report_md": report_md,
        "report_json": report_json,
        "diff_json": diff_json,
        "payload": payload,
    }


def render_stage23_6_md(payload: dict[str, Any]) -> str:
    b = payload["baseline"]
    a = payload["after"]
    d = payload["deltas"]
    sd = payload.get("sizing_delta", {})
    criteria = payload.get("criteria", {})
    bottleneck = payload.get("biggest_remaining_bottleneck", {})
    lines = [
        "# Stage-23.6 Sizing Integrity Repair Report",
        "",
        f"- seed: `{payload['seed']}`",
        f"- baseline_run_id: `{b['run_id']}`",
        f"- after_run_id: `{a['run_id']}`",
        f"- same_seed: `{payload['same_seed']}`",
        f"- same_data_hash: `{payload['same_data_hash']}`",
        "",
        "## Baseline vs After",
        "| metric | baseline | after | delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in ("zero_trade_pct", "invalid_pct", "walkforward_executed_true_pct", "mc_trigger_rate"):
        lines.append(
            f"| {key} | {float(b['metrics'][key]):.6f} | {float(a['metrics'][key]):.6f} | {float(d[key]):.6f} |"
        )
    for key in ("context", "orders", "execution"):
        lines.append(
            f"| death_{key} | {float(b['death_rates'][key]):.6f} | {float(a['death_rates'][key]):.6f} | {float(d[f'death_{key}']):.6f} |"
        )

    lines.extend(
        [
            "",
            "## Sizing Trace Summary (Baseline vs After)",
            "| counter | baseline | after | delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for key in ("zero_size_count", "rescued_by_ceil_count", "bumped_to_min_notional_count", "cap_binding_reject_count"):
        base_val = float((b.get("sizing_trace_summary", {}) or {}).get(key, 0.0))
        after_val = float((a.get("sizing_trace_summary", {}) or {}).get(key, 0.0))
        lines.append(f"| {key} | {base_val:.6f} | {after_val:.6f} | {float(sd.get(key, after_val - base_val)):.6f} |")

    lines.extend(
        [
            "",
            "## Top 10 Reject Reasons (Baseline vs After)",
            "| reason | baseline_count | after_count |",
            "| --- | ---: | ---: |",
        ]
    )
    baseline_map = {str(item["reason"]): int(item["count"]) for item in b.get("top_reject_reasons", [])}
    for item in a.get("top_reject_reasons", []):
        reason = str(item["reason"])
        lines.append(f"| {reason} | {int(baseline_map.get(reason, 0))} | {int(item['count'])} |")

    lines.extend(
        [
            "",
            "## Improvement Criteria",
            f"- size_zero_share_baseline: `{float(criteria.get('size_zero_share_baseline', 0.0)):.6f}`",
            f"- size_zero_share_after: `{float(criteria.get('size_zero_share_after', 0.0)):.6f}`",
            f"- criterion_size_zero_pass: `{bool(criteria.get('criterion_size_zero_pass', False))}`",
            f"- criterion_choke_pass: `{bool(criteria.get('criterion_choke_pass', False))}`",
            f"- improvement_sufficient: `{bool(criteria.get('improvement_sufficient', False))}`",
            "",
            "## Biggest Remaining Bottleneck",
            f"- gate: `{bottleneck.get('gate', '')}`",
            f"- death_rate: `{float(bottleneck.get('death_rate', 0.0)):.6f}`",
        ]
    )

    deep_why = payload.get("deep_why")
    if deep_why:
        lines.extend(
            [
                "",
                "## Deep Why (Triggered)",
                f"- likely_root_cause: `{deep_why.get('likely_root_cause', '')}`",
                f"- reason_rank: `{deep_why.get('reason_rank', [])}`",
                f"- raw_size_quantiles: `{deep_why.get('raw_size_quantiles', {})}`",
                f"- step_stats: `{deep_why.get('step_stats', {})}`",
                f"- by_symbol_side_top3: `{deep_why.get('by_symbol_side_top3', [])}`",
                "",
                "### Next Actions",
                "- If SIZE_TOO_SMALL dominates: revisit capital/min-notional feasibility before strategy changes.",
                "- If POLICY_CAP_HIT dominates: inspect cap calibration rather than relaxing validation gates.",
                "- If upstream scarcity dominates (RAW_SIGNAL_ZERO/CONTEXT_REJECT): tune signal density before execution changes.",
            ]
        )

    lines.extend(
        [
            "",
            "## Evidence Paths",
            f"- baseline trace: `{b['trace_dir']}`",
            f"- after trace: `{a['trace_dir']}`",
            f"- baseline sizing trace: `{Path(b['trace_dir']) / 'sizing_trace.csv'}`",
            f"- after sizing trace: `{Path(a['trace_dir']) / 'sizing_trace.csv'}`",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _run_deep_why(*, after_trace_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    sizing_path = after_trace_dir / "sizing_trace.csv"
    if not sizing_path.exists():
        return {"likely_root_cause": "NO_SIZING_TRACE", "reason_rank": []}
    df = pd.read_csv(sizing_path)
    if df.empty:
        return {"likely_root_cause": "NO_SIZING_TRACE_ROWS", "reason_rank": []}
    reject_df = df.loc[df.get("decision", "").astype(str) == "REJECTED"].copy()
    reason_rank = (
        reject_df.get("reject_reason", pd.Series(dtype=str))
        .astype(str)
        .replace("", "UNKNOWN")
        .value_counts()
        .head(10)
        .reset_index()
    )
    reason_rank.columns = ["reason", "count"]
    top3 = reason_rank["reason"].head(3).tolist()
    by_symbol_side: list[dict[str, Any]] = []
    for reason in top3:
        g = reject_df.loc[reject_df["reject_reason"].astype(str) == str(reason)]
        grp = (
            g.groupby(["symbol", "side"], dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .head(5)
        )
        by_symbol_side.append({"reason": str(reason), "breakdown": grp.to_dict(orient="records")})

    raw_size = pd.to_numeric(df.get("raw_size", 0.0), errors="coerce").fillna(0.0)
    step = pd.to_numeric(df.get("qty_step", 0.0), errors="coerce").fillna(0.0)
    min_notional = pd.to_numeric(df.get("min_notional", 0.0), errors="coerce").fillna(0.0)
    raw_q = {
        "p01": float(raw_size.quantile(0.01)),
        "p05": float(raw_size.quantile(0.05)),
        "p10": float(raw_size.quantile(0.10)),
    }
    step_stats = {
        "step_median": float(step.median()),
        "step_p95": float(step.quantile(0.95)),
        "min_notional_median": float(min_notional.median()),
    }

    dominant = str(reason_rank.iloc[0]["reason"]) if not reason_rank.empty else ""
    if dominant in {"SIZE_TOO_SMALL", "SIZE_ZERO"}:
        likely = "capital_too_small_or_min_notional_constraints"
    elif dominant == "POLICY_CAP_HIT":
        likely = "policy_caps_too_tight"
    elif dominant == "MARGIN_FAIL":
        likely = "margin_model_too_strict"
    elif dominant in {"SLIPPAGE_TOO_HIGH", "SPREAD_TOO_HIGH", "NO_FILL"}:
        likely = "execution_thresholds_or_liquidity_constraints"
    else:
        top_gate = (summary.get("top_bottlenecks") or [{}])[0]
        if str(top_gate.get("gate", "")) in {"death_context", "death_confirm", "death_riskgate"}:
            likely = "signal_scarcity_upstream_context_gate"
        else:
            likely = "mixed_or_unresolved"

    return {
        "likely_root_cause": likely,
        "reason_rank": reason_rank.to_dict(orient="records"),
        "by_symbol_side_top3": by_symbol_side,
        "raw_size_quantiles": raw_q,
        "step_stats": step_stats,
    }


def _metrics_from_summary(summary: dict[str, Any]) -> dict[str, float]:
    return {
        "zero_trade_pct": float(summary.get("zero_trade_pct", 0.0)),
        "invalid_pct": float(summary.get("invalid_pct", 0.0)),
        "walkforward_executed_true_pct": float(summary.get("walkforward_executed_true_pct", 0.0)),
        "mc_trigger_rate": float(summary.get("mc_trigger_rate", 0.0)),
    }


def _death_from_summary(summary: dict[str, Any]) -> dict[str, float]:
    means = dict(summary.get("overall_gate_means", {}) or {})
    return {
        "context": float(means.get("death_context", 0.0)),
        "orders": float(means.get("death_orders", 0.0)),
        "execution": float(means.get("death_execution", 0.0)),
    }


def _top_reject_reasons(payload: dict[str, Any], limit: int = 10) -> list[dict[str, Any]]:
    counts = dict(payload.get("reject_reason_counts", {}) or {})
    rows = [{"reason": str(reason), "count": int(count)} for reason, count in counts.items() if int(count) > 0]
    rows = sorted(rows, key=lambda item: (-item["count"], item["reason"]))
    return rows[: int(limit)]


def _size_zero_share(payload: dict[str, Any]) -> float:
    attempted = float(payload.get("total_orders_attempted", 0.0))
    size_zero = float((payload.get("reject_reason_counts", {}) or {}).get("SIZE_ZERO", 0.0))
    return float(size_zero / attempted) if attempted > 0 else 0.0


def _delta_map(left: dict[str, Any], right: dict[str, Any], *, keys: tuple[str, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in keys:
        out[key] = float(float(right.get(key, 0.0)) - float(left.get(key, 0.0)))
    return out


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")))
