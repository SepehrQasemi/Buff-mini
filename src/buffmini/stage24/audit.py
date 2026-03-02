"""Stage-24 unified audit helpers."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.forensics.signal_flow import run_signal_flow_trace
from buffmini.stage24.capital_sim import run_stage24_capital_sim
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def run_stage24_audit(
    *,
    config: dict[str, Any],
    seed: int,
    dry_run: bool,
    symbols: list[str],
    base_timeframe: str,
    operational_timeframe: str,
    initial_equities: list[float],
    runs_root: Path,
    data_dir: Path,
    derived_dir: Path,
    docs_dir: Path = Path("docs"),
) -> dict[str, Any]:
    """Run baseline vs Stage-24 sizing modes and write unified Stage-24 docs."""

    started = time.perf_counter()
    baseline_trace = _run_mode_trace(
        config=config,
        seed=seed,
        dry_run=dry_run,
        symbols=symbols,
        timeframe=operational_timeframe,
        mode_name="baseline",
        stage24_enabled=False,
        stage24_mode="risk_pct",
        runs_root=runs_root,
        data_dir=data_dir,
        derived_dir=derived_dir,
    )
    risk_trace = _run_mode_trace(
        config=config,
        seed=seed,
        dry_run=dry_run,
        symbols=symbols,
        timeframe=operational_timeframe,
        mode_name="risk_pct",
        stage24_enabled=True,
        stage24_mode="risk_pct",
        runs_root=runs_root,
        data_dir=data_dir,
        derived_dir=derived_dir,
    )
    alloc_trace = _run_mode_trace(
        config=config,
        seed=seed,
        dry_run=dry_run,
        symbols=symbols,
        timeframe=operational_timeframe,
        mode_name="alloc_pct",
        stage24_enabled=True,
        stage24_mode="alloc_pct",
        runs_root=runs_root,
        data_dir=data_dir,
        derived_dir=derived_dir,
    )

    capital = run_stage24_capital_sim(
        config=deepcopy(config),
        seed=int(seed),
        dry_run=bool(dry_run),
        symbols=list(symbols),
        base_timeframe=str(base_timeframe),
        operational_timeframe=str(operational_timeframe),
        mode="risk_pct",
        initial_equities=list(initial_equities),
        runs_root=runs_root,
        data_dir=data_dir,
        derived_dir=derived_dir,
        docs_dir=docs_dir,
    )

    baseline_metrics = _mode_metrics(baseline_trace)
    risk_metrics = _mode_metrics(risk_trace)
    alloc_metrics = _mode_metrics(alloc_trace)
    capital_summary = dict(capital["summary"])
    top_reasons = {
        "baseline": baseline_metrics.get("top_reject_reasons", []),
        "risk_pct": risk_metrics.get("top_reject_reasons", []),
        "alloc_pct": alloc_metrics.get("top_reject_reasons", []),
    }

    next_bottleneck = _next_bottleneck(risk_trace)
    verdict = _stage24_verdict(
        baseline=baseline_metrics,
        risk=risk_metrics,
        alloc=alloc_metrics,
    )

    payload = {
        "stage": "24",
        "generated_at": utc_now_compact(),
        "seed": int(seed),
        "base_timeframe": str(base_timeframe),
        "operational_timeframe": str(operational_timeframe),
        "symbols": list(symbols),
        "runtime_seconds": float(time.perf_counter() - started),
        "baseline": baseline_metrics,
        "risk_pct_mode": risk_metrics,
        "alloc_pct_mode": alloc_metrics,
        "capital_sim": {
            "run_id": str(capital_summary.get("run_id", "")),
            "results_hash": str(capital_summary.get("results_hash", "")),
            "scale_invariance_check": dict(capital_summary.get("scale_invariance_check", {})),
            "rows": list(capital_summary.get("rows", [])),
        },
        "top_reject_reasons": top_reasons,
        "same_seed_all_modes": bool(
            baseline_metrics.get("seed", 0) == risk_metrics.get("seed", -1) == alloc_metrics.get("seed", -2) == int(seed)
        ),
        "same_data_hash_all_modes": bool(
            str(baseline_metrics.get("data_hash", ""))
            == str(risk_metrics.get("data_hash", ""))
            == str(alloc_metrics.get("data_hash", ""))
        ),
        "verdict": str(verdict),
        "next_bottleneck": next_bottleneck,
        "summary_hash": stable_hash(
            {
                "seed": int(seed),
                "baseline": baseline_metrics,
                "risk": risk_metrics,
                "alloc": alloc_metrics,
                "capital_hash": capital_summary.get("results_hash", ""),
                "next_bottleneck": next_bottleneck,
            },
            length=16,
        ),
    }

    docs_dir.mkdir(parents=True, exist_ok=True)
    report_md = docs_dir / "stage24_report.md"
    report_json = docs_dir / "stage24_report_summary.json"
    report_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    report_md.write_text(render_stage24_report_md(payload), encoding="utf-8")

    return {
        "payload": payload,
        "report_md": report_md,
        "report_json": report_json,
        "baseline_run_id": baseline_trace["run_id"],
        "risk_run_id": risk_trace["run_id"],
        "alloc_run_id": alloc_trace["run_id"],
        "capital_run_id": capital_summary.get("run_id", ""),
    }


def render_stage24_report_md(payload: dict[str, Any]) -> str:
    baseline = dict(payload.get("baseline", {}))
    risk = dict(payload.get("risk_pct_mode", {}))
    alloc = dict(payload.get("alloc_pct_mode", {}))
    rows = list((payload.get("capital_sim", {}) or {}).get("rows", []))
    lines = [
        "# Stage-24 Report",
        "",
        "## What Stage-24 Adds",
        "- Dual sizing modes: `risk_pct` (primary) and `alloc_pct` (secondary).",
        "- Cost-aware notional sizing: `notional = equity * risk_pct / (stop_distance_pct + cost_rt_pct)`.",
        "- Capital-level diagnostics for min-notional/cap bottlenecks.",
        "",
        "## Reproducibility",
        f"- seed: `{payload.get('seed', 0)}`",
        f"- same_seed_all_modes: `{bool(payload.get('same_seed_all_modes', False))}`",
        f"- same_data_hash_all_modes: `{bool(payload.get('same_data_hash_all_modes', False))}`",
        "",
        "## Baseline vs Stage-24 Modes",
        "| metric | baseline | risk_pct | alloc_pct |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in ("trade_count", "zero_trade_pct", "invalid_order_pct", "walkforward_executed_true_pct", "mc_trigger_rate"):
        lines.append(
            f"| {key} | {float(baseline.get(key, 0.0)):.6f} | {float(risk.get(key, 0.0)):.6f} | {float(alloc.get(key, 0.0)):.6f} |"
        )

    lines.extend(
        [
            "",
            "## Capital-Level Simulation (risk_pct mode)",
            "| initial_equity | final_equity | return_pct | max_drawdown | trade_count | avg_notional | avg_risk_pct_used | invalid_pct | top_reason |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in rows:
        lines.append(
            "| {initial_equity:.2f} | {final_equity:.6f} | {return_pct:.6f} | {max_drawdown:.6f} | {trade_count:.2f} | {avg_notional:.6f} | {avg_risk_pct_used:.6f} | {invalid_order_pct:.6f} | {top_invalid_reason} |".format(
                **{
                    "initial_equity": float(row.get("initial_equity", 0.0)),
                    "final_equity": float(row.get("final_equity", 0.0)),
                    "return_pct": float(row.get("return_pct", 0.0)),
                    "max_drawdown": float(row.get("max_drawdown", 0.0)),
                    "trade_count": float(row.get("trade_count", 0.0)),
                    "avg_notional": float(row.get("avg_notional", 0.0)),
                    "avg_risk_pct_used": float(row.get("avg_risk_pct_used", 0.0)),
                    "invalid_order_pct": float(row.get("invalid_order_pct", 0.0)),
                    "top_invalid_reason": str(row.get("top_invalid_reason", "VALID")),
                }
            )
        )

    bottleneck = dict(payload.get("next_bottleneck", {}))
    lines.extend(
        [
            "",
            "## Why It May Still Choke",
            f"- next_bottleneck_gate: `{bottleneck.get('gate', '')}`",
            f"- next_bottleneck_death_rate: `{float(bottleneck.get('death_rate', 0.0)):.6f}`",
            "",
            "## Top Reject Reasons (risk_pct)",
        ]
    )
    for item in (payload.get("top_reject_reasons", {}) or {}).get("risk_pct", [])[:10]:
        lines.append(f"- {item.get('reason', '')}: {int(item.get('count', 0))}")

    lines.extend(
        [
            "",
            "## Next Actions",
            "- If `SIZE_TOO_SMALL` dominates small equity tiers: verify minimum tradable size feasibility before alpha changes.",
            "- If cap/margin rejects dominate: recalibrate risk caps rather than weakening validation gates.",
            "- If trade_count remains low despite valid sizing: prioritize signal-density/discovery stages, not sizing.",
            "",
            f"## Verdict\n- `{payload.get('verdict', '')}`",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _run_mode_trace(
    *,
    config: dict[str, Any],
    seed: int,
    dry_run: bool,
    symbols: list[str],
    timeframe: str,
    mode_name: str,
    stage24_enabled: bool,
    stage24_mode: str,
    runs_root: Path,
    data_dir: Path,
    derived_dir: Path,
) -> dict[str, Any]:
    cfg = deepcopy(config)
    cfg.setdefault("universe", {})["base_timeframe"] = "1m"
    cfg.setdefault("data", {})["resample_source"] = "base"
    stage24 = cfg.setdefault("evaluation", {}).setdefault("stage24", {})
    stage24["enabled"] = bool(stage24_enabled)
    stage24.setdefault("sizing", {})
    stage24["sizing"]["mode"] = str(stage24_mode)
    stage24.setdefault("simulation", {})
    stage24["simulation"]["seed"] = int(seed)
    return run_signal_flow_trace(
        config=cfg,
        seed=int(seed),
        symbols=list(symbols),
        timeframes=[str(timeframe)],
        mode="classic",
        stages=["classic"],
        families=["price"],
        composers=["none"],
        max_combos=0,
        dry_run=bool(dry_run),
        runs_root=runs_root,
        data_dir=data_dir,
        derived_dir=derived_dir,
    )


def _mode_metrics(trace: dict[str, Any]) -> dict[str, Any]:
    rows = pd.DataFrame(trace.get("rows", pd.DataFrame()))
    summary = dict(trace.get("summary", {}))
    trace_dir = Path(trace.get("trace_dir", ""))
    breakdown = _read_json(trace_dir / "execution_reject_breakdown.json")
    stage24_summary = _read_json(trace_dir / "stage24_sizing_summary.json")
    shadow_live = _read_json(trace_dir / "shadow_live_summary.json")
    attempted = float(breakdown.get("total_orders_attempted", 0.0))
    rejected = float(breakdown.get("total_orders_rejected", 0.0))
    trade_count = float(pd.to_numeric(rows.get("trades_executed_count", 0.0), errors="coerce").fillna(0.0).sum()) if not rows.empty else 0.0
    top_reasons = _top_reasons(dict(breakdown.get("reject_reason_counts", {})), limit=10)
    return {
        "run_id": str(trace.get("run_id", "")),
        "trace_dir": str(trace_dir),
        "seed": int(summary.get("seed", 0)),
        "config_hash": str(summary.get("config_hash", "")),
        "data_hash": str(summary.get("data_hash", "")),
        "resolved_end_ts": str(summary.get("resolved_end_ts", "")),
        "trade_count": trade_count,
        "zero_trade_pct": float(summary.get("zero_trade_pct", 0.0)),
        "invalid_pct": float(summary.get("invalid_pct", 0.0)),
        "walkforward_executed_true_pct": float(summary.get("walkforward_executed_true_pct", 0.0)),
        "mc_trigger_rate": float(summary.get("mc_trigger_rate", 0.0)),
        "invalid_order_pct": float((rejected / attempted) * 100.0) if attempted > 0 else 0.0,
        "top_reject_reason": str(top_reasons[0]["reason"]) if top_reasons else "VALID",
        "top_reject_reasons": top_reasons,
        "stage24_valid_count": int(stage24_summary.get("valid_count", 0)),
        "stage24_invalid_count": int(stage24_summary.get("invalid_count", 0)),
        "shadow_live_summary": shadow_live,
    }


def _top_reasons(reason_counts: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    rows = [{"reason": str(reason), "count": int(count)} for reason, count in reason_counts.items() if int(count) > 0]
    rows.sort(key=lambda item: (-item["count"], item["reason"]))
    return rows[: int(limit)]


def _next_bottleneck(trace: dict[str, Any]) -> dict[str, Any]:
    summary = dict(trace.get("summary", {}))
    top = list(summary.get("top_bottlenecks", []))
    if top:
        first = dict(top[0])
        return {"gate": str(first.get("gate", "")), "death_rate": float(first.get("death_rate", 0.0))}
    return {"gate": "", "death_rate": 0.0}


def _stage24_verdict(*, baseline: dict[str, Any], risk: dict[str, Any], alloc: dict[str, Any]) -> str:
    risk_active = int(risk.get("stage24_valid_count", 0)) > 0
    alloc_active = int(alloc.get("stage24_valid_count", 0)) > 0
    if not (risk_active or alloc_active):
        return "NO_OP_BUG"
    risk_improved = (
        float(risk.get("invalid_order_pct", 0.0)) < float(baseline.get("invalid_order_pct", 0.0))
        or float(risk.get("zero_trade_pct", 0.0)) < float(baseline.get("zero_trade_pct", 0.0))
    )
    alloc_improved = (
        float(alloc.get("invalid_order_pct", 0.0)) < float(baseline.get("invalid_order_pct", 0.0))
        or float(alloc.get("zero_trade_pct", 0.0)) < float(baseline.get("zero_trade_pct", 0.0))
    )
    if risk_improved or alloc_improved:
        return "IMPROVED"
    return "SIZING_ACTIVE_NO_EDGE_CHANGE"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")))
