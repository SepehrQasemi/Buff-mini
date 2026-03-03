"""Stage-25B family edge program runner."""

from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.data.snapshot import snapshot_metadata_from_config
from buffmini.forensics.signal_flow import run_signal_flow_trace
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def run_stage25b_edge_program(
    *,
    config: dict[str, Any],
    seed: int,
    dry_run: bool,
    mode: str,
    symbols: list[str],
    timeframes: list[str],
    families: list[str],
    composers: list[str],
    cost_levels: list[str],
    runs_root: Path,
    data_dir: Path,
    derived_dir: Path,
    docs_dir: Path = Path("docs"),
    out_run_id: str | None = None,
) -> dict[str, Any]:
    """Run bounded Stage-25B sweeps and export run + docs artifacts."""

    clean_mode = str(mode).strip().lower()
    if clean_mode not in {"research", "live"}:
        raise ValueError("mode must be research|live")
    clean_families = [str(item).strip() for item in families if str(item).strip()]
    clean_composers = [str(item).strip() for item in composers if str(item).strip()]
    clean_cost_levels = [str(item).strip().lower() for item in cost_levels if str(item).strip()]
    if not clean_families:
        raise ValueError("families must be non-empty")
    if not clean_composers:
        clean_composers = ["weighted_sum"]
    if not clean_cost_levels:
        clean_cost_levels = ["realistic", "high"]

    started = time.perf_counter()
    run_id = out_run_id or (
        f"{utc_now_compact()}_"
        f"{stable_hash({'seed': int(seed), 'mode': clean_mode, 'symbols': symbols, 'timeframes': timeframes, 'families': clean_families, 'composers': clean_composers, 'cost_levels': clean_cost_levels}, length=12)}"
        "_stage25B"
    )
    run_dir = runs_root / run_id
    out_dir = run_dir / "stage25B"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    trace_refs: list[dict[str, Any]] = []
    for cost_level in clean_cost_levels:
        cfg = _prepare_cfg(
            config=config,
            mode=clean_mode,
            cost_level=cost_level,
        )
        trace = run_signal_flow_trace(
            config=cfg,
            seed=int(seed),
            symbols=list(symbols),
            timeframes=list(timeframes),
            mode="v2",
            stages=["15", "17"],
            families=list(clean_families),
            composers=list(clean_composers),
            max_combos=0,
            dry_run=bool(dry_run),
            runs_root=runs_root,
            data_dir=data_dir,
            derived_dir=derived_dir,
        )
        trace_refs.append(
            {
                "cost_level": str(cost_level),
                "trace_run_id": str(trace["run_id"]),
                "trace_dir": str(trace["trace_dir"]),
                "config_hash": str(trace["summary"].get("config_hash", "")),
                "data_hash": str(trace["summary"].get("data_hash", "")),
                "resolved_end_ts": str(trace["summary"].get("resolved_end_ts", "")),
            }
        )
        trace_rows = pd.DataFrame(trace.get("rows", pd.DataFrame()))
        if trace_rows.empty:
            continue
        for _, row in trace_rows.iterrows():
            exit_variant = "atr_trailing" if str(row.get("stage", "")) == "17" else "fixed_atr"
            trade_count = float(_to_num(row.get("trades_executed_count", 0.0)))
            expectancy = float(_to_num(row.get("expectancy", 0.0)))
            exp_lcb = float(_to_num(row.get("exp_lcb", 0.0)))
            combo = {
                "run_id": str(run_id),
                "trace_run_id": str(trace["run_id"]),
                "mode": clean_mode,
                "cost_level": str(cost_level),
                "stage": str(row.get("stage", "")),
                "symbol": str(row.get("symbol", "")),
                "timeframe": str(row.get("timeframe", "")),
                "family": str(row.get("family", "")),
                "composer": str(row.get("composer", "")),
                "exit_variant": exit_variant,
                "trade_count": trade_count,
                "tpm": float(_to_num(row.get("tpm", 0.0))),
                "exposure_ratio": float(_to_num(row.get("exposure_ratio", 0.0))),
                "PF_raw": float(_to_num(row.get("PF_raw", 0.0))),
                "PF_clipped": float(_to_num(row.get("PF_clipped", 0.0))),
                "expectancy": expectancy,
                "exp_lcb": exp_lcb,
                "maxDD": float(_to_num(row.get("maxDD", 0.0))),
                "walkforward_executed_true": bool(row.get("walkforward_executed_true", False)),
                "usable_windows": int(_to_num(row.get("usable_windows", 0))),
                "MC_triggered": bool(row.get("MC_triggered", False)),
                "top_reject_reason": str(row.get("top_reject_reason", "")),
                "classification": _combo_classification(
                    trade_count=trade_count,
                    expectancy=expectancy,
                    exp_lcb=exp_lcb,
                ),
            }
            rows.append(combo)

    results_df = pd.DataFrame(rows)
    if not results_df.empty:
        results_df = results_df.sort_values(
            ["cost_level", "family", "symbol", "timeframe", "exp_lcb", "expectancy"],
            ascending=[True, True, True, True, False, False],
        ).reset_index(drop=True)

    results_csv = out_dir / "family_results.csv"
    results_json = out_dir / "family_results.json"
    best_candidates_json = out_dir / "best_candidates.json"
    results_df.to_csv(results_csv, index=False)

    best_candidates = _best_candidates(results_df)
    best_candidates_json.write_text(json.dumps(best_candidates, indent=2, allow_nan=False), encoding="utf-8")

    summary = _build_summary(
        run_id=run_id,
        seed=int(seed),
        mode=clean_mode,
        dry_run=bool(dry_run),
        symbols=list(symbols),
        timeframes=list(timeframes),
        families=list(clean_families),
        composers=list(clean_composers),
        cost_levels=list(clean_cost_levels),
        rows_df=results_df,
        trace_refs=trace_refs,
        runtime_seconds=float(time.perf_counter() - started),
        snapshot_meta=snapshot_metadata_from_config(config),
    )
    results_json.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    docs_dir.mkdir(parents=True, exist_ok=True)
    report_md = docs_dir / "stage25B_edge_program_report.md"
    report_json = docs_dir / "stage25B_edge_program_summary.json"
    report_json.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    report_md.write_text(_render_report(summary), encoding="utf-8")

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "results_csv": results_csv,
        "results_json": results_json,
        "best_candidates_json": best_candidates_json,
        "report_md": report_md,
        "report_json": report_json,
        "summary": summary,
    }


def _prepare_cfg(*, config: dict[str, Any], mode: str, cost_level: str) -> dict[str, Any]:
    cfg = deepcopy(config)
    cfg.setdefault("universe", {})["base_timeframe"] = "1m"
    cfg.setdefault("data", {})["resample_source"] = "base"
    eval_cfg = cfg.setdefault("evaluation", {})
    eval_cfg.setdefault("constraints", {})
    eval_cfg["constraints"]["mode"] = str(mode)
    eval_cfg.setdefault("stage23", {})
    eval_cfg["stage23"]["enabled"] = True
    eval_cfg.setdefault("stage24", {})
    eval_cfg["stage24"]["enabled"] = False

    scenario = _cost_scenario(cfg=cfg, level=cost_level)
    v2 = cfg.setdefault("cost_model", {}).setdefault("v2", {})
    for key in ("slippage_bps_base", "slippage_bps_vol_mult", "spread_bps", "delay_bars"):
        if key in scenario:
            v2[key] = scenario[key]
    cfg.setdefault("cost_model", {})["mode"] = "v2"
    return cfg


def _cost_scenario(*, cfg: dict[str, Any], level: str) -> dict[str, Any]:
    clean = str(level).strip().lower()
    stage12 = (((cfg.get("evaluation", {}) or {}).get("stage12", {})) or {})
    scenarios = dict((stage12.get("cost_scenarios", {}) or {}))
    if clean in scenarios and isinstance(scenarios[clean], dict):
        chosen = dict(scenarios[clean])
        if bool(chosen.get("use_config_default", False)):
            return {}
        return chosen
    if clean == "low":
        return {"slippage_bps_base": 0.25, "slippage_bps_vol_mult": 1.0, "spread_bps": 0.25, "delay_bars": 0}
    if clean == "high":
        return {"slippage_bps_base": 1.5, "slippage_bps_vol_mult": 3.0, "spread_bps": 1.5, "delay_bars": 1}
    return {}


def _combo_classification(*, trade_count: float, expectancy: float, exp_lcb: float) -> str:
    if float(trade_count) <= 0:
        return "ZERO_TRADE"
    if float(exp_lcb) > 0 and float(expectancy) > 0:
        return "EDGE_CANDIDATE"
    return "NO_EDGE"


def _best_candidates(df: pd.DataFrame) -> dict[str, list[dict[str, Any]]]:
    if df.empty:
        return {}
    out: dict[str, list[dict[str, Any]]] = {}
    for family, group in df.groupby("family", dropna=False):
        ranked = group.sort_values(["exp_lcb", "expectancy", "trade_count"], ascending=[False, False, False]).head(3)
        out[str(family)] = ranked[
            ["mode", "cost_level", "symbol", "timeframe", "composer", "exit_variant", "trade_count", "expectancy", "exp_lcb", "maxDD"]
        ].to_dict(orient="records")
    return out


def _build_summary(
    *,
    run_id: str,
    seed: int,
    mode: str,
    dry_run: bool,
    symbols: list[str],
    timeframes: list[str],
    families: list[str],
    composers: list[str],
    cost_levels: list[str],
    rows_df: pd.DataFrame,
    trace_refs: list[dict[str, Any]],
    runtime_seconds: float,
    snapshot_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metrics = _aggregate_metrics(rows_df)
    best_candidates = _best_candidates(rows_df)
    has_edge = bool(
        not rows_df.empty
        and (
            (pd.to_numeric(rows_df["exp_lcb"], errors="coerce").fillna(0.0) > 0.0)
            & (pd.to_numeric(rows_df["trade_count"], errors="coerce").fillna(0.0) >= 10.0)
        ).any()
    )
    status = "EDGE_CANDIDATE_FOUND" if has_edge else ("NO_EDGE_IN_RESEARCH" if mode == "research" else "NO_EDGE_IN_LIVE")
    summary_hash = stable_hash(
        {
            "run_id": run_id,
            "seed": seed,
            "mode": mode,
            "rows_hash": stable_hash(rows_df.to_dict(orient="records"), length=16) if not rows_df.empty else stable_hash([], length=16),
            "trace_refs": trace_refs,
        },
        length=16,
    )
    return {
        "stage": "25B",
        "run_id": str(run_id),
        "seed": int(seed),
        "mode": str(mode),
        "dry_run": bool(dry_run),
        "symbols": list(symbols),
        "timeframes": list(timeframes),
        "families": list(families),
        "composers": list(composers),
        "cost_levels": list(cost_levels),
        "runtime_seconds": float(runtime_seconds),
        "trace_refs": trace_refs,
        "metrics": metrics,
        "best_candidates": best_candidates,
        "status": status,
        "summary_hash": summary_hash,
        **dict(snapshot_meta or {}),
    }


def _aggregate_metrics(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "rows": 0,
            "trade_count_total": 0.0,
            "tpm_median": 0.0,
            "exp_lcb_best": 0.0,
            "exp_lcb_median": 0.0,
            "expectancy_median": 0.0,
            "maxdd_median": 0.0,
            "zero_trade_pct": 100.0,
            "walkforward_executed_true_pct": 0.0,
            "mc_trigger_rate": 0.0,
            "invalid_pct": 100.0,
            "per_family": {},
        }
    trade_count = pd.to_numeric(df.get("trade_count", 0.0), errors="coerce").fillna(0.0)
    exp_lcb = pd.to_numeric(df.get("exp_lcb", 0.0), errors="coerce").fillna(0.0)
    expectancy = pd.to_numeric(df.get("expectancy", 0.0), errors="coerce").fillna(0.0)
    tpm = pd.to_numeric(df.get("tpm", 0.0), errors="coerce").fillna(0.0)
    maxdd = pd.to_numeric(df.get("maxDD", 0.0), errors="coerce").fillna(0.0)
    wf_true = pd.to_numeric(df.get("walkforward_executed_true", False), errors="coerce").fillna(0.0)
    mc_true = pd.to_numeric(df.get("MC_triggered", False), errors="coerce").fillna(0.0)
    invalid = pd.to_numeric(df.get("classification", "").astype(str) == "ZERO_TRADE", errors="coerce").fillna(0.0)
    per_family: dict[str, Any] = {}
    for family, group in df.groupby("family", dropna=False):
        g_trade = pd.to_numeric(group.get("trade_count", 0.0), errors="coerce").fillna(0.0)
        g_lcb = pd.to_numeric(group.get("exp_lcb", 0.0), errors="coerce").fillna(0.0)
        per_family[str(family)] = {
            "rows": int(len(group)),
            "trade_count_total": float(g_trade.sum()),
            "exp_lcb_best": float(g_lcb.max()),
            "exp_lcb_median": float(g_lcb.median()),
            "zero_trade_pct": float((g_trade <= 0.0).mean() * 100.0),
        }
    return {
        "rows": int(len(df)),
        "trade_count_total": float(trade_count.sum()),
        "tpm_median": float(tpm.median()),
        "exp_lcb_best": float(exp_lcb.max()),
        "exp_lcb_median": float(exp_lcb.median()),
        "expectancy_median": float(expectancy.median()),
        "maxdd_median": float(maxdd.median()),
        "zero_trade_pct": float((trade_count <= 0.0).mean() * 100.0),
        "walkforward_executed_true_pct": float(wf_true.mean() * 100.0),
        "mc_trigger_rate": float(mc_true.mean() * 100.0),
        "invalid_pct": float(invalid.mean() * 100.0),
        "per_family": per_family,
    }


def _render_report(summary: dict[str, Any]) -> str:
    metrics = dict(summary.get("metrics", {}))
    per_family = dict(metrics.get("per_family", {}))
    lines = [
        "# Stage-25B Edge Program Report",
        "",
        "## Scope",
        "- Family-level quality sweeps under fixed validation and costs.",
        "- Research mode isolates signal quality from exchange minima; live mode replays feasibility.",
        "",
        "## Run Context",
        f"- run_id: `{summary.get('run_id', '')}`",
        f"- seed: `{summary.get('seed', 0)}`",
        f"- mode: `{summary.get('mode', '')}`",
        f"- dry_run: `{summary.get('dry_run', False)}`",
        f"- symbols: `{summary.get('symbols', [])}`",
        f"- timeframes: `{summary.get('timeframes', [])}`",
        f"- cost_levels: `{summary.get('cost_levels', [])}`",
        "",
        "## Core Metrics",
        f"- rows: `{int(metrics.get('rows', 0))}`",
        f"- trade_count_total: `{float(metrics.get('trade_count_total', 0.0)):.6f}`",
        f"- exp_lcb_best: `{float(metrics.get('exp_lcb_best', 0.0)):.6f}`",
        f"- exp_lcb_median: `{float(metrics.get('exp_lcb_median', 0.0)):.6f}`",
        f"- expectancy_median: `{float(metrics.get('expectancy_median', 0.0)):.6f}`",
        f"- maxdd_median: `{float(metrics.get('maxdd_median', 0.0)):.6f}`",
        f"- zero_trade_pct: `{float(metrics.get('zero_trade_pct', 0.0)):.6f}`",
        f"- walkforward_executed_true_pct: `{float(metrics.get('walkforward_executed_true_pct', 0.0)):.6f}`",
        f"- mc_trigger_rate: `{float(metrics.get('mc_trigger_rate', 0.0)):.6f}`",
        "",
        "## Per-family Snapshot",
        "| family | rows | trade_count_total | exp_lcb_best | exp_lcb_median | zero_trade_pct |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family, item in sorted(per_family.items()):
        lines.append(
            f"| {family} | {int(item.get('rows', 0))} | {float(item.get('trade_count_total', 0.0)):.6f} | "
            f"{float(item.get('exp_lcb_best', 0.0)):.6f} | {float(item.get('exp_lcb_median', 0.0)):.6f} | "
            f"{float(item.get('zero_trade_pct', 0.0)):.6f} |"
        )

    lines.extend(
        [
            "",
            "## Best Candidates",
        ]
    )
    best = dict(summary.get("best_candidates", {}))
    if not best:
        lines.append("- none")
    for family, candidates in best.items():
        lines.append(f"- {family}:")
        for cand in list(candidates)[:3]:
            lines.append(
                "  - {symbol} {timeframe} cost={cost_level} exit={exit_variant} exp_lcb={exp_lcb:.6f} expectancy={expectancy:.6f} trades={trade_count:.0f}".format(
                    symbol=str(cand.get("symbol", "")),
                    timeframe=str(cand.get("timeframe", "")),
                    cost_level=str(cand.get("cost_level", "")),
                    exit_variant=str(cand.get("exit_variant", "")),
                    exp_lcb=float(cand.get("exp_lcb", 0.0)),
                    expectancy=float(cand.get("expectancy", 0.0)),
                    trade_count=float(cand.get("trade_count", 0.0)),
                )
            )

    lines.extend(
        [
            "",
            f"## Status\n- `{summary.get('status', '')}`",
            "",
            "## Artifacts",
            f"- runs/{summary.get('run_id', '')}/stage25B/family_results.csv",
            f"- runs/{summary.get('run_id', '')}/stage25B/family_results.json",
            f"- runs/{summary.get('run_id', '')}/stage25B/best_candidates.json",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _to_num(value: Any) -> float:
    try:
        num = float(value)
    except Exception:
        return 0.0
    if not np.isfinite(num):
        return 0.0
    return num


def run_stage25_master(
    *,
    config: dict[str, Any],
    seed: int,
    dry_run: bool,
    symbols: list[str],
    timeframes: list[str],
    families: list[str],
    composers: list[str],
    cost_levels: list[str],
    runs_root: Path,
    data_dir: Path,
    derived_dir: Path,
    docs_dir: Path = Path("docs"),
) -> dict[str, Any]:
    """Run Stage-25 research/live program and emit unified master report."""

    started = time.perf_counter()
    research = run_stage25b_edge_program(
        config=config,
        seed=int(seed),
        dry_run=bool(dry_run),
        mode="research",
        symbols=list(symbols),
        timeframes=list(timeframes),
        families=list(families),
        composers=list(composers),
        cost_levels=list(cost_levels),
        runs_root=runs_root,
        data_dir=data_dir,
        derived_dir=derived_dir,
        docs_dir=docs_dir,
    )
    live = run_stage25b_edge_program(
        config=config,
        seed=int(seed),
        dry_run=bool(dry_run),
        mode="live",
        symbols=list(symbols),
        timeframes=list(timeframes),
        families=list(families),
        composers=list(composers),
        cost_levels=list(cost_levels),
        runs_root=runs_root,
        data_dir=data_dir,
        derived_dir=derived_dir,
        docs_dir=docs_dir,
    )

    research_df = _read_csv(Path(research["results_csv"]))
    live_df = _read_csv(Path(live["results_csv"]))

    promising = _promising_research_candidates(research_df)
    replay = _live_replay(promising_df=promising, live_df=live_df)
    exit_upgrade = _exit_upgrade_summary(live_df)

    regime_live = _run_regime_conditional_live(
        config=config,
        seed=int(seed),
        dry_run=bool(dry_run),
        symbols=list(symbols),
        timeframes=list(timeframes),
        families=list(families),
        composers=list(composers),
        runs_root=runs_root,
        data_dir=data_dir,
        derived_dir=derived_dir,
    )

    research_summary = dict(research["summary"])
    live_summary = dict(live["summary"])
    regime_summary = dict(regime_live["summary"])
    verdict = _master_verdict(live_summary=live_summary, replay=replay)
    next_bottleneck = _master_bottleneck(live_summary)
    master = {
        "stage": "25",
        "seed": int(seed),
        "dry_run": bool(dry_run),
        "runtime_seconds": float(time.perf_counter() - started),
        "research_run_id": str(research_summary.get("run_id", "")),
        "live_run_id": str(live_summary.get("run_id", "")),
        "regime_run_id": str(regime_summary.get("run_id", "")),
        "research": {
            "status": str(research_summary.get("status", "")),
            "metrics": dict(research_summary.get("metrics", {})),
        },
        "live": {
            "status": str(live_summary.get("status", "")),
            "metrics": dict(live_summary.get("metrics", {})),
        },
        "promising_research_candidates_count": int(len(promising)),
        "live_replay": replay,
        "exit_upgrade": exit_upgrade,
        "regime_conditional": {
            "metrics": dict(regime_summary.get("metrics", {})),
            "deltas_vs_live": _metrics_delta(
                base=dict(live_summary.get("metrics", {})),
                other=dict(regime_summary.get("metrics", {})),
                keys=("exp_lcb_best", "exp_lcb_median", "trade_count_total", "zero_trade_pct", "walkforward_executed_true_pct", "mc_trigger_rate"),
            ),
        },
        "next_bottleneck": next_bottleneck,
        "final_verdict": verdict,
        "summary_hash": stable_hash(
            {
                "seed": int(seed),
                "research_hash": str(research_summary.get("summary_hash", "")),
                "live_hash": str(live_summary.get("summary_hash", "")),
                "regime_hash": str(regime_summary.get("summary_hash", "")),
                "replay": replay,
                "exit_upgrade": exit_upgrade,
                "verdict": verdict,
            },
            length=16,
        ),
    }

    docs_dir.mkdir(parents=True, exist_ok=True)
    report_json = docs_dir / "stage25_master_summary.json"
    report_md = docs_dir / "stage25_master_report.md"
    report_json.write_text(json.dumps(master, indent=2, allow_nan=False), encoding="utf-8")
    report_md.write_text(_render_master_report(master), encoding="utf-8")
    return {
        "report_md": report_md,
        "report_json": report_json,
        "summary": master,
        "research": research,
        "live": live,
        "regime": regime_live,
    }


def _run_regime_conditional_live(
    *,
    config: dict[str, Any],
    seed: int,
    dry_run: bool,
    symbols: list[str],
    timeframes: list[str],
    families: list[str],
    composers: list[str],
    runs_root: Path,
    data_dir: Path,
    derived_dir: Path,
) -> dict[str, Any]:
    cfg = _prepare_cfg(config=config, mode="live", cost_level="realistic")
    stage23 = cfg.setdefault("evaluation", {}).setdefault("stage23", {})
    stage23["enabled"] = True
    stage23.setdefault("eligibility", {})
    stage23["eligibility"]["per_regime_thresholds"] = {
        "TREND": 0.25,
        "RANGE": 0.30,
        "VOL_COMPRESSION": 0.30,
        "VOL_EXPANSION": 0.45,
        "CHOP": 0.65,
    }
    trace = run_signal_flow_trace(
        config=cfg,
        seed=int(seed),
        symbols=list(symbols),
        timeframes=list(timeframes),
        mode="v2",
        stages=["15", "17"],
        families=list(families),
        composers=list(composers),
        max_combos=0,
        dry_run=bool(dry_run),
        runs_root=runs_root,
        data_dir=data_dir,
        derived_dir=derived_dir,
    )
    rows_df = pd.DataFrame(trace.get("rows", pd.DataFrame()))
    summary = {
        "run_id": str(trace.get("run_id", "")),
        "trace_run_id": str(trace.get("run_id", "")),
        "trace_dir": str(trace.get("trace_dir", "")),
        "metrics": _aggregate_metrics(_rows_to_program_rows(rows_df, mode="live", cost_level="realistic")),
    }
    return {"summary": summary}


def _rows_to_program_rows(rows_df: pd.DataFrame, *, mode: str, cost_level: str) -> pd.DataFrame:
    if rows_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in rows_df.iterrows():
        exit_variant = "atr_trailing" if str(row.get("stage", "")) == "17" else "fixed_atr"
        trade_count = float(_to_num(row.get("trades_executed_count", 0.0)))
        expectancy = float(_to_num(row.get("expectancy", 0.0)))
        exp_lcb = float(_to_num(row.get("exp_lcb", 0.0)))
        rows.append(
            {
                "mode": str(mode),
                "cost_level": str(cost_level),
                "stage": str(row.get("stage", "")),
                "symbol": str(row.get("symbol", "")),
                "timeframe": str(row.get("timeframe", "")),
                "family": str(row.get("family", "")),
                "composer": str(row.get("composer", "")),
                "exit_variant": exit_variant,
                "trade_count": trade_count,
                "tpm": float(_to_num(row.get("tpm", 0.0))),
                "exposure_ratio": float(_to_num(row.get("exposure_ratio", 0.0))),
                "PF_raw": float(_to_num(row.get("PF_raw", 0.0))),
                "PF_clipped": float(_to_num(row.get("PF_clipped", 0.0))),
                "expectancy": expectancy,
                "exp_lcb": exp_lcb,
                "maxDD": float(_to_num(row.get("maxDD", 0.0))),
                "walkforward_executed_true": bool(row.get("walkforward_executed_true", False)),
                "usable_windows": int(_to_num(row.get("usable_windows", 0))),
                "MC_triggered": bool(row.get("MC_triggered", False)),
                "top_reject_reason": str(row.get("top_reject_reason", "")),
                "classification": _combo_classification(trade_count=trade_count, expectancy=expectancy, exp_lcb=exp_lcb),
            }
        )
    return pd.DataFrame(rows)


def _promising_research_candidates(research_df: pd.DataFrame) -> pd.DataFrame:
    if research_df.empty:
        return pd.DataFrame()
    safe = research_df.copy()
    safe["exp_lcb"] = pd.to_numeric(safe.get("exp_lcb", 0.0), errors="coerce").fillna(0.0)
    safe["trade_count"] = pd.to_numeric(safe.get("trade_count", 0.0), errors="coerce").fillna(0.0)
    filtered = safe.loc[(safe["exp_lcb"] > 0.0) & (safe["trade_count"] >= 10.0)].copy()
    if filtered.empty:
        filtered = safe.sort_values(["exp_lcb", "expectancy", "trade_count"], ascending=[False, False, False]).head(10)
    keep_cols = ["family", "symbol", "timeframe", "composer", "exit_variant", "cost_level", "exp_lcb", "expectancy", "trade_count"]
    return filtered.loc[:, [col for col in keep_cols if col in filtered.columns]].reset_index(drop=True)


def _live_replay(*, promising_df: pd.DataFrame, live_df: pd.DataFrame) -> dict[str, Any]:
    if promising_df.empty or live_df.empty:
        return {"rows": 0, "survived_count": 0, "survived_pct": 0.0, "exp_lcb_median": 0.0}
    keys = ["family", "symbol", "timeframe", "composer", "exit_variant", "cost_level"]
    left = promising_df.copy()
    right = live_df.copy()
    merged = left.merge(right, on=keys, how="left", suffixes=("_research", "_live"))
    live_trade = pd.to_numeric(merged.get("trade_count_live", 0.0), errors="coerce").fillna(0.0)
    live_lcb = pd.to_numeric(merged.get("exp_lcb_live", 0.0), errors="coerce").fillna(0.0)
    survived = live_trade > 0.0
    return {
        "rows": int(len(merged)),
        "survived_count": int(survived.sum()),
        "survived_pct": float(survived.mean() * 100.0) if len(merged) else 0.0,
        "exp_lcb_median": float(live_lcb.median()) if len(merged) else 0.0,
        "sample": merged.head(10).to_dict(orient="records"),
    }


def _exit_upgrade_summary(live_df: pd.DataFrame) -> dict[str, Any]:
    if live_df.empty:
        return {"rows": 0, "upgraded_count": 0, "fixed_selected": 0, "trailing_selected": 0}
    grouped = live_df.groupby(["family", "symbol", "timeframe", "cost_level", "composer"], dropna=False)
    rows = 0
    upgraded = 0
    fixed_selected = 0
    trailing_selected = 0
    for _, group in grouped:
        rows += 1
        ranked = group.sort_values(["exp_lcb", "expectancy", "trade_count"], ascending=[False, False, False])
        chosen = str(ranked.iloc[0].get("exit_variant", "fixed_atr"))
        if chosen == "atr_trailing":
            trailing_selected += 1
            upgraded += 1
        else:
            fixed_selected += 1
    return {
        "rows": int(rows),
        "upgraded_count": int(upgraded),
        "fixed_selected": int(fixed_selected),
        "trailing_selected": int(trailing_selected),
    }


def _metrics_delta(*, base: dict[str, Any], other: dict[str, Any], keys: tuple[str, ...]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in keys:
        out[key] = float(_to_num(other.get(key, 0.0)) - _to_num(base.get(key, 0.0)))
    return out


def _master_bottleneck(live_summary: dict[str, Any]) -> dict[str, Any]:
    metrics = dict(live_summary.get("metrics", {}))
    per_family = dict(metrics.get("per_family", {}))
    if not per_family:
        return {"type": "NO_DATA", "value": ""}
    ranked = sorted(
        per_family.items(),
        key=lambda item: (
            float(dict(item[1]).get("exp_lcb_best", 0.0)),
            -float(dict(item[1]).get("zero_trade_pct", 0.0)),
        ),
    )
    fam, data = ranked[0]
    return {
        "type": "WEAKEST_FAMILY",
        "value": str(fam),
        "exp_lcb_best": float(dict(data).get("exp_lcb_best", 0.0)),
        "zero_trade_pct": float(dict(data).get("zero_trade_pct", 0.0)),
    }


def _master_verdict(*, live_summary: dict[str, Any], replay: dict[str, Any]) -> str:
    metrics = dict(live_summary.get("metrics", {}))
    best_lcb = float(_to_num(metrics.get("exp_lcb_best", 0.0)))
    survived = int(replay.get("survived_count", 0))
    if best_lcb > 0.0 and survived > 0:
        return "WEAK_EDGE"
    return "NO_EDGE"


def _render_master_report(summary: dict[str, Any]) -> str:
    research = dict(summary.get("research", {}))
    live = dict(summary.get("live", {}))
    replay = dict(summary.get("live_replay", {}))
    exit_upgrade = dict(summary.get("exit_upgrade", {}))
    regime = dict(summary.get("regime_conditional", {}))
    lines = [
        "# Stage-25 Master Report",
        "",
        "## Scope",
        "- Stage-25A: margin/caps correctness and research/live constraint split.",
        "- Stage-25B: family quality in research mode and live feasibility replay.",
        "- Stage-25.5: minimal exit/regime conditional levers with strict validation preserved.",
        "",
        "## Run IDs",
        f"- research_run_id: `{summary.get('research_run_id', '')}`",
        f"- live_run_id: `{summary.get('live_run_id', '')}`",
        f"- regime_run_id: `{summary.get('regime_run_id', '')}`",
        "",
        "## Research vs Live",
        f"- research status: `{research.get('status', '')}`",
        f"- live status: `{live.get('status', '')}`",
        f"- research exp_lcb_best: `{float(dict(research.get('metrics', {})).get('exp_lcb_best', 0.0)):.6f}`",
        f"- live exp_lcb_best: `{float(dict(live.get('metrics', {})).get('exp_lcb_best', 0.0)):.6f}`",
        "",
        "## Live Feasibility Replay",
        f"- promising_research_candidates_count: `{int(summary.get('promising_research_candidates_count', 0))}`",
        f"- survived_count: `{int(replay.get('survived_count', 0))}` / `{int(replay.get('rows', 0))}`",
        f"- survived_pct: `{float(replay.get('survived_pct', 0.0)):.6f}`",
        f"- live_replay_exp_lcb_median: `{float(replay.get('exp_lcb_median', 0.0)):.6f}`",
        "",
        "## Minimal Improvement Levers",
        "- Exit upgrade selection:",
        f"  - rows: `{int(exit_upgrade.get('rows', 0))}`",
        f"  - trailing_selected: `{int(exit_upgrade.get('trailing_selected', 0))}`",
        f"  - fixed_selected: `{int(exit_upgrade.get('fixed_selected', 0))}`",
        f"  - upgraded_count: `{int(exit_upgrade.get('upgraded_count', 0))}`",
        "- Regime-conditional activation deltas vs live baseline:",
    ]
    for key, value in dict(regime.get("deltas_vs_live", {})).items():
        lines.append(f"  - {key}: `{float(value):.6f}`")
    lines.extend(
        [
            "",
            "## Bottleneck",
            f"- next_bottleneck: `{summary.get('next_bottleneck', {})}`",
            "",
            f"## Final Verdict\n- `{summary.get('final_verdict', '')}`",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)
