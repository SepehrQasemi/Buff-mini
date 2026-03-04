"""Stage-15.9 signal-flow tracing and bottleneck analysis."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from buffmini.alpha_v2.context import compute_context_states, context_weight_series
from buffmini.alpha_v2.mtf import MtfPolicyConfig, apply_mtf_policy, causal_join_bias
from buffmini.alpha_v2.transitions import combined_transition_score
from buffmini.backtest.engine import run_backtest
from buffmini.baselines.stage0 import generate_signals, trend_pullback
from buffmini.config import compute_config_hash
from buffmini.signals.composer import compose_signals, normalize_weights
from buffmini.signals.family_base import FamilyContext
from buffmini.signals.registry import build_families, family_names
from buffmini.stage23.eligibility import evaluate_eligibility
from buffmini.stage23.order_builder import build_adaptive_orders
from buffmini.stage23.rejects import RejectBreakdown
from buffmini.stage10.evaluate import _build_features
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact
from buffmini.validation.walkforward_v2 import build_windows


DEFAULT_TIMEFRAMES = ["15m", "30m", "1h", "2h", "4h"]
DEFAULT_STAGES = ["classic", "15", "16", "17", "18", "19", "20", "21", "22"]
DEFAULT_COMPOSERS = ["none", "vote", "weighted_sum", "gated"]


@dataclass(frozen=True)
class StageProfile:
    stage: str
    stage_type: str
    context_enabled: bool
    transition_enabled: bool
    mtf_enabled: bool
    trading: bool
    exit_mode: str
    expect_walkforward: bool
    expect_mc: bool


STAGE_PROFILES: dict[str, StageProfile] = {
    "classic": StageProfile("classic", "trading", False, False, False, True, "fixed_atr", True, True),
    "15": StageProfile("15", "trading", False, False, False, True, "fixed_atr", True, True),
    "16": StageProfile("16", "trading", True, False, False, True, "fixed_atr", True, True),
    "17": StageProfile("17", "trading", True, False, False, True, "trailing_atr", True, True),
    "18": StageProfile("18", "non_trading", True, False, False, False, "fixed_atr", False, False),
    "19": StageProfile("19", "trading", True, True, False, True, "fixed_atr", True, True),
    "20": StageProfile("20", "non_trading", True, False, False, False, "fixed_atr", False, False),
    "21": StageProfile("21", "non_trading", True, False, False, False, "fixed_atr", False, False),
    "22": StageProfile("22", "trading", True, False, True, True, "fixed_atr", True, True),
}


def parse_csv_list(value: str | None, *, default: list[str]) -> list[str]:
    if value is None:
        return list(default)
    out = [item.strip() for item in str(value).split(",") if item.strip()]
    return out or list(default)


def parse_stage_arg(value: str | None) -> list[str]:
    if value is None:
        return list(DEFAULT_STAGES)
    text = str(value).strip()
    if text == "15..22":
        return ["15", "16", "17", "18", "19", "20", "21", "22"]
    out = [item.strip() for item in text.split(",") if item.strip()]
    return out or list(DEFAULT_STAGES)


def run_signal_flow_trace(
    *,
    config: dict[str, Any],
    seed: int,
    symbols: list[str],
    timeframes: list[str],
    mode: str,
    stages: list[str],
    families: list[str],
    composers: list[str],
    max_combos: int,
    dry_run: bool,
    runs_root: Path,
    data_dir: Path,
    derived_dir: Path,
) -> dict[str, Any]:
    """Run deterministic full-flow trace and write artifacts under runs/<run_id>/trace."""

    started = time.perf_counter()
    cfg = json.loads(json.dumps(config))
    cfg.setdefault("universe", {})["base_timeframe"] = "1m"
    cfg.setdefault("data", {})["resample_source"] = "base"
    cfg_hash = compute_config_hash(cfg)
    run_seed = int(seed)
    selected_mode = str(mode).strip().lower()
    if selected_mode not in {"classic", "v2", "both"}:
        raise ValueError("mode must be classic|v2|both")

    valid_stages = [s for s in stages if s in STAGE_PROFILES]
    if not valid_stages:
        raise ValueError("no valid stages selected")
    valid_families = [f for f in families if f in family_names()]
    if not valid_families:
        valid_families = family_names()
    valid_composers = [c for c in composers if c in DEFAULT_COMPOSERS]
    if not valid_composers:
        valid_composers = list(DEFAULT_COMPOSERS)

    run_id = (
        f"{utc_now_compact()}_"
        f"{stable_hash({'seed': run_seed, 'stages': valid_stages, 'timeframes': timeframes, 'mode': selected_mode, 'symbols': symbols, 'families': valid_families, 'composers': valid_composers, 'config_hash': cfg_hash}, length=12)}"
        "_stage15_9_trace"
    )
    run_dir = runs_root / run_id
    trace_dir = run_dir / "trace"
    trace_dir.mkdir(parents=True, exist_ok=True)

    feature_started = time.perf_counter()
    features_cache: dict[tuple[str, str], pd.DataFrame] = {}
    load_seconds_by_tf: dict[str, float] = {}
    for tf in timeframes:
        tf_t0 = time.perf_counter()
        frames = _build_features(
            config=cfg,
            symbols=symbols,
            timeframe=str(tf),
            dry_run=bool(dry_run),
            seed=run_seed,
            data_dir=data_dir,
            derived_dir=derived_dir,
        )
        load_seconds_by_tf[str(tf)] = float(time.perf_counter() - tf_t0)
        for symbol, frame in frames.items():
            work = frame.copy().reset_index(drop=True)
            work.attrs["symbol"] = symbol
            work.attrs["timeframe"] = str(tf)
            features_cache[(symbol, str(tf))] = work
    feature_compute_seconds = float(time.perf_counter() - feature_started)
    if not features_cache:
        raise RuntimeError("trace: no features loaded")

    combos = _build_combo_rows(
        selected_mode=selected_mode,
        stages=valid_stages,
        symbols=symbols,
        timeframes=timeframes,
        families=valid_families,
        composers=valid_composers,
        max_combos=max_combos,
    )

    stage13_cfg = (((cfg.get("evaluation", {}) or {}).get("stage13", {})) or {})
    composer_weights = normalize_weights(
        dict(stage13_cfg.get("composer", {}).get("weights", {})),
        list(valid_families),
    )
    family_objects = build_families(enabled=valid_families, cfg=cfg)
    if not family_objects:
        raise RuntimeError("trace: no family instances")

    family_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    composer_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
    context_cache: dict[tuple[str, str], pd.Series] = {}
    rows: list[dict[str, Any]] = []
    reject_reason_rows: list[dict[str, Any]] = []
    execution_reject_events: list[dict[str, Any]] = []
    execution_adjustment_events: list[dict[str, Any]] = []
    eligibility_trace_rows: list[dict[str, Any]] = []
    sizing_trace_rows: list[dict[str, Any]] = []
    stage24_sizing_trace_rows: list[dict[str, Any]] = []
    shadow_live_rows: list[dict[str, Any]] = []
    research_infeasible_rows: list[dict[str, Any]] = []
    execution_breakdown = RejectBreakdown()

    for combo in combos:
        symbol = combo["symbol"]
        timeframe = combo["timeframe"]
        stage = combo["stage"]
        mode_name = combo["mode"]
        family = combo["family"]
        composer = combo["composer"]
        profile = STAGE_PROFILES[stage]
        frame = features_cache.get((symbol, timeframe))
        if frame is None or frame.empty:
            continue
        work = frame.copy()
        if profile.context_enabled:
            context_key = (symbol, timeframe)
            if context_key not in context_cache:
                context_cache[context_key] = context_weight_series(compute_context_states(work))
            context_weights = context_cache[context_key]
        else:
            context_weights = pd.Series(1.0, index=work.index, dtype=float)

        signal_t0 = time.perf_counter()
        if mode_name == "classic":
            output = _classic_output(work)
            required_cols = ["timestamp", "open", "high", "low", "close", "volume", "atr_14"]
        else:
            if composer == "none":
                cache_key = (symbol, timeframe, family)
                if cache_key not in family_cache:
                    fam_obj = family_objects[family]
                    fam_ctx = FamilyContext(symbol=symbol, timeframe=timeframe, seed=run_seed, config=cfg, params={})
                    fam_scores = fam_obj.compute_scores(work, fam_ctx)
                    family_cache[cache_key] = fam_obj.propose_entries(fam_scores, work, fam_ctx)
                output = family_cache[cache_key]
                required_cols = list(family_objects[family].required_features())
            else:
                cache_key2 = (symbol, timeframe, composer)
                if cache_key2 not in composer_cache:
                    family_outputs: dict[str, pd.DataFrame] = {}
                    for fam in valid_families:
                        key = (symbol, timeframe, fam)
                        if key not in family_cache:
                            fam_obj2 = family_objects[fam]
                            fam_ctx2 = FamilyContext(symbol=symbol, timeframe=timeframe, seed=run_seed, config=cfg, params={})
                            fam_scores2 = fam_obj2.compute_scores(work, fam_ctx2)
                            family_cache[key] = fam_obj2.propose_entries(fam_scores2, work, fam_ctx2)
                        family_outputs[fam] = family_cache[key]
                    composer_cache[cache_key2] = compose_signals(
                        family_outputs=family_outputs,
                        mode=composer,
                        weights=composer_weights,
                        gated_config=dict(stage13_cfg.get("composer", {}).get("gated", {})),
                    )
                output = composer_cache[cache_key2]
                required_cols = sorted({c for fam in valid_families for c in family_objects[fam].required_features()})
        signal_compute_seconds = float(time.perf_counter() - signal_t0)

        feature_stats = _feature_stats(work, required_cols)
        counts = _count_flow(
            frame=work,
            output=output,
            context_weights=context_weights,
            stage_profile=profile,
            cfg=cfg,
            symbol=symbol,
            family=family,
        )
        signal_for_backtest = counts["signal_series"]
        runtime_backtest = 0.0
        runtime_wf = 0.0
        runtime_mc = 0.0
        bt_metrics = _empty_bt_metrics()
        wf = _empty_wf_metrics()
        mc = _empty_mc_metrics()
        if profile.trading:
            bt_t0 = time.perf_counter()
            bt_metrics = _run_backtest_metrics(
                frame=work,
                symbol=symbol,
                signal=signal_for_backtest,
                cfg=cfg,
                strategy_name=f"trace::{stage}::{mode_name}::{family}::{composer}",
                exit_mode=profile.exit_mode,
            )
            runtime_backtest = float(time.perf_counter() - bt_t0)
            wf_t0 = time.perf_counter()
            wf = _walkforward_trace(
                frame=work,
                signal=signal_for_backtest,
                cfg=cfg,
            )
            runtime_wf = float(time.perf_counter() - wf_t0)
            mc_t0 = time.perf_counter()
            mc = _mc_trace(
                trade_count=float(bt_metrics["trades_executed_count"]),
                trade_pnls=np.asarray(bt_metrics["trade_pnls"], dtype=float),
                cfg=cfg,
            )
            runtime_mc = float(time.perf_counter() - mc_t0)

        row = {
            "stage": stage,
            "stage_type": profile.stage_type,
            "mode": mode_name,
            "symbol": symbol,
            "timeframe": timeframe,
            "family": family,
            "composer": composer,
            **feature_stats,
            **counts,
            **bt_metrics,
            **wf,
            **mc,
            "runtime_load_resample_seconds": float(load_seconds_by_tf.get(timeframe, 0.0)) * 0.5,
            "runtime_feature_compute_seconds": float(load_seconds_by_tf.get(timeframe, 0.0)) * 0.5,
            "runtime_signal_compute_seconds": signal_compute_seconds,
            "runtime_backtest_seconds": runtime_backtest,
            "runtime_walkforward_seconds": runtime_wf,
            "runtime_mc_seconds": runtime_mc,
            "runtime_reporting_seconds": 0.0,
        }
        attempted = int(row.get("orders_attempted_count", row.get("orders_sent_count", 0)) or 0)
        accepted = int(row.get("orders_sent_count", 0) or 0)
        execution_breakdown.register_attempt(attempted)
        execution_breakdown.register_accept(accepted)
        for event in counts.get("execution_reject_events", []):
            reason = str(event.get("reason", "UNKNOWN"))
            execution_breakdown.register_reject(reason, 1)
            execution_reject_events.append(
                {
                    "stage": stage,
                    "mode": mode_name,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "family": family,
                    "composer": composer,
                    "timestamp": str(event.get("timestamp", "")),
                    "side": str(event.get("side", "MIXED")),
                    "reason": reason,
                    "count": 1,
                    "details": str(event.get("details", "")),
                }
            )
        execution_adjustment_events.extend(list(counts.get("execution_adjustment_events", [])))
        eligibility_trace_rows.extend(list(counts.get("eligibility_trace_rows", [])))
        for item in counts.get("sizing_trace_rows", []):
            sizing_trace_rows.append(
                {
                    "stage": stage,
                    "mode": mode_name,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "family": family,
                    "composer": composer,
                    **dict(item),
                }
            )
        for item in counts.get("stage24_sizing_trace_rows", []):
            stage24_sizing_trace_rows.append(
                {
                    "stage": stage,
                    "mode": mode_name,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "family": family,
                    "composer": composer,
                    **dict(item),
                }
            )
        shadow_live_payload = dict(counts.get("shadow_live_summary", {}))
        if shadow_live_payload:
            shadow_live_rows.append(
                {
                    "stage": stage,
                    "mode": mode_name,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "family": family,
                    "composer": composer,
                    **shadow_live_payload,
                }
            )
        for item in counts.get("research_infeasible_flags", []):
            research_infeasible_rows.append(
                {
                    "stage": stage,
                    "mode": mode_name,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "family": family,
                    "composer": composer,
                    **dict(item),
                }
            )

        row["top_reject_reason"] = _top_reject_reason(row)
        rows.append(_safe_json(row))
        reject_reason_rows.append(
            {
                "stage": stage,
                "mode": mode_name,
                "symbol": symbol,
                "timeframe": timeframe,
                "family": family,
                "composer": composer,
                "reason": row["top_reject_reason"],
                "raw_signal_count": row["raw_signal_count"],
                "after_context_count": row["after_context_count"],
                "after_riskgate_count": row["after_riskgate_count"],
                "orders_sent_count": row["orders_sent_count"],
                "trades_executed_count": row["trades_executed_count"],
                "walkforward_usable_windows": row["usable_windows"],
            }
        )

    rows_df = pd.DataFrame(rows)
    if rows_df.empty:
        raise RuntimeError("trace: no rows produced")

    summary = summarize_trace(rows_df=rows_df, reject_reasons=pd.DataFrame(reject_reason_rows))
    summary["run_id"] = run_id
    summary["seed"] = run_seed
    summary["config_hash"] = cfg_hash
    summary["data_hash"] = _trace_data_hash(features_cache)
    summary["resolved_end_ts"] = _resolved_end_ts(features_cache)
    summary["head_commit"] = _git_head()
    summary["runtime_seconds"] = float(time.perf_counter() - started)
    summary["feature_compute_total_seconds"] = feature_compute_seconds

    rows_df.to_csv(trace_dir / "signal_flow_trace.csv", index=False)
    (trace_dir / "signal_flow_trace.json").write_text(
        json.dumps({"rows": rows, "summary": summary}, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.DataFrame(reject_reason_rows).to_csv(trace_dir / "reject_reasons.csv", index=False)
    pd.DataFrame(eligibility_trace_rows).to_csv(trace_dir / "eligibility_trace.csv", index=False)
    pd.DataFrame(execution_reject_events).to_csv(trace_dir / "execution_reject_events.csv", index=False)
    pd.DataFrame(execution_adjustment_events).to_csv(trace_dir / "execution_adjustments.csv", index=False)
    sizing_trace_df = pd.DataFrame(sizing_trace_rows)
    sizing_trace_df.to_csv(trace_dir / "sizing_trace.csv", index=False)
    (trace_dir / "sizing_trace_summary.json").write_text(
        json.dumps(_aggregate_sizing_trace_summary(sizing_trace_df), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    stage24_trace_df = pd.DataFrame(stage24_sizing_trace_rows)
    stage24_trace_df.to_csv(trace_dir / "stage24_sizing_trace.csv", index=False)
    (trace_dir / "stage24_sizing_summary.json").write_text(
        json.dumps(_aggregate_stage24_trace_summary(stage24_trace_df), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    shadow_live_df = pd.DataFrame(shadow_live_rows)
    shadow_live_df.to_csv(trace_dir / "shadow_live_feasibility.csv", index=False)
    (trace_dir / "shadow_live_summary.json").write_text(
        json.dumps(_aggregate_shadow_live_summary(shadow_live_df), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    pd.DataFrame(research_infeasible_rows).to_csv(trace_dir / "research_infeasible_flags.csv", index=False)
    (trace_dir / "execution_reject_breakdown.json").write_text(
        json.dumps(execution_breakdown.to_payload(), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (trace_dir / "trace_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "trace_dir": trace_dir,
        "rows": rows_df,
        "reject_reasons": pd.DataFrame(reject_reason_rows),
        "summary": summary,
    }


def summarize_trace(rows_df: pd.DataFrame, reject_reasons: pd.DataFrame) -> dict[str, Any]:
    """Aggregate bottlenecks, death rates, and pass/fail by stage."""

    df = rows_df.copy()
    for col in (
        "raw_signal_count",
        "after_context_count",
        "after_confirm_count",
        "after_riskgate_count",
        "orders_sent_count",
        "trades_executed_count",
    ):
        df[col] = pd.to_numeric(df.get(col, 0.0), errors="coerce").fillna(0.0)
    df["death_context"] = _death_rate(df["raw_signal_count"], df["after_context_count"])
    df["death_confirm"] = _death_rate(df["after_context_count"], df["after_confirm_count"])
    df["death_riskgate"] = _death_rate(df["after_confirm_count"], df["after_riskgate_count"])
    df["death_orders"] = _death_rate(df["after_riskgate_count"], df["orders_sent_count"])
    df["death_execution"] = _death_rate(df["orders_sent_count"], df["trades_executed_count"])
    gate_cols = ["death_context", "death_confirm", "death_riskgate", "death_orders", "death_execution"]

    def _agg_group(group_col: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for key, g in df.groupby(group_col, dropna=False):
            means = {c: float(pd.to_numeric(g[c], errors="coerce").fillna(0.0).mean()) for c in gate_cols}
            top_gate = max(means.items(), key=lambda kv: kv[1])[0]
            out.append(
                {
                    group_col: str(key),
                    "rows": int(len(g)),
                    "top_bottleneck_gate": top_gate,
                    "top_bottleneck_death_rate": float(means[top_gate]),
                    **means,
                }
            )
        return sorted(out, key=lambda x: x["top_bottleneck_death_rate"], reverse=True)

    overall_gate_means = {c: float(pd.to_numeric(df[c], errors="coerce").fillna(0.0).mean()) for c in gate_cols}
    top_overall = sorted(overall_gate_means.items(), key=lambda kv: kv[1], reverse=True)
    top_bottlenecks = [{"gate": gate, "death_rate": float(rate)} for gate, rate in top_overall[:5]]

    stage_pass_fail: dict[str, dict[str, int]] = {}
    for stage, g in df.groupby("stage", dropna=False):
        stage_key = str(stage)
        failed = int((pd.to_numeric(g["trades_executed_count"], errors="coerce").fillna(0.0) <= 0.0).sum())
        stage_pass_fail[stage_key] = {
            "rows": int(len(g)),
            "failed_like_zero_trade_rows": failed,
            "pass_like_rows": int(len(g) - failed),
        }

    reject_counts = (
        reject_reasons.groupby(["reason"], dropna=False).size().reset_index(name="count").sort_values("count", ascending=False)
    )
    return {
        "rows_count": int(len(df)),
        "top_bottlenecks": top_bottlenecks,
        "overall_gate_means": overall_gate_means,
        "by_stage": _agg_group("stage"),
        "by_timeframe": _agg_group("timeframe"),
        "by_family": _agg_group("family"),
        "by_mode": _agg_group("mode"),
        "stage_pass_fail": stage_pass_fail,
        "reject_reason_counts": reject_counts.to_dict(orient="records"),
        "zero_trade_pct": float((pd.to_numeric(df["trades_executed_count"], errors="coerce").fillna(0.0) <= 0.0).mean() * 100.0),
        "invalid_pct": float((df["top_reject_reason"].astype(str) != "VALID").mean() * 100.0),
        "walkforward_executed_true_pct": float(pd.to_numeric(df["walkforward_executed_true"], errors="coerce").fillna(0.0).mean() * 100.0),
        "mc_trigger_rate": float(pd.to_numeric(df["MC_triggered"], errors="coerce").fillna(0.0).mean() * 100.0),
    }


def write_stage15_9_report(
    *,
    report_md: Path,
    report_json: Path,
    pre_summary: dict[str, Any],
    post_summary: dict[str, Any],
    pre_run_id: str,
    post_run_id: str,
    defects_fixed: list[str],
) -> None:
    """Write Stage-15.9 report and summary JSON."""

    deltas = {
        "zero_trade_pct": float(post_summary.get("zero_trade_pct", 0.0) - pre_summary.get("zero_trade_pct", 0.0)),
        "invalid_pct": float(post_summary.get("invalid_pct", 0.0) - pre_summary.get("invalid_pct", 0.0)),
        "walkforward_executed_true_pct": float(
            post_summary.get("walkforward_executed_true_pct", 0.0) - pre_summary.get("walkforward_executed_true_pct", 0.0)
        ),
        "mc_trigger_rate": float(post_summary.get("mc_trigger_rate", 0.0) - pre_summary.get("mc_trigger_rate", 0.0)),
    }
    payload = {
        "head_commit": _git_head(),
        "run_ids": {"pre_fix": pre_run_id, "post_fix": post_run_id},
        "top_bottlenecks": post_summary.get("top_bottlenecks", []),
        "per_tf": post_summary.get("by_timeframe", []),
        "per_family": post_summary.get("by_family", []),
        "stage_pass_fail": post_summary.get("stage_pass_fail", {}),
        "before_after": {"pre": pre_summary, "post": post_summary, "delta": deltas},
        "defects_fixed": list(defects_fixed),
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(_safe_json(payload), indent=2, allow_nan=False), encoding="utf-8")

    lines = [
        "# Stage-15.9 Signal Flow Bottleneck Report",
        "",
        "## 1) Executive Summary",
        f"- pre_fix run_id: `{pre_run_id}`",
        f"- post_fix run_id: `{post_run_id}`",
        f"- top bottlenecks: `{post_summary.get('top_bottlenecks', [])}`",
        "",
        "## 2) System Flow Diagram",
        "- raw -> context -> confirm -> riskgate -> orders -> trades -> WF -> MC",
        "",
        "## 3) Bottleneck Tables",
        "### Overall",
    ]
    for item in post_summary.get("top_bottlenecks", []):
        lines.append(f"- {item.get('gate')}: death_rate={float(item.get('death_rate', 0.0)):.6f}")
    lines.extend(["", "### Per Stage"])
    for row in post_summary.get("by_stage", []):
        lines.append(
            f"- stage={row.get('stage')} top_gate={row.get('top_bottleneck_gate')} death_rate={float(row.get('top_bottleneck_death_rate', 0.0)):.6f}"
        )
    lines.extend(["", "### Per Timeframe"])
    for row in post_summary.get("by_timeframe", []):
        lines.append(
            f"- tf={row.get('timeframe')} top_gate={row.get('top_bottleneck_gate')} death_rate={float(row.get('top_bottleneck_death_rate', 0.0)):.6f}"
        )
    lines.extend(["", "### Per Family"])
    for row in post_summary.get("by_family", []):
        lines.append(
            f"- family={row.get('family')} top_gate={row.get('top_bottleneck_gate')} death_rate={float(row.get('top_bottleneck_death_rate', 0.0)):.6f}"
        )
    lines.extend(
        [
            "",
            "## 4) Findings",
            "- Bug-like findings are listed under Fixes Applied.",
            "- Design bottlenecks are retained and reported without relaxing gates.",
            "",
            "## 5) Fixes Applied",
        ]
    )
    if defects_fixed:
        for item in defects_fixed:
            lines.append(f"- {item}")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## 6) Before/After Deltas",
            f"- zero_trade_pct delta: `{deltas['zero_trade_pct']:.6f}`",
            f"- invalid_pct delta: `{deltas['invalid_pct']:.6f}`",
            f"- walkforward_executed_true_pct delta: `{deltas['walkforward_executed_true_pct']:.6f}`",
            f"- mc_trigger_rate delta: `{deltas['mc_trigger_rate']:.6f}`",
            "",
            "## 7) Next Steps",
            "- Tune signal-generation density (score/threshold shaping) by family where raw counts collapse.",
            "- Inspect context and risk-gate death rates on weak timeframes before search-space expansion.",
            "- Prioritize timeframes/families with non-zero WF execution before MC-heavy sweeps.",
        ]
    )
    report_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _build_combo_rows(
    *,
    selected_mode: str,
    stages: list[str],
    symbols: list[str],
    timeframes: list[str],
    families: list[str],
    composers: list[str],
    max_combos: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for stage in stages:
        if stage == "classic":
            if selected_mode not in {"classic", "both"}:
                continue
            for symbol in symbols:
                for tf in timeframes:
                    rows.append(
                        {
                            "stage": "classic",
                            "mode": "classic",
                            "symbol": symbol,
                            "timeframe": str(tf),
                            "family": "classic_trend_pullback",
                            "composer": "none",
                        }
                    )
            continue
        if selected_mode not in {"v2", "both"}:
            continue
        for symbol in symbols:
            for tf in timeframes:
                for fam in families:
                    rows.append(
                        {
                            "stage": stage,
                            "mode": "v2",
                            "symbol": symbol,
                            "timeframe": str(tf),
                            "family": fam,
                            "composer": "none",
                        }
                    )
                for comp in composers:
                    if comp == "none":
                        continue
                    rows.append(
                        {
                            "stage": stage,
                            "mode": "v2",
                            "symbol": symbol,
                            "timeframe": str(tf),
                            "family": "combined",
                            "composer": comp,
                        }
                    )
    if max_combos > 0 and len(rows) > int(max_combos):
        rows = rows[: int(max_combos)]
    return rows


def _classic_output(frame: pd.DataFrame) -> pd.DataFrame:
    strategy = trend_pullback()
    signal = generate_signals(frame.copy(), strategy=strategy, gating_mode="none")
    direction = pd.to_numeric(signal, errors="coerce").fillna(0).astype(int)
    score = pd.Series(direction.to_numpy(dtype=float), index=frame.index, dtype=float)
    confidence = score.abs().clip(0.0, 1.0)
    return pd.DataFrame(
        {
            "score": score,
            "direction": direction,
            "confidence": confidence,
            "reasons": pd.Series(np.where(direction > 0, "classic_long", np.where(direction < 0, "classic_short", ""))),
            "signal": direction.astype(int),
            "signal_family": "classic_trend_pullback",
        }
    )


def _feature_stats(frame: pd.DataFrame, required_cols: list[str]) -> dict[str, Any]:
    cols = [c for c in required_cols if c in frame.columns]
    missing = sorted(set(required_cols).difference(cols))
    rows_total = int(len(frame))
    if cols:
        req = frame.loc[:, cols]
        rows_after_dropna = int(len(req.dropna()))
        nan_rate = float(req.isna().mean().mean())
    else:
        rows_after_dropna = 0
        nan_rate = 1.0
    return {
        "rows_total": rows_total,
        "rows_after_dropna": rows_after_dropna,
        "nan_rate_required_cols": nan_rate,
        "missing_feature_columns": "|".join(missing),
    }


def _count_flow(
    *,
    frame: pd.DataFrame,
    output: pd.DataFrame,
    context_weights: pd.Series,
    stage_profile: StageProfile,
    cfg: dict[str, Any],
    symbol: str,
    family: str,
) -> dict[str, Any]:
    score = pd.to_numeric(output.get("score", 0.0), errors="coerce").fillna(0.0)
    direction = pd.to_numeric(output.get("direction", 0), errors="coerce").fillna(0).astype(int)
    raw_signal_count = int((direction != 0).sum())
    raw_signal_rate = float(raw_signal_count * 1000.0 / max(1, len(frame)))

    raw_dir = direction.to_numpy(dtype=int)
    raw_active = raw_dir != 0
    if stage_profile.context_enabled:
        weighted_score = score.to_numpy(dtype=float) * pd.to_numeric(context_weights, errors="coerce").fillna(1.0).to_numpy(dtype=float)
        keep = raw_active & (np.abs(weighted_score) >= 0.25)
        after_context_direction = np.where(keep, raw_dir, 0).astype(int)
        after_context_count = int(np.count_nonzero(keep))
        context_reject_count = max(0, raw_signal_count - after_context_count)
        context_reasons = "context_weight_zero_or_low" if context_reject_count > 0 else ""
    else:
        weighted_score = score.to_numpy(dtype=float)
        after_context_direction = raw_dir
        after_context_count = int(np.count_nonzero(raw_active))
        context_reject_count = 0
        context_reasons = ""

    after_confirm_count = after_context_count
    after_riskgate_count = after_context_count
    cooldown_reject_count = 0
    conflict_reject_count = 0
    final_side = np.sign(after_context_direction).astype(int)
    eligibility_trace_rows: list[dict[str, Any]] = []

    if stage_profile.transition_enabled:
        trans = combined_transition_score(frame)
        trans_num = pd.to_numeric(trans, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        trans_side = np.where(trans_num >= 0.2, 1, np.where(trans_num <= -0.2, -1, 0))
        final_side = np.sign(0.7 * final_side.astype(float) + 0.3 * trans_side.astype(float)).astype(int)

    if stage_profile.mtf_enabled:
        bias_df = frame.iloc[::4][["timestamp", "ema_slope_50"]].copy()
        bias_df["bias_score"] = np.tanh(pd.to_numeric(bias_df["ema_slope_50"], errors="coerce").fillna(0.0) / 0.01)
        joined = causal_join_bias(base_df=frame[["timestamp"]], bias_df=bias_df[["timestamp", "bias_score"]], bias_col="bias_score")
        policy = pd.DataFrame(
            {
                "timestamp": frame["timestamp"],
                "entry_score": pd.Series(final_side, index=frame.index, dtype=float),
                "bias_score": pd.to_numeric(joined["bias_score"], errors="coerce").fillna(0.0),
            }
        )
        mtf_signal, stats = apply_mtf_policy(
            base_df=policy,
            entry_score_col="entry_score",
            bias_score_col="bias_score",
            cfg=MtfPolicyConfig(bias_threshold=0.02, entry_threshold=0.15, conflict_mode="net"),
        )
        final_side = pd.to_numeric(mtf_signal, errors="coerce").fillna(0).astype(int).to_numpy(dtype=int)
        conflict_reject_count = int(round(float(stats.get("conflict_rate_pct", 0.0)) * len(frame) / 100.0))

    signal_pre = pd.Series(final_side, index=frame.index, dtype=int)
    stage23_cfg = dict((cfg.get("evaluation", {}) or {}).get("stage23", {}))
    stage23_enabled = bool(stage23_cfg.get("enabled", False))
    stage24_cfg = dict((cfg.get("evaluation", {}) or {}).get("stage24", {}))
    stage24_enabled = bool(stage24_cfg.get("enabled", False))
    execution_reject_events: list[dict[str, Any]] = []
    execution_adjustment_events: list[dict[str, Any]] = []
    shadow_live_summary: dict[str, Any] = {}
    research_infeasible_flags: list[dict[str, Any]] = []
    if stage_profile.trading and stage23_enabled:
        eligibility = evaluate_eligibility(
            frame=frame,
            raw_side=pd.Series(final_side, index=frame.index, dtype=int),
            family=str(family),
            policy_snapshot=stage23_cfg,
            symbol=str(symbol),
        )
        eligible_mask = pd.to_numeric(eligibility["eligible"], errors="coerce").fillna(False).astype(bool)
        final_side = np.where(eligible_mask.to_numpy(dtype=bool), final_side, 0).astype(int)
        after_context_count = int(np.count_nonzero(final_side))
        after_confirm_count = after_context_count
        after_riskgate_count = after_context_count
        context_reject_count = max(0, raw_signal_count - after_context_count)
        reason_vals = (
            pd.to_numeric(eligibility["eligible"], errors="coerce").fillna(False).astype(bool)
        )
        rejected_reasons = pd.Series(eligibility["reasons"], index=frame.index).loc[~reason_vals].astype(str)
        rejected_reasons = rejected_reasons[rejected_reasons != ""]
        context_reasons = (
            "|".join(sorted(rejected_reasons.value_counts().head(3).index.tolist()))
            if not rejected_reasons.empty
            else ""
        )
        eligibility_trace_rows = list(eligibility.get("trace_rows", []))

    if stage_profile.trading and (stage23_enabled or stage24_enabled):
        adaptive = build_adaptive_orders(
            frame=frame,
            raw_side=pd.Series(final_side, index=frame.index, dtype=int),
            score=pd.Series(weighted_score, index=frame.index, dtype=float),
            cfg=cfg,
            symbol=str(symbol),
        )
        accepted_signal = pd.to_numeric(adaptive["accepted_signal"], errors="coerce").fillna(0).astype(int)
        signal_series = accepted_signal.shift(1).fillna(0).astype(int)
        orders_attempted_count = int(adaptive["breakdown"]["total_orders_attempted"])
        orders_sent_count = int((signal_series != 0).sum())
        execution_reject_events = list(adaptive.get("reject_events", []))
        execution_adjustment_events = list(adaptive.get("adjustment_events", []))
        sizing_trace_rows = list(adaptive.get("sizing_trace", pd.DataFrame()).to_dict(orient="records"))
        stage24_sizing_trace_rows = list(adaptive.get("stage24_sizing_trace", pd.DataFrame()).to_dict(orient="records"))
        shadow_live_summary = dict(adaptive.get("shadow_live_summary", {}))
        research_infeasible_flags = list(adaptive.get("research_infeasible_flags", []))
    else:
        signal_series = signal_pre.shift(1).fillna(0).astype(int)
        orders_attempted_count = int((signal_pre != 0).sum()) if stage_profile.trading else 0
        orders_sent_count = int((signal_series != 0).sum()) if stage_profile.trading else 0
        sizing_trace_rows = []
        stage24_sizing_trace_rows = []
        shadow_live_summary = {}
        research_infeasible_flags = []

    return {
        "raw_signal_count": raw_signal_count,
        "raw_signal_rate_per_1000_bars": raw_signal_rate,
        "after_context_count": after_context_count,
        "context_reject_count": context_reject_count,
        "context_reject_reasons": context_reasons,
        "after_confirm_count": after_confirm_count,
        "after_riskgate_count": after_riskgate_count,
        "cooldown_reject_count": cooldown_reject_count,
        "conflict_reject_count": conflict_reject_count,
        "orders_attempted_count": orders_attempted_count,
        "orders_sent_count": orders_sent_count,
        "execution_reject_events": execution_reject_events,
        "execution_adjustment_events": execution_adjustment_events,
        "eligibility_trace_rows": eligibility_trace_rows,
        "sizing_trace_rows": sizing_trace_rows,
        "stage24_sizing_trace_rows": stage24_sizing_trace_rows,
        "shadow_live_summary": shadow_live_summary,
        "research_infeasible_flags": research_infeasible_flags,
        "signal_series": signal_series,
        "signal_pre": signal_pre,
    }


def _aggregate_sizing_trace_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "attempted": 0,
            "accepted": 0,
            "rejected": 0,
            "zero_size_count": 0,
            "raw_size_min": 0.0,
            "raw_size_median": 0.0,
            "raw_size_p95": 0.0,
            "margin_required_min": 0.0,
            "margin_required_median": 0.0,
            "margin_required_max": 0.0,
            "margin_limit_min": 0.0,
            "margin_limit_median": 0.0,
            "margin_limit_max": 0.0,
            "rescued_by_ceil_count": 0,
            "bumped_to_min_notional_count": 0,
            "cap_binding_reject_count": 0,
            "reject_reason_counts": {},
        }
    raw_size = pd.to_numeric(df.get("raw_size", 0.0), errors="coerce").fillna(0.0)
    rounded = pd.to_numeric(df.get("rounded_size_after", 0.0), errors="coerce").fillna(0.0)
    margin_required = pd.to_numeric(df.get("margin_required", 0.0), errors="coerce").fillna(0.0)
    margin_limit = pd.to_numeric(df.get("margin_limit", 0.0), errors="coerce").fillna(0.0)
    decision = df.get("decision", pd.Series(dtype=str)).astype(str)
    reject_reason = df.get("reject_reason", pd.Series(dtype=str)).astype(str).replace("", "ACCEPTED")
    attempted = int(len(df))
    accepted = int((decision == "ACCEPTED").sum())
    rejected = int(attempted - accepted)
    return {
        "attempted": attempted,
        "accepted": accepted,
        "rejected": rejected,
        "zero_size_count": int(((raw_size > 0.0) & (rounded <= 0.0)).sum()),
        "raw_size_min": float(raw_size.min()),
        "raw_size_median": float(raw_size.median()),
        "raw_size_p95": float(raw_size.quantile(0.95)),
        "margin_required_min": float(margin_required.min()),
        "margin_required_median": float(margin_required.median()),
        "margin_required_max": float(margin_required.max()),
        "margin_limit_min": float(margin_limit.min()),
        "margin_limit_median": float(margin_limit.median()),
        "margin_limit_max": float(margin_limit.max()),
        "rescued_by_ceil_count": int(pd.to_numeric(df.get("ceil_rescue_applied", False), errors="coerce").fillna(False).astype(bool).sum()),
        "bumped_to_min_notional_count": int(pd.to_numeric(df.get("bumped_to_min_notional", False), errors="coerce").fillna(False).astype(bool).sum()),
        "cap_binding_reject_count": int(((decision == "REJECTED") & (df.get("cap_binding", pd.Series(dtype=str)).astype(str) != "")).sum()),
        "reject_reason_counts": {
            str(k): int(v) for k, v in reject_reason.value_counts().sort_index().items()
        },
    }


def _aggregate_stage24_trace_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "valid_count": 0,
            "invalid_count": 0,
            "top_invalid_reasons": {},
            "notional_min": 0.0,
            "notional_median": 0.0,
            "notional_max": 0.0,
            "risk_used_min": 0.0,
            "risk_used_median": 0.0,
            "risk_used_max": 0.0,
        }
    status = df.get("status", pd.Series(dtype=str)).astype(str)
    reason = df.get("reason", pd.Series(dtype=str)).astype(str).replace("", "UNKNOWN")
    notional = pd.to_numeric(df.get("notional", 0.0), errors="coerce").fillna(0.0)
    risk_used = pd.to_numeric(df.get("risk_used", 0.0), errors="coerce").fillna(0.0)
    invalid_counts = (
        reason.loc[status != "VALID"].value_counts().head(5).to_dict()
    )
    return {
        "valid_count": int((status == "VALID").sum()),
        "invalid_count": int((status != "VALID").sum()),
        "top_invalid_reasons": {str(k): int(v) for k, v in invalid_counts.items()},
        "notional_min": float(notional.min()),
        "notional_median": float(notional.median()),
        "notional_max": float(notional.max()),
        "risk_used_min": float(risk_used.min()),
        "risk_used_median": float(risk_used.median()),
        "risk_used_max": float(risk_used.max()),
    }


def _aggregate_shadow_live_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "enabled": False,
            "rows": 0,
            "research_accepted_count": 0,
            "live_pass_count": 0,
            "research_accepted_but_live_rejected_count": 0,
            "live_reject_rate": 0.0,
            "live_reject_reason_counts": {},
        }
    enabled = pd.to_numeric(df.get("enabled", False), errors="coerce").fillna(False).astype(bool)
    research_accepted = pd.to_numeric(df.get("research_accepted_count", 0), errors="coerce").fillna(0).astype(int)
    live_pass = pd.to_numeric(df.get("live_pass_count", 0), errors="coerce").fillna(0).astype(int)
    live_rejected = pd.to_numeric(df.get("research_accepted_but_live_rejected_count", 0), errors="coerce").fillna(0).astype(int)
    reason_counts: dict[str, int] = {}
    for item in df.get("live_reject_reason_counts", pd.Series(dtype=object)).tolist():
        if not isinstance(item, dict):
            continue
        for reason, count in item.items():
            key = str(reason)
            reason_counts[key] = int(reason_counts.get(key, 0)) + int(count)
    total_research = int(research_accepted.sum())
    total_live_rejected = int(live_rejected.sum())
    return {
        "enabled": bool(enabled.any()),
        "rows": int(len(df)),
        "research_accepted_count": total_research,
        "live_pass_count": int(live_pass.sum()),
        "research_accepted_but_live_rejected_count": total_live_rejected,
        "live_reject_rate": float(total_live_rejected / max(total_research, 1)),
        "live_reject_reason_counts": {str(k): int(v) for k, v in sorted(reason_counts.items())},
    }


def _run_backtest_metrics(
    *,
    frame: pd.DataFrame,
    symbol: str,
    signal: pd.Series,
    cfg: dict[str, Any],
    strategy_name: str,
    exit_mode: str,
) -> dict[str, Any]:
    work = frame.copy()
    work["signal"] = pd.to_numeric(signal, errors="coerce").fillna(0).astype(int)
    eval_cfg = (((cfg.get("evaluation", {}) or {}).get("stage10", {})) or {}).get("evaluation", {})
    stage24_cfg = dict((cfg.get("evaluation", {}) or {}).get("stage24", {}))
    initial_capital = 10_000.0
    if bool(stage24_cfg.get("enabled", False)):
        initial_equities = stage24_cfg.get("simulation", {}).get("initial_equities", [initial_capital])
        if isinstance(initial_equities, (list, tuple)) and initial_equities:
            try:
                initial_capital = float(initial_equities[0])
            except Exception:
                initial_capital = 10_000.0
    result = run_backtest(
        frame=work,
        strategy_name=strategy_name,
        symbol=symbol,
        signal_col="signal",
        initial_capital=float(initial_capital),
        max_hold_bars=int(eval_cfg.get("max_hold_bars", 24)),
        stop_atr_multiple=float(eval_cfg.get("stop_atr_multiple", 1.5)),
        take_profit_atr_multiple=float(eval_cfg.get("take_profit_atr_multiple", 3.0)),
        exit_mode=str(exit_mode),
        trailing_atr_k=float(eval_cfg.get("trailing_atr_k", 1.5)),
        round_trip_cost_pct=float(cfg.get("costs", {}).get("round_trip_cost_pct", 0.1)),
        slippage_pct=float(cfg.get("costs", {}).get("slippage_pct", 0.0005)),
        cost_model_cfg=cfg.get("cost_model", {}),
    )
    trades = result.trades.copy()
    bars_held = pd.to_numeric(trades.get("bars_held", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    pnl = pd.to_numeric(trades.get("pnl", pd.Series(dtype=float)), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    trade_count = float(result.metrics.get("trade_count", 0.0))
    tpm = float(trade_count / max(_months(work), 1e-9))
    exposure_ratio = float(bars_held.sum() / max(1.0, float(len(work)))) if not trades.empty else 0.0
    pf_raw = float(result.metrics.get("profit_factor", 0.0))
    pf_clip = float(np.clip(pf_raw, 0.0, 10.0))
    expectancy = float(result.metrics.get("expectancy", 0.0))
    exp_lcb = _exp_lcb(pnl.to_numpy(dtype=float))
    maxdd = float(result.metrics.get("max_drawdown", 0.0))
    return {
        "trades_executed_count": float(trade_count),
        "immediate_exit_count": int((bars_held <= 1).sum()) if not trades.empty else 0,
        "avg_hold_bars": float(bars_held.mean()) if not trades.empty else 0.0,
        "median_hold_bars": float(bars_held.median()) if not trades.empty else 0.0,
        "tpm": tpm,
        "exposure_ratio": exposure_ratio,
        "PF_raw": pf_raw,
        "PF_clipped": pf_clip,
        "expectancy": expectancy,
        "exp_lcb": exp_lcb,
        "maxDD": maxdd,
        "trade_pnls": pnl.to_numpy(dtype=float),
    }


def _walkforward_trace(
    *,
    frame: pd.DataFrame,
    signal: pd.Series,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    wf_cfg = (((cfg.get("evaluation", {}) or {}).get("stage8", {})) or {}).get("walkforward_v2", {})
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    if ts.dropna().empty:
        return _empty_wf_metrics()
    windows = build_windows(
        start_ts=ts.iloc[0],
        end_ts=ts.iloc[-1],
        train_days=int(wf_cfg.get("train_days", 180)),
        holdout_days=int(wf_cfg.get("holdout_days", 30)),
        forward_days=int(wf_cfg.get("forward_days", 30)),
        step_days=int(wf_cfg.get("step_days", 30)),
        reserve_tail_days=int(wf_cfg.get("reserve_tail_days", 0)),
    )
    min_trades = int(max(1, wf_cfg.get("min_trades", 10)))
    min_exposure = float(max(0.0, wf_cfg.get("min_exposure", 0.01)))
    expected_windows = int(len(windows))
    evaluated = 0
    usable = 0
    reject = {"min_trades_window": 0, "min_exposure": 0, "non_finite_metrics": 0, "insufficient_bars": 0}
    for w in windows:
        mask = (ts >= w.forward_start) & (ts < w.forward_end)
        fwd = frame.loc[mask].copy().reset_index(drop=True)
        sig = pd.to_numeric(signal.loc[mask], errors="coerce").fillna(0).astype(int).reset_index(drop=True)
        if fwd.empty:
            continue
        evaluated += 1
        if len(fwd) < 10:
            reject["insufficient_bars"] += 1
            continue
        trades_proxy = int((sig != 0).sum())
        exposure_proxy = float((sig != 0).mean()) if len(sig) else 0.0
        if trades_proxy < min_trades:
            reject["min_trades_window"] += 1
            continue
        if exposure_proxy < min_exposure:
            reject["min_exposure"] += 1
            continue
        if not np.isfinite(pd.to_numeric(fwd["close"], errors="coerce").to_numpy(dtype=float)).all():
            reject["non_finite_metrics"] += 1
            continue
        usable += 1
    return {
        "walkforward_attempted": bool(expected_windows > 0),
        "walkforward_executed_true": bool(evaluated > 0),
        "expected_windows": expected_windows,
        "evaluated_windows": int(evaluated),
        "usable_windows": int(usable),
        "wf_reject_min_trades_window": int(reject["min_trades_window"]),
        "wf_reject_min_exposure": int(reject["min_exposure"]),
        "wf_reject_non_finite_metrics": int(reject["non_finite_metrics"]),
        "wf_reject_insufficient_bars": int(reject["insufficient_bars"]),
    }


def _mc_trace(*, trade_count: float, trade_pnls: np.ndarray, cfg: dict[str, Any]) -> dict[str, Any]:
    arr = np.asarray(trade_pnls, dtype=float)
    mc_cfg = (((cfg.get("evaluation", {}) or {}).get("stage12", {})) or {}).get("monte_carlo", {})
    min_trades = int(max(1, mc_cfg.get("min_trades", 10)))
    attempted = bool(trade_count > 0)
    if not attempted:
        return {"MC_attempted": False, "MC_triggered": False, "mc_reject_min_trades": 1, "mc_reject_insufficient_return_series": 0, "mc_reject_non_finite": 0}
    if int(trade_count) < min_trades:
        return {"MC_attempted": True, "MC_triggered": False, "mc_reject_min_trades": 1, "mc_reject_insufficient_return_series": 0, "mc_reject_non_finite": 0}
    if arr.size < 2:
        return {"MC_attempted": True, "MC_triggered": False, "mc_reject_min_trades": 0, "mc_reject_insufficient_return_series": 1, "mc_reject_non_finite": 0}
    if not np.isfinite(arr).all():
        return {"MC_attempted": True, "MC_triggered": False, "mc_reject_min_trades": 0, "mc_reject_insufficient_return_series": 0, "mc_reject_non_finite": 1}
    return {"MC_attempted": True, "MC_triggered": True, "mc_reject_min_trades": 0, "mc_reject_insufficient_return_series": 0, "mc_reject_non_finite": 0}


def _top_reject_reason(row: dict[str, Any]) -> str:
    if int(row.get("rows_after_dropna", 0)) <= 0 or str(row.get("missing_feature_columns", "")):
        return "MISSING_FEATURES"
    if float(row.get("raw_signal_count", 0.0)) <= 0.0:
        return "RAW_SIGNAL_ZERO"
    if float(row.get("after_context_count", 0.0)) <= 0.0 and float(row.get("raw_signal_count", 0.0)) > 0.0:
        return "CONTEXT_REJECT"
    if float(row.get("after_riskgate_count", 0.0)) <= 0.0 and float(row.get("after_context_count", 0.0)) > 0.0:
        return "RISKGATE_REJECT"
    if float(row.get("orders_sent_count", 0.0)) > 0.0 and float(row.get("trades_executed_count", 0.0)) <= 0.0:
        return "ORDERS_NO_TRADES"
    if bool(row.get("walkforward_attempted", False)) and int(row.get("usable_windows", 0)) <= 0:
        return "LOW_USABLE_WINDOWS"
    if bool(row.get("MC_attempted", False)) and not bool(row.get("MC_triggered", False)):
        return "MC_NOT_TRIGGERED"
    return "VALID"


def _trace_data_hash(features_cache: dict[tuple[str, str], pd.DataFrame]) -> str:
    payload = {
        f"{symbol}|{tf}": stable_hash(
            frame.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]].to_dict(orient="list"),
            length=16,
        )
        for (symbol, tf), frame in sorted(features_cache.items())
    }
    return stable_hash(payload, length=16)


def _resolved_end_ts(features_cache: dict[tuple[str, str], pd.DataFrame]) -> str | None:
    vals: list[pd.Timestamp] = []
    for frame in features_cache.values():
        ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
        if not ts.empty:
            vals.append(ts.max())
    if not vals:
        return None
    return max(vals).isoformat()


def _git_head() -> str:
    head = Path(".git/HEAD")
    if not head.exists():
        return ""
    text = head.read_text(encoding="utf-8").strip()
    if text.startswith("ref: "):
        ref_path = Path(".git") / text.split(" ", 1)[1].strip()
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8").strip()
    return text


def _death_rate(before: Iterable[Any], after: Iterable[Any]) -> pd.Series:
    b = pd.to_numeric(pd.Series(list(before)), errors="coerce").fillna(0.0)
    a = pd.to_numeric(pd.Series(list(after)), errors="coerce").fillna(0.0)
    values = 1.0 - (a / np.maximum(1.0, b))
    return values.clip(lower=0.0, upper=1.0)


def _months(frame: pd.DataFrame) -> float:
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return 0.0
    return max(float((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400.0) / 30.0, 1e-9)


def _exp_lcb(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    mean = float(np.mean(arr))
    if arr.size <= 1:
        return mean
    std = float(np.std(arr, ddof=0))
    return float(mean - std / math.sqrt(float(arr.size)))


def _empty_bt_metrics() -> dict[str, Any]:
    return {
        "trades_executed_count": 0.0,
        "immediate_exit_count": 0,
        "avg_hold_bars": 0.0,
        "median_hold_bars": 0.0,
        "tpm": 0.0,
        "exposure_ratio": 0.0,
        "PF_raw": 0.0,
        "PF_clipped": 0.0,
        "expectancy": 0.0,
        "exp_lcb": 0.0,
        "maxDD": 0.0,
        "trade_pnls": np.asarray([], dtype=float),
    }


def _empty_wf_metrics() -> dict[str, Any]:
    return {
        "walkforward_attempted": False,
        "walkforward_executed_true": False,
        "expected_windows": 0,
        "evaluated_windows": 0,
        "usable_windows": 0,
        "wf_reject_min_trades_window": 0,
        "wf_reject_min_exposure": 0,
        "wf_reject_non_finite_metrics": 0,
        "wf_reject_insufficient_bars": 0,
    }


def _empty_mc_metrics() -> dict[str, Any]:
    return {
        "MC_attempted": False,
        "MC_triggered": False,
        "mc_reject_min_trades": 0,
        "mc_reject_insufficient_return_series": 0,
        "mc_reject_non_finite": 0,
    }


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_safe_json(v) for v in value]
    if isinstance(value, tuple):
        return [_safe_json(v) for v in value]
    if isinstance(value, pd.Series):
        return _safe_json(value.tolist())
    if isinstance(value, pd.DataFrame):
        return _safe_json(value.to_dict(orient="records"))
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if np.isfinite(x) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return _safe_json(value.tolist())
    if isinstance(value, pd.Timestamp):
        ts = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
        return ts.isoformat()
    return value
