"""Stage-12 full price-family sweep with walk-forward and Monte Carlo robustness checks."""

from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from buffmini.backtest.engine import run_backtest
from buffmini.baselines.stage0 import generate_signals, stage06_strategies
from buffmini.config import compute_config_hash, validate_config
from buffmini.constants import DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.data.cache import ohlcv_data_hash
from buffmini.portfolio.monte_carlo import simulate_equity_paths, summarize_mc
from buffmini.stage10.evaluate import _build_features
from buffmini.stage10.exits import normalize_exit_mode
from buffmini.stage10.signals import SIGNAL_FAMILIES, generate_signal_family
from buffmini.stage12.forensics import (
    aggregate_execution_diagnostics,
    classify_invalid_reason,
    classify_stage12_1,
    extract_trade_context_rows,
    metric_logic_validation,
    summarize_signal_forensics,
    write_stage12_1_docs,
    write_stage12_2_docs,
)
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact
from buffmini.validation.walkforward_v2 import WindowTriplet, aggregate_windows, build_windows


_COST_LEVEL_ORDER = {"low": 0, "realistic": 1, "high": 2}
_SUMMARY_STAGE = "12"
_STAGE12_4_SCORE_CACHE: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class StrategyVariant:
    strategy_key: str
    strategy_name: str
    source: str
    signal_builder: Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class ExitVariant:
    exit_type: str
    engine_exit_mode: str
    stop_atr_multiple: float
    take_profit_atr_multiple: float
    trailing_atr_k: float
    max_hold_bars: int


def run_stage12_sweep(
    *,
    config: dict[str, Any],
    seed: int = 42,
    dry_run: bool = False,
    symbols: list[str] | None = None,
    timeframes: list[str] | None = None,
    runs_root: Path = RUNS_DIR,
    docs_dir: Path = Path("docs"),
    data_dir: Path = RAW_DATA_DIR,
    derived_dir: Path = DERIVED_DATA_DIR,
) -> dict[str, Any]:
    """Run full Stage-12 matrix sweep and write run/docs artifacts."""

    cfg = deepcopy(config)
    validate_config(cfg)
    stage12_cfg = cfg.get("evaluation", {}).get("stage12", {})
    if not isinstance(stage12_cfg, dict):
        raise ValueError("evaluation.stage12 must be configured")

    resolved_symbols = list(symbols or stage12_cfg.get("symbols", ["BTC/USDT", "ETH/USDT"]))
    resolved_timeframes = list(timeframes or stage12_cfg.get("timeframes", ["15m", "30m", "1h", "2h", "4h", "1d"]))

    strategies = _resolve_strategies(cfg=cfg, stage12_cfg=stage12_cfg)
    exits = _resolve_exits(cfg=cfg, stage12_cfg=stage12_cfg)
    cost_scenarios = _resolve_cost_scenarios(cfg=cfg, stage12_cfg=stage12_cfg)
    if not strategies:
        raise ValueError("No Stage-12 strategies resolved")
    if not exits:
        raise ValueError("No Stage-12 exit variants resolved")
    if not cost_scenarios:
        raise ValueError("No Stage-12 cost scenarios resolved")

    run_started = time.perf_counter()
    runtime_by_tf: dict[str, float] = {}
    feature_hashes: dict[str, dict[str, str]] = {}
    rows: list[dict[str, Any]] = []
    trade_pnls_by_combo: dict[str, np.ndarray] = {}
    window_details_rows: list[dict[str, Any]] = []
    forensic_rows: list[dict[str, Any]] = []
    trade_context_rows: list[dict[str, Any]] = []
    reject_pipeline_rows: list[dict[str, Any]] = []
    min_usable_windows_valid = int(stage12_cfg.get("min_usable_windows_valid", 3))
    stage12_3_cfg = dict(cfg.get("evaluation", {}).get("stage12_3", {}))
    stage12_4_cfg = dict(cfg.get("evaluation", {}).get("stage12_4", {}))
    forensic_cfg = dict(stage12_cfg.get("forensics", {}))
    context_cfg = dict(forensic_cfg.get("context_model", {}))
    stage12_4_rows: list[dict[str, Any]] = []
    trace: dict[str, Any] = {
        "stage12_3": {
            "enabled": bool(stage12_3_cfg.get("enabled", False)),
            "applied_soft_weight_count": 0,
            "combos_seen": 0,
            "adaptive_usability_samples": [],
            "fallback_windows_used_count": 0,
        },
        "stage12_4": {
            "enabled": bool(stage12_4_cfg.get("enabled", False)),
            "score_computed_count": 0,
            "threshold_eval_count": 0,
            "cache_hit_count": 0,
            "cache_miss_count": 0,
        },
    }

    for timeframe in resolved_timeframes:
        tf_started = time.perf_counter()
        cfg_tf = _config_for_timeframe(
            cfg=cfg,
            timeframe=str(timeframe),
            base_timeframe=str(stage12_cfg.get("base_timeframe", "1m")),
        )
        features_by_symbol = _build_features(
            config=cfg_tf,
            symbols=resolved_symbols,
            timeframe=str(timeframe),
            dry_run=bool(dry_run),
            seed=int(seed),
            data_dir=data_dir,
            derived_dir=derived_dir,
        )
        feature_hashes[str(timeframe)] = {
            str(symbol): ohlcv_data_hash(frame.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]])
            for symbol, frame in features_by_symbol.items()
            if not frame.empty
        }
        for symbol in resolved_symbols:
            frame = features_by_symbol.get(symbol)
            if frame is None or frame.empty:
                continue
            frame_sorted = frame.copy().sort_values("timestamp").reset_index(drop=True)
            frame_sorted.attrs["timeframe"] = str(timeframe)
            windows = _build_windows_for_frame(
                frame=frame_sorted,
                cfg=cfg_tf,
                stage12_3_cfg=stage12_3_cfg,
                trace=trace,
            )
            for strategy in strategies:
                for exit_variant in exits:
                    for cost_level, cost_model_cfg in cost_scenarios.items():
                        frame_sorted.attrs["cost_level"] = cost_level
                        combo_key = _combo_key(
                            symbol=symbol,
                            timeframe=str(timeframe),
                            strategy=strategy.strategy_key,
                            exit_type=exit_variant.exit_type,
                            cost_level=cost_level,
                        )
                        combo_eval = _evaluate_combo(
                            frame=frame_sorted,
                            symbol=symbol,
                            strategy=strategy,
                            exit_variant=exit_variant,
                            cost_model_cfg=cost_model_cfg,
                            cfg=cfg_tf,
                            windows=windows,
                            min_usable_windows_valid=min_usable_windows_valid,
                            context_cfg=context_cfg,
                            stage12_3_cfg=stage12_3_cfg,
                            stage12_4_cfg=stage12_4_cfg,
                            seed=int(seed),
                            trace=trace,
                        )
                        combo_eval["combo_key"] = combo_key
                        forensic_rows.append(
                            {
                                "combo_key": combo_key,
                                "symbol": symbol,
                                "timeframe": str(timeframe),
                                "strategy": strategy.strategy_name,
                                "exit_variant": exit_variant.exit_type,
                                "cost_level": cost_level,
                                "trade_count": float(combo_eval["trade_count"]),
                                "raw_backtest_seconds": float(combo_eval["_raw_backtest_seconds"]),
                                "walkforward_windows_count": int(combo_eval["_walkforward_windows_count"]),
                                "walkforward_expected_windows": int(combo_eval["_walkforward_expected_windows"]),
                                "walkforward_seconds": float(combo_eval["_walkforward_seconds"]),
                                "walkforward_executed": bool(combo_eval["_walkforward_executed"]),
                                "walkforward_integrity_ok": bool(combo_eval["_walkforward_integrity_ok"]),
                                "MC_executed": False,
                                "invalid_reason": combo_eval["_invalid_reason"],
                                "metric_logic_all_ok": bool(combo_eval["_metric_logic"]["all_ok"]),
                                "metric_logic_exp_lcb_ok": bool(combo_eval["_metric_logic"]["exp_lcb_ok"]),
                                "metric_logic_zero_trade_pf_ok": bool(combo_eval["_metric_logic"]["zero_trade_pf_ok"]),
                                "metric_logic_no_losing_pf_ok": bool(combo_eval["_metric_logic"]["no_losing_pf_ok"]),
                                "metric_logic_stability_threshold_ok": bool(combo_eval["_metric_logic"]["stability_threshold_ok"]),
                                "avg_weight": float(combo_eval["_avg_weight"]),
                                "min_trades_required_per_window": int(combo_eval["_min_trades_required_per_window"]),
                            }
                        )
                        reject_pipeline_rows.append(
                            {
                                "combo_id": combo_key,
                                "symbol": symbol,
                                "timeframe": str(timeframe),
                                "strategy": strategy.strategy_name,
                                "exit_variant": exit_variant.exit_type,
                                "cost_level": cost_level,
                                "reason": combo_eval["_invalid_reason"] if combo_eval["_invalid_reason"] is not None else "VALID",
                                "stage": _reject_stage_label(
                                    invalid_reason=combo_eval["_invalid_reason"],
                                    raw_trade_count=float(combo_eval["_raw_trade_count"]),
                                    posthook_trade_count=float(combo_eval["trade_count"]),
                                ),
                                "raw_trade_count": float(combo_eval["_raw_trade_count"]),
                                "posthook_trade_count": float(combo_eval["trade_count"]),
                                "wf_required_trades": int(combo_eval["_min_trades_required_per_window"]),
                                "wf_actual_trades": float(combo_eval["_wf_avg_forward_trade_count"]),
                            }
                        )
                        trade_pnls_by_combo[combo_key] = combo_eval.pop("_trade_pnls")
                        for window_row in combo_eval.pop("_window_rows"):
                            window_details_rows.append(
                                {
                                    "combo_key": combo_key,
                                    "symbol": symbol,
                                    "timeframe": str(timeframe),
                                    "strategy": strategy.strategy_name,
                                    "exit_type": exit_variant.exit_type,
                                    "cost_level": cost_level,
                                    **window_row,
                                }
                            )
                        context_rows = combo_eval.pop("_trade_context_rows")
                        for item in context_rows:
                            item["combo_key"] = combo_key
                        trade_context_rows.extend(context_rows)
                        stage12_4_meta = dict(combo_eval.pop("_stage12_4_meta", {}))
                        if stage12_4_meta:
                            stage12_4_rows.append(
                                {
                                    "combo_key": combo_key,
                                    "symbol": symbol,
                                    "timeframe": str(timeframe),
                                    "strategy": strategy.strategy_name,
                                    "strategy_key": strategy.strategy_key,
                                    "exit_type": exit_variant.exit_type,
                                    "cost_level": cost_level,
                                    **stage12_4_meta,
                                }
                            )
                        for internal_key in (
                            "_raw_backtest_seconds",
                            "_walkforward_windows_count",
                            "_walkforward_expected_windows",
                            "_walkforward_seconds",
                            "_walkforward_executed",
                            "_walkforward_integrity_ok",
                            "_invalid_reason",
                            "_metric_logic",
                            "_avg_weight",
                            "_min_trades_required_per_window",
                            "_trade_timestamps",
                        ):
                            combo_eval.pop(internal_key, None)
                        rows.append(combo_eval)
        runtime_by_tf[str(timeframe)] = float(time.perf_counter() - tf_started)

    leaderboard = pd.DataFrame(rows)
    if leaderboard.empty:
        raise RuntimeError("Stage-12 produced no combinations")
    leaderboard = _apply_cost_sensitivity_and_robust_score(leaderboard=leaderboard, stage12_cfg=stage12_cfg)
    leaderboard, mc_keys = _apply_monte_carlo(
        leaderboard=leaderboard,
        trade_pnls_by_combo=trade_pnls_by_combo,
        stage12_cfg=stage12_cfg,
        seed=int(seed),
    )
    forensic_matrix = pd.DataFrame(forensic_rows)
    if not forensic_matrix.empty and mc_keys:
        forensic_matrix.loc[forensic_matrix["combo_key"].isin(list(mc_keys)), "MC_executed"] = True

    leaderboard["stability_rank"] = leaderboard["stability_classification"].map(
        {"STABLE": 0, "UNSTABLE": 1, "ZERO_TRADE": 2, "INVALID": 3, "INSUFFICIENT_DATA": 4}
    ).fillna(4)
    leaderboard = leaderboard.sort_values(
        ["exp_lcb", "stability_rank", "cost_sensitivity_slope", "symbol", "timeframe", "strategy", "exit_type", "cost_level"],
        ascending=[False, True, True, True, True, True, True, True],
    ).reset_index(drop=True)
    leaderboard = leaderboard.drop(columns=["stability_rank"], errors="ignore")

    total_runtime = float(time.perf_counter() - run_started)
    valid_combos = int((leaderboard["is_valid"] == True).sum())  # noqa: E712
    total_combos = int(len(leaderboard))
    verdict = _final_verdict(leaderboard)
    top_robust = _top_robust_rows(leaderboard, limit=10)

    summary_payload = {
        "stage": _SUMMARY_STAGE,
        "symbols": resolved_symbols,
        "timeframes": resolved_timeframes,
        "total_combinations": total_combos,
        "valid_combinations": valid_combos,
        "top_robust": top_robust,
        "verdict": verdict,
        "runtime_seconds": total_runtime,
        "pytest_pass_count": None,
        "seed": int(seed),
        "config_hash": compute_config_hash(cfg),
        "data_hash": stable_hash(feature_hashes, length=16),
        "runtime_by_timeframe": runtime_by_tf,
        "strategy_count": len(strategies),
        "exit_count": len(exits),
        "cost_levels": list(cost_scenarios.keys()),
    }
    run_id = f"{utc_now_compact()}_{stable_hash(summary_payload, length=12)}_stage12"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "resolved_config.json").write_text(
        json.dumps(cfg, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    leaderboard.to_csv(run_dir / "leaderboard.csv", index=False)
    pd.DataFrame(window_details_rows).to_csv(run_dir / "window_metrics.csv", index=False)
    forensic_matrix.to_csv(run_dir / "stage12_forensic_matrix.csv", index=False)
    pd.DataFrame(reject_pipeline_rows).to_csv(run_dir / "stage12_reject_pipeline.csv", index=False)
    pd.DataFrame(
        [{"timeframe": tf, "runtime_seconds": seconds} for tf, seconds in runtime_by_tf.items()]
    ).to_csv(run_dir / "runtime_by_timeframe.csv", index=False)

    suspicious_ms_threshold = float(forensic_cfg.get("suspicious_backtest_ms_threshold", 5.0))
    forensic_summary = aggregate_execution_diagnostics(
        matrix=forensic_matrix,
        suspicious_backtest_ms_threshold=suspicious_ms_threshold,
    )
    stage12_1_classification = classify_stage12_1(
        diagnostics=forensic_summary,
        leaderboard=leaderboard,
    )
    forensic_summary["final_stage12_1_classification"] = stage12_1_classification
    (run_dir / "stage12_forensic_summary.json").write_text(
        json.dumps(forensic_summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (run_dir / "stage12_trace.json").write_text(
        json.dumps(trace, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    trade_rate_audit = _build_trade_rate_audit(
        leaderboard=leaderboard,
        forensic_matrix=forensic_matrix,
        min_usable_windows_valid=min_usable_windows_valid,
    )
    trade_rate_audit.to_csv(run_dir / "stage12_trade_rate_audit.csv", index=False)
    stage12_3_metrics = _build_stage12_3_metrics(
        forensic_summary=forensic_summary,
        target_thresholds={
            "zero_trade_pct_max": 25.0,
            "walkforward_executed_true_pct_min": 40.0,
            "mc_trigger_rate_min": 10.0,
            "invalid_pct_max": 60.0,
        },
    )
    (run_dir / "stage12_3_metrics.json").write_text(
        json.dumps(stage12_3_metrics, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    trade_context_df = pd.DataFrame(trade_context_rows)
    false_positive_map, signal_forensics_summary, by_strategy = summarize_signal_forensics(
        trade_context=trade_context_df,
        context_cfg=context_cfg,
    )
    false_positive_map.to_csv(run_dir / "stage12_false_positive_map.csv", index=False)
    by_strategy.to_csv(run_dir / "stage12_signal_forensics_by_strategy.csv", index=False)
    (run_dir / "stage12_signal_forensics_summary.json").write_text(
        json.dumps(signal_forensics_summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    summary_payload["run_id"] = run_id
    (run_dir / "stage12_summary.json").write_text(
        json.dumps(summary_payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    docs_dir.mkdir(parents=True, exist_ok=True)
    report_md = docs_dir / "stage12_report.md"
    report_json = docs_dir / "stage12_report_summary.json"
    _write_docs_report(
        leaderboard=leaderboard,
        summary=summary_payload,
        report_md=report_md,
        report_json=report_json,
    )
    write_stage12_1_docs(
        report_md=docs_dir / "stage12_1_execution_forensics_report.md",
        report_json=docs_dir / "stage12_1_execution_forensics_summary.json",
        run_id=run_id,
        diagnostics=forensic_summary,
        classification=stage12_1_classification,
    )
    write_stage12_2_docs(
        report_md=docs_dir / "stage12_2_signal_forensics_report.md",
        report_json=docs_dir / "stage12_2_signal_forensics_summary.json",
        run_id=run_id,
        summary=signal_forensics_summary,
        by_strategy=by_strategy,
    )
    _write_stage12_3_docs(
        report_md=docs_dir / "stage12_3_report.md",
        report_json=docs_dir / "stage12_3_summary.json",
        run_id=run_id,
        stage12_3_metrics=stage12_3_metrics,
        trade_rate_audit=trade_rate_audit,
        leaderboard=leaderboard,
        baseline_stage12_1=docs_dir / "stage12_1_execution_forensics_summary.json",
    )
    stage12_4_table = pd.DataFrame(stage12_4_rows)
    if not stage12_4_table.empty:
        stage12_4_table.to_csv(run_dir / "stage12_4_selection.csv", index=False)
    stage12_4_metrics = _build_stage12_4_metrics(
        stage12_4_table=stage12_4_table,
        forensic_summary=forensic_summary,
        leaderboard=leaderboard,
        enabled=bool(stage12_4_cfg.get("enabled", False)),
    )
    (run_dir / "stage12_4_metrics.json").write_text(
        json.dumps(stage12_4_metrics, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_stage12_4_docs(
        report_md=docs_dir / "stage12_4_report.md",
        report_json=docs_dir / "stage12_4_summary.json",
        run_id=run_id,
        stage12_4_metrics=stage12_4_metrics,
        stage12_3_metrics=stage12_3_metrics,
    )
    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "summary": summary_payload,
        "leaderboard": leaderboard,
        "forensic_summary": forensic_summary,
        "signal_forensics_summary": signal_forensics_summary,
        "stage12_3_metrics": stage12_3_metrics,
        "stage12_4_metrics": stage12_4_metrics,
        "stage12_trace": trace,
        "report_md": report_md,
        "report_json": report_json,
    }


def validate_stage12_summary_schema(payload: dict[str, Any]) -> None:
    required = {
        "stage",
        "symbols",
        "timeframes",
        "total_combinations",
        "valid_combinations",
        "top_robust",
        "verdict",
        "runtime_seconds",
        "pytest_pass_count",
    }
    missing = required.difference(payload.keys())
    if missing:
        raise ValueError(f"Missing Stage-12 summary keys: {sorted(missing)}")
    if str(payload["stage"]) != _SUMMARY_STAGE:
        raise ValueError("stage must be '12'")
    if str(payload["verdict"]) not in {"STRONG EDGE FOUND", "WEAK EDGE", "NO ROBUST EDGE"}:
        raise ValueError("verdict must be STRONG EDGE FOUND / WEAK EDGE / NO ROBUST EDGE")
    if int(payload["total_combinations"]) < 0:
        raise ValueError("total_combinations must be >= 0")
    if int(payload["valid_combinations"]) < 0:
        raise ValueError("valid_combinations must be >= 0")
    if float(payload["runtime_seconds"]) < 0:
        raise ValueError("runtime_seconds must be >= 0")
    if not isinstance(payload["top_robust"], list):
        raise ValueError("top_robust must be a list")


def _resolve_strategies(cfg: dict[str, Any], stage12_cfg: dict[str, Any]) -> list[StrategyVariant]:
    variants: list[StrategyVariant] = []
    if bool(stage12_cfg.get("include_stage06_baselines", True)):
        for spec in stage06_strategies():
            strategy_key = str(spec.name).replace(" ", "_")
            variants.append(
                StrategyVariant(
                    strategy_key=strategy_key,
                    strategy_name=str(spec.name),
                    source="stage06",
                    signal_builder=lambda frame, s=spec: generate_signals(frame, strategy=s, gating_mode="none"),
                )
            )
    if bool(stage12_cfg.get("include_stage10_families", True)):
        defaults = cfg.get("evaluation", {}).get("stage10", {}).get("signals", {}).get("defaults", {})
        families = cfg.get("evaluation", {}).get("stage10", {}).get("signals", {}).get("families", list(SIGNAL_FAMILIES))
        for family in families:
            family_name = str(family)
            if family_name not in SIGNAL_FAMILIES:
                continue
            params = dict(defaults.get(family_name, {}))
            variants.append(
                StrategyVariant(
                    strategy_key=family_name,
                    strategy_name=_family_display_name(family_name),
                    source="stage10",
                    signal_builder=lambda frame, fam=family_name, par=params: generate_signal_family(frame, family=fam, params=par)[
                        "signal"
                    ].astype(int),
                )
            )
    dedup: dict[str, StrategyVariant] = {}
    for item in variants:
        dedup[item.strategy_key] = item
    return [dedup[key] for key in sorted(dedup)]


def _resolve_exits(cfg: dict[str, Any], stage12_cfg: dict[str, Any]) -> list[ExitVariant]:
    eval_cfg = cfg.get("evaluation", {}).get("stage10", {}).get("evaluation", {})
    exits_cfg = cfg.get("evaluation", {}).get("stage10", {}).get("exits", {})
    base_stop = float(eval_cfg.get("stop_atr_multiple", 1.5))
    base_tp = float(eval_cfg.get("take_profit_atr_multiple", 3.0))
    base_max_hold = int(eval_cfg.get("max_hold_bars", 24))
    trailing_k = float(exits_cfg.get("trailing_atr_k", 1.5))
    variants = []
    for name in stage12_cfg.get("exits", {}).get("variants", []):
        key = str(name)
        if key == "fixed_atr":
            variants.append(
                ExitVariant(
                    exit_type="fixed_atr",
                    engine_exit_mode=normalize_exit_mode("fixed_atr"),
                    stop_atr_multiple=base_stop,
                    take_profit_atr_multiple=base_tp,
                    trailing_atr_k=trailing_k,
                    max_hold_bars=base_max_hold,
                )
            )
        elif key == "structure_trailing":
            variants.append(
                ExitVariant(
                    exit_type="structure_trailing",
                    engine_exit_mode=normalize_exit_mode("atr_trailing"),
                    stop_atr_multiple=base_stop,
                    take_profit_atr_multiple=base_tp,
                    trailing_atr_k=trailing_k,
                    max_hold_bars=base_max_hold,
                )
            )
        elif key == "time_based":
            variants.append(
                ExitVariant(
                    exit_type="time_based",
                    engine_exit_mode=normalize_exit_mode("fixed_atr"),
                    stop_atr_multiple=float(stage12_cfg["exits"]["time_based_stop_atr_multiple"]),
                    take_profit_atr_multiple=float(stage12_cfg["exits"]["time_based_take_profit_atr_multiple"]),
                    trailing_atr_k=trailing_k,
                    max_hold_bars=base_max_hold,
                )
            )
    return variants


def _resolve_cost_scenarios(cfg: dict[str, Any], stage12_cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    base_cfg = deepcopy(cfg.get("cost_model", {}))
    base_cfg["mode"] = "v2"
    base_cfg.setdefault("v2", {})
    scenarios = stage12_cfg.get("cost_scenarios", {})
    out: dict[str, dict[str, Any]] = {}
    for level in ("low", "realistic", "high"):
        if level == "realistic":
            out[level] = deepcopy(base_cfg)
            continue
        payload = deepcopy(base_cfg)
        overrides = dict(scenarios.get(level, {}))
        payload["v2"]["slippage_bps_base"] = float(overrides.get("slippage_bps_base", payload["v2"].get("slippage_bps_base", 0.5)))
        payload["v2"]["slippage_bps_vol_mult"] = float(
            overrides.get("slippage_bps_vol_mult", payload["v2"].get("slippage_bps_vol_mult", 2.0))
        )
        payload["v2"]["spread_bps"] = float(overrides.get("spread_bps", payload["v2"].get("spread_bps", 0.5)))
        payload["v2"]["delay_bars"] = int(overrides.get("delay_bars", payload["v2"].get("delay_bars", 0)))
        out[level] = payload
    return out


def _config_for_timeframe(cfg: dict[str, Any], timeframe: str, base_timeframe: str) -> dict[str, Any]:
    out = deepcopy(cfg)
    universe = out.setdefault("universe", {})
    base_tf = str(base_timeframe)
    universe["base_timeframe"] = base_tf
    universe["timeframe"] = str(timeframe)
    universe["operational_timeframe"] = str(timeframe)
    if base_tf == "1m" and str(timeframe) != "1m":
        out.setdefault("data", {})["resample_source"] = "base"
    else:
        out.setdefault("data", {}).setdefault("resample_source", "direct")
    return out


def _build_windows_for_frame(
    *,
    frame: pd.DataFrame,
    cfg: dict[str, Any],
    stage12_3_cfg: dict[str, Any],
    trace: dict[str, Any],
) -> list[WindowTriplet]:
    wf_cfg = cfg.get("evaluation", {}).get("stage8", {}).get("walkforward_v2", {})
    if frame.empty:
        return []
    windows = build_windows(
        start_ts=frame["timestamp"].iloc[0],
        end_ts=frame["timestamp"].iloc[-1],
        train_days=int(wf_cfg.get("train_days", 180)),
        holdout_days=int(wf_cfg.get("holdout_days", 30)),
        forward_days=int(wf_cfg.get("forward_days", 30)),
        step_days=int(wf_cfg.get("step_days", 30)),
        reserve_tail_days=int(wf_cfg.get("reserve_tail_days", 0)),
    )
    if windows:
        return windows
    # Stage-12.3 fallback: keep walkforward execution active on short histories.
    if not bool(stage12_3_cfg.get("enabled", False)):
        return windows
    fallback = _build_fallback_windows(frame)
    if fallback:
        trace["stage12_3"]["fallback_windows_used_count"] = int(trace["stage12_3"].get("fallback_windows_used_count", 0)) + 1
    return fallback


def _build_fallback_windows(frame: pd.DataFrame) -> list[WindowTriplet]:
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return []
    ts = ts.sort_values().reset_index(drop=True)
    n = int(len(ts))
    if n < 30:
        return []
    delta = ts.diff().dropna()
    bar_step = delta.median() if not delta.empty else pd.Timedelta(hours=1)
    train_bars = max(12, int(round(n * 0.50)))
    holdout_bars = max(6, int(round(n * 0.25)))
    forward_bars = max(6, int(round(n * 0.25)))
    if train_bars + holdout_bars + forward_bars > n:
        train_bars = max(10, int(round(n * 0.45)))
        rem = max(2, n - train_bars)
        holdout_bars = max(1, rem // 2)
        forward_bars = max(1, rem - holdout_bars)
    if train_bars < 1 or holdout_bars < 1 or forward_bars < 1:
        return []

    windows: list[WindowTriplet] = []
    idx = 0
    train_end_idx = train_bars
    step = max(1, forward_bars)
    while train_end_idx + holdout_bars + forward_bars <= n:
        train_start_idx = train_end_idx - train_bars
        holdout_start_idx = train_end_idx
        holdout_end_idx = holdout_start_idx + holdout_bars
        forward_start_idx = holdout_end_idx
        forward_end_idx = forward_start_idx + forward_bars
        forward_end_ts = ts.iloc[forward_end_idx] if forward_end_idx < n else ts.iloc[-1] + bar_step
        windows.append(
            WindowTriplet(
                window_idx=idx,
                train_start=ts.iloc[train_start_idx],
                train_end=ts.iloc[train_end_idx],
                holdout_start=ts.iloc[holdout_start_idx],
                holdout_end=ts.iloc[holdout_end_idx],
                forward_start=ts.iloc[forward_start_idx],
                forward_end=forward_end_ts,
            )
        )
        idx += 1
        train_end_idx += step
    return windows


def _evaluate_combo(
    *,
    frame: pd.DataFrame,
    symbol: str,
    strategy: StrategyVariant,
    exit_variant: ExitVariant,
    cost_model_cfg: dict[str, Any],
    cfg: dict[str, Any],
    windows: list[Any],
    min_usable_windows_valid: int,
    context_cfg: dict[str, Any],
    stage12_3_cfg: dict[str, Any],
    stage12_4_cfg: dict[str, Any],
    seed: int,
    trace: dict[str, Any],
) -> dict[str, Any]:
    raw_signal_series = strategy.signal_builder(frame)
    raw_trade_count = math.nan
    if bool(stage12_4_cfg.get("enabled", False)):
        raw_result = _run_backtest_with_signal(
            frame=frame,
            signal=raw_signal_series,
            symbol=symbol,
            strategy_name=strategy.strategy_name,
            exit_variant=exit_variant,
            cfg=cfg,
            cost_model_cfg=cost_model_cfg,
        )
        raw_trade_count = float(raw_result.metrics.get("trade_count", 0.0))
    signal_series, stage12_4_meta = _stage12_4_scored_signal(
        frame=frame,
        strategy=strategy,
        raw_signal=raw_signal_series,
        stage12_4_cfg=stage12_4_cfg,
        seed=int(seed),
        trace=trace,
    )
    signal_abs = pd.to_numeric(signal_series, errors="coerce").fillna(0).abs()
    weight_series = _soft_weight_series(
        frame=frame,
        strategy=strategy,
        stage12_3_cfg=stage12_3_cfg,
    )
    if float(signal_abs.sum()) > 0:
        avg_weight = float((weight_series * signal_abs).sum() / signal_abs.sum())
    else:
        avg_weight = float(weight_series.mean()) if len(weight_series) else 1.0
    trace["stage12_3"]["combos_seen"] = int(trace["stage12_3"].get("combos_seen", 0)) + 1
    if bool(stage12_3_cfg.get("enabled", False)):
        reduced = int(((weight_series < 1.0) & (signal_abs > 0)).sum())
        trace["stage12_3"]["applied_soft_weight_count"] = int(trace["stage12_3"].get("applied_soft_weight_count", 0)) + reduced

    bt_started = time.perf_counter()
    backtest_result = _run_backtest_with_signal(
        frame=frame,
        signal=signal_series,
        symbol=symbol,
        strategy_name=strategy.strategy_name,
        exit_variant=exit_variant,
        cfg=cfg,
        cost_model_cfg=cost_model_cfg,
    )
    raw_backtest_seconds = float(time.perf_counter() - bt_started)
    metrics = _metrics_from_backtest(backtest_result=backtest_result, frame=frame)
    wf_started = time.perf_counter()
    window_rows, wf_summary = _evaluate_walkforward_combo(
        frame=frame,
        symbol=symbol,
        strategy=strategy,
        exit_variant=exit_variant,
        cfg=cfg,
        cost_model_cfg=cost_model_cfg,
        windows=windows,
        trades_per_month=float(metrics["trades_per_month"]),
        stage12_3_cfg=stage12_3_cfg,
        stage12_4_cfg=stage12_4_cfg,
        seed=int(seed),
        trace=trace,
    )
    walkforward_seconds = float(time.perf_counter() - wf_started)
    expected_windows = int(len(windows))
    evaluated_windows = int(len(window_rows))
    walkforward_integrity_ok = bool(expected_windows <= 0 or expected_windows == evaluated_windows)
    if expected_windows > 0 and not walkforward_integrity_ok:
        raise RuntimeError(
            f"Walkforward integrity mismatch for {symbol}/{frame.attrs.get('timeframe','')}/{strategy.strategy_key}: "
            f"expected_windows={expected_windows}, evaluated_windows={evaluated_windows}"
        )
    stability_classification = str(wf_summary.get("classification", "INSUFFICIENT_DATA"))
    usable_windows = int(wf_summary.get("usable_windows", 0))
    metric_logic = metric_logic_validation(
        trade_count=float(metrics["trade_count"]),
        profit_factor=float(metrics["profit_factor"]),
        pnl_values=pd.to_numeric(backtest_result.trades.get("pnl", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .to_numpy(dtype=float),
        exp_lcb_reported=float(metrics["exp_lcb"]),
        stability_classification=stability_classification,
        usable_windows=usable_windows,
        min_usable_windows_valid=int(min_usable_windows_valid),
    )
    invalid_reason = classify_invalid_reason(
        trade_count=float(metrics["trade_count"]),
        stability_classification=stability_classification,
        usable_windows=usable_windows,
        min_usable_windows_valid=int(min_usable_windows_valid),
    )
    if usable_windows < int(min_usable_windows_valid):
        stability_effective = "ZERO_TRADE" if str(invalid_reason) == "ZERO_TRADE" else "INVALID"
        is_valid = False
    else:
        stability_effective = "ZERO_TRADE" if str(invalid_reason) == "ZERO_TRADE" else stability_classification
        is_valid = stability_classification != "INSUFFICIENT_DATA"
    output = {
        "symbol": str(symbol),
        "timeframe": str(frame.attrs.get("timeframe", "unknown")),
        "strategy": strategy.strategy_name,
        "strategy_key": strategy.strategy_key,
        "strategy_source": strategy.source,
        "exit_type": exit_variant.exit_type,
        "cost_level": str(frame.attrs.get("cost_level", "unknown")),
        "PF": float(metrics["profit_factor"]),
        "exp_lcb": float(metrics["exp_lcb"]),
        "expectancy": float(metrics["expectancy"]),
        "trades_per_month": float(metrics["trades_per_month"]),
        "maxDD": float(metrics["max_drawdown"]),
        "trade_count": float(metrics["trade_count"]),
        "stability_classification": stability_effective,
        "usable_windows": usable_windows,
        "is_valid": bool(is_valid),
        "walkforward_classification_raw": stability_classification,
        "cost_sensitivity_slope": 0.0,
        "robust_score": 0.0,
        "MC_p_ruin": math.nan,
        "MC_p_return_negative": math.nan,
        "MC_maxDD_p95": math.nan,
        "MC_expected_log_growth": math.nan,
        "avg_weight": float(avg_weight),
        "raw_trade_count": float(raw_trade_count) if np.isfinite(raw_trade_count) else float(metrics["trade_count"]),
        "_raw_backtest_seconds": raw_backtest_seconds,
        "_walkforward_windows_count": evaluated_windows,
        "_walkforward_expected_windows": expected_windows,
        "_walkforward_seconds": walkforward_seconds,
        "_walkforward_executed": bool(expected_windows > 0 and evaluated_windows > 0),
        "_walkforward_integrity_ok": walkforward_integrity_ok,
        "_invalid_reason": invalid_reason,
        "_metric_logic": metric_logic,
        "_raw_trade_count": float(raw_trade_count) if np.isfinite(raw_trade_count) else float(metrics["trade_count"]),
        "_wf_avg_forward_trade_count": float(wf_summary.get("avg_forward_trade_count", 0.0)),
        "_avg_weight": float(avg_weight),
        "_min_trades_required_per_window": int(wf_summary.get("min_trades_required_per_window", 0)),
        "_trade_context_rows": extract_trade_context_rows(
            combo_key="",
            symbol=str(symbol),
            timeframe=str(frame.attrs.get("timeframe", "unknown")),
            strategy=strategy.strategy_name,
            strategy_key=str(strategy.strategy_key),
            strategy_source=str(strategy.source),
            exit_type=str(exit_variant.exit_type),
            cost_level=str(frame.attrs.get("cost_level", "unknown")),
            frame=frame,
            trades=backtest_result.trades.copy(),
            context_cfg=context_cfg,
        ),
        "_trade_pnls": pd.to_numeric(backtest_result.trades.get("pnl", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .to_numpy(dtype=float),
        "_trade_timestamps": [
            (
                _to_iso_utc(row.get("entry_time")),
                _to_iso_utc(row.get("exit_time")),
            )
            for row in backtest_result.trades.to_dict(orient="records")
        ],
        "_window_rows": window_rows,
        "_stage12_4_meta": stage12_4_meta,
    }
    return output


def _run_backtest_with_signal(
    *,
    frame: pd.DataFrame,
    signal: pd.Series,
    symbol: str,
    strategy_name: str,
    exit_variant: ExitVariant,
    cfg: dict[str, Any],
    cost_model_cfg: dict[str, Any],
) -> Any:
    eval_cfg = cfg.get("evaluation", {}).get("stage10", {}).get("evaluation", {})
    costs = cfg.get("costs", {})
    work = frame.copy()
    work["signal"] = pd.to_numeric(signal, errors="coerce").fillna(0).astype(int)
    return run_backtest(
        frame=work,
        strategy_name=str(strategy_name),
        symbol=str(symbol),
        signal_col="signal",
        exit_mode=str(exit_variant.engine_exit_mode),
        trailing_atr_k=float(exit_variant.trailing_atr_k),
        max_hold_bars=int(exit_variant.max_hold_bars),
        stop_atr_multiple=float(exit_variant.stop_atr_multiple),
        take_profit_atr_multiple=float(exit_variant.take_profit_atr_multiple),
        round_trip_cost_pct=float(costs.get("round_trip_cost_pct", 0.1)),
        slippage_pct=float(costs.get("slippage_pct", 0.0005)),
        initial_capital=float(eval_cfg.get("initial_capital", 10000.0)),
        cost_model_cfg=cost_model_cfg,
    )


def _evaluate_walkforward_combo(
    *,
    frame: pd.DataFrame,
    symbol: str,
    strategy: StrategyVariant,
    exit_variant: ExitVariant,
    cfg: dict[str, Any],
    cost_model_cfg: dict[str, Any],
    windows: list[Any],
    trades_per_month: float,
    stage12_3_cfg: dict[str, Any],
    stage12_4_cfg: dict[str, Any],
    seed: int,
    trace: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not windows:
        return [], {"classification": "INSUFFICIENT_DATA", "usable_windows": 0, "min_trades_required_per_window": 0}
    wf_cfg = cfg.get("evaluation", {}).get("stage8", {}).get("walkforward_v2", {})
    min_trades_required = _resolve_min_trades_required(
        trades_per_month=trades_per_month,
        window_days=int(wf_cfg.get("forward_days", 30)),
        base_min_trades=float(wf_cfg.get("min_trades", 10)),
        stage12_3_cfg=stage12_3_cfg,
    )
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    rows: list[dict[str, Any]] = []
    for window in windows:
        holdout = frame.loc[(ts >= window.holdout_start) & (ts < window.holdout_end)].copy().reset_index(drop=True)
        forward = frame.loc[(ts >= window.forward_start) & (ts < window.forward_end)].copy().reset_index(drop=True)
        hold_signal_raw = strategy.signal_builder(holdout) if not holdout.empty else pd.Series(dtype=int)
        fwd_signal_raw = strategy.signal_builder(forward) if not forward.empty else pd.Series(dtype=int)
        hold_signal, _ = _stage12_4_scored_signal(
            frame=holdout,
            strategy=strategy,
            raw_signal=hold_signal_raw,
            stage12_4_cfg=stage12_4_cfg,
            seed=int(seed) + int(window.window_idx) + 10_000,
            trace=trace,
        ) if not holdout.empty else (pd.Series(dtype=int), {})
        fwd_signal, _ = _stage12_4_scored_signal(
            frame=forward,
            strategy=strategy,
            raw_signal=fwd_signal_raw,
            stage12_4_cfg=stage12_4_cfg,
            seed=int(seed) + int(window.window_idx) + 20_000,
            trace=trace,
        ) if not forward.empty else (pd.Series(dtype=int), {})
        hold_result = _run_backtest_with_signal(
            frame=holdout,
            signal=hold_signal,
            symbol=symbol,
            strategy_name=strategy.strategy_name,
            exit_variant=exit_variant,
            cfg=cfg,
            cost_model_cfg=cost_model_cfg,
        ) if not holdout.empty else None
        fwd_result = _run_backtest_with_signal(
            frame=forward,
            signal=fwd_signal,
            symbol=symbol,
            strategy_name=strategy.strategy_name,
            exit_variant=exit_variant,
            cfg=cfg,
            cost_model_cfg=cost_model_cfg,
        ) if not forward.empty else None
        hold_metrics = _window_metrics(result=hold_result, frame=holdout)
        fwd_metrics = _window_metrics(result=fwd_result, frame=forward)
        usable, reasons = _usable_window(
            forward_metrics=fwd_metrics,
            min_trades=float(min_trades_required),
            min_exposure=float(wf_cfg.get("min_exposure", 0.01)),
        )
        rows.append(
            {
                "window_idx": int(window.window_idx),
                "train_start": window.train_start.isoformat(),
                "train_end": window.train_end.isoformat(),
                "holdout_start": window.holdout_start.isoformat(),
                "holdout_end": window.holdout_end.isoformat(),
                "forward_start": window.forward_start.isoformat(),
                "forward_end": window.forward_end.isoformat(),
                "train_bars": int(((ts >= window.train_start) & (ts < window.train_end)).sum()),
                "holdout_bars": int(len(holdout)),
                "forward_bars": int(len(forward)),
                "usable": bool(usable),
                "exclude_reasons": ";".join(reasons),
                "holdout_expectancy": float(hold_metrics["expectancy"]),
                "holdout_profit_factor": float(hold_metrics["profit_factor"]),
                "holdout_max_drawdown": float(hold_metrics["max_drawdown"]),
                "holdout_return_pct": float(hold_metrics["return_pct"]),
                "holdout_trade_count": int(hold_metrics["trade_count"]),
                "holdout_exposure_ratio": float(hold_metrics["exposure_ratio"]),
                "forward_expectancy": float(fwd_metrics["expectancy"]),
                "forward_profit_factor": float(fwd_metrics["profit_factor"]),
                "forward_max_drawdown": float(fwd_metrics["max_drawdown"]),
                "forward_return_pct": float(fwd_metrics["return_pct"]),
                "forward_trade_count": int(fwd_metrics["trade_count"]),
                "forward_exposure_ratio": float(fwd_metrics["exposure_ratio"]),
                "min_trades_required": int(min_trades_required),
            }
        )
    summary = aggregate_windows(rows, cfg=cfg)
    summary["min_trades_required_per_window"] = int(min_trades_required)
    if rows:
        summary["avg_forward_trade_count"] = float(np.mean([float(row.get("forward_trade_count", 0.0)) for row in rows]))
    else:
        summary["avg_forward_trade_count"] = 0.0
    samples = trace["stage12_3"].get("adaptive_usability_samples", [])
    if len(samples) < 50:
        samples.append(
            {
                "window_count": int(len(rows)),
                "min_trades_required_per_window": int(min_trades_required),
                "avg_forward_trade_count": float(summary["avg_forward_trade_count"]),
            }
        )
        trace["stage12_3"]["adaptive_usability_samples"] = samples
    return rows, summary


def _window_metrics(result: Any, frame: pd.DataFrame) -> dict[str, float]:
    if result is None or frame.empty:
        return {
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
            "return_pct": 0.0,
            "trade_count": 0.0,
            "exposure_ratio": 0.0,
            "finite": True,
        }
    trades = result.trades.copy()
    equity = result.equity_curve.copy()
    trade_count = float(result.metrics.get("trade_count", 0.0))
    exposure_ratio = 0.0
    if not trades.empty and "bars_held" in trades.columns:
        exposure_ratio = float(pd.to_numeric(trades["bars_held"], errors="coerce").fillna(0.0).sum() / max(1, len(frame)))
    if not equity.empty:
        initial = float(equity["equity"].iloc[0])
        final = float(equity["equity"].iloc[-1])
        return_pct = float((final / initial) - 1.0) if initial != 0 else 0.0
    else:
        return_pct = 0.0
    payload = {
        "expectancy": _finite(result.metrics.get("expectancy", 0.0), default=0.0),
        "profit_factor": _finite(result.metrics.get("profit_factor", 0.0), default=0.0),
        "max_drawdown": _finite(result.metrics.get("max_drawdown", 0.0), default=0.0),
        "return_pct": _finite(return_pct, default=0.0),
        "trade_count": _finite(trade_count, default=0.0),
        "exposure_ratio": _finite(exposure_ratio, default=0.0),
    }
    payload["finite"] = all(np.isfinite(float(value)) for value in payload.values())
    return payload


def _resolve_min_trades_required(
    *,
    trades_per_month: float,
    window_days: int,
    base_min_trades: float,
    stage12_3_cfg: dict[str, Any],
) -> int:
    min_required = int(max(1, round(float(base_min_trades))))
    if not bool(stage12_3_cfg.get("enabled", False)):
        return min_required
    adaptive_cfg = dict(stage12_3_cfg.get("usability_adaptive", {}))
    if not bool(adaptive_cfg.get("enabled", True)):
        return min_required
    min_floor = int(adaptive_cfg.get("min_floor", 5))
    max_floor = int(adaptive_cfg.get("max_floor", 80))
    alpha = float(adaptive_cfg.get("alpha", 0.35))
    expected = float(max(0.0, trades_per_month) * (max(1, int(window_days)) / 30.0))
    adaptive = int(math.floor(expected * max(0.0, alpha)))
    clipped = int(max(min_floor, min(max_floor, adaptive)))
    return int(max(1, clipped))


def _stage12_4_scored_signal(
    *,
    frame: pd.DataFrame,
    strategy: StrategyVariant,
    raw_signal: pd.Series,
    stage12_4_cfg: dict[str, Any],
    seed: int,
    trace: dict[str, Any],
) -> tuple[pd.Series, dict[str, Any]]:
    raw = pd.to_numeric(raw_signal, errors="coerce").fillna(0).astype(int)
    if frame.empty:
        return raw, {"enabled": False, "cache_hit": False, "search_evaluations": 0}
    if not bool(stage12_4_cfg.get("enabled", False)):
        return raw, {"enabled": False, "cache_hit": False, "search_evaluations": 0}

    cache_cfg = dict(stage12_4_cfg.get("cache", {}))
    cache_enabled = bool(cache_cfg.get("enabled", True))
    key_payload = {
        "strategy_key": str(strategy.strategy_key),
        "rows": int(len(frame)),
        "seed": int(seed),
        "threshold_grid": list(stage12_4_cfg.get("threshold_grid", [])),
        "weight_values": list(stage12_4_cfg.get("weight_values", [])),
        "frame_hash": stable_hash(
            frame.loc[:, [c for c in ("timestamp", "close", "atr_14", "ema_50", "bb_mid_20", "atr_pct_rank_252") if c in frame.columns]]
            .to_dict(orient="list"),
            length=16,
        ),
        "signal_hash": stable_hash(raw.tolist(), length=16),
    }
    cache_key = stable_hash(key_payload, length=24)
    if cache_enabled and cache_key in _STAGE12_4_SCORE_CACHE:
        cached = dict(_STAGE12_4_SCORE_CACHE[cache_key])
        gated_values = np.asarray(cached.get("gated_signal_values", []), dtype=int)
        if gated_values.shape[0] == len(frame):
            trace["stage12_4"]["cache_hit_count"] = int(trace["stage12_4"].get("cache_hit_count", 0)) + 1
            return pd.Series(gated_values, index=frame.index, dtype=int), {
                "enabled": True,
                "cache_hit": True,
                "search_evaluations": int(cached.get("search_evaluations", 0)),
                "chosen_threshold": float(cached.get("chosen_threshold", 0.0)),
                "chosen_weights": list(cached.get("chosen_weights", [1.0, 1.0, 1.0])),
                "raw_signal_count": int((raw != 0).sum()),
                "gated_signal_count": int((gated_values != 0).sum()),
            }

    components = _stage12_4_score_components(frame=frame)
    trace["stage12_4"]["score_computed_count"] = int(trace["stage12_4"].get("score_computed_count", 0)) + 1
    trace["stage12_4"]["cache_miss_count"] = int(trace["stage12_4"].get("cache_miss_count", 0)) + 1
    threshold_grid = [float(v) for v in stage12_4_cfg.get("threshold_grid", [0.5])]
    weight_values = [float(v) for v in stage12_4_cfg.get("weight_values", [1.0])]
    tpm_cfg = dict(stage12_4_cfg.get("trade_rate_target", {}))
    tpm_min = float(tpm_cfg.get("tpm_min", 2.0))
    tpm_max = float(tpm_cfg.get("tpm_max", 40.0))
    months = _estimate_months(frame)
    raw_count = int((raw != 0).sum())
    active_mask = (raw != 0).to_numpy(dtype=bool)

    best_score = -float("inf")
    best_threshold = threshold_grid[0]
    best_weights = (1.0, 1.0, 1.0)
    best_signal = raw.to_numpy(dtype=int, copy=True)
    evaluations = 0
    for w1 in weight_values:
        for w2 in weight_values:
            for w3 in weight_values:
                weight_sum = float(max(1e-12, w1 + w2 + w3))
                combined = (w1 * components[:, 0] + w2 * components[:, 1] + w3 * components[:, 2]) / weight_sum
                for threshold in threshold_grid:
                    evaluations += 1
                    gated = raw.to_numpy(dtype=int, copy=True)
                    gated[(active_mask) & (combined < float(threshold))] = 0
                    gated_count = int(np.count_nonzero(gated))
                    tpm = float(gated_count / months) if months > 0 else 0.0
                    in_band = float(tpm_min <= tpm <= tpm_max)
                    band_penalty = 0.0
                    if tpm < tpm_min:
                        band_penalty = float(tpm_min - tpm)
                    elif tpm > tpm_max:
                        band_penalty = float(tpm - tpm_max)
                    keep_ratio = float(gated_count / raw_count) if raw_count > 0 else 0.0
                    objective = (5.0 * in_band) + keep_ratio - (0.1 * band_penalty)
                    if objective > best_score:
                        best_score = objective
                        best_threshold = float(threshold)
                        best_weights = (float(w1), float(w2), float(w3))
                        best_signal = gated

    trace["stage12_4"]["threshold_eval_count"] = int(trace["stage12_4"].get("threshold_eval_count", 0)) + int(evaluations)
    out_signal = pd.Series(best_signal, index=frame.index, dtype=int)
    meta = {
        "enabled": True,
        "cache_hit": False,
        "search_evaluations": int(evaluations),
        "chosen_threshold": float(best_threshold),
        "chosen_weights": [float(best_weights[0]), float(best_weights[1]), float(best_weights[2])],
        "raw_signal_count": int(raw_count),
        "gated_signal_count": int(np.count_nonzero(best_signal)),
    }
    if cache_enabled:
        _STAGE12_4_SCORE_CACHE[cache_key] = {
            **meta,
            "gated_signal_values": out_signal.to_numpy(dtype=int).tolist(),
        }
    return out_signal, meta


def _stage12_4_score_components(frame: pd.DataFrame) -> np.ndarray:
    n = len(frame)
    if n == 0:
        return np.zeros((0, 3), dtype=float)
    close = pd.to_numeric(frame.get("close", np.nan), errors="coerce").astype(float)
    ema50 = pd.to_numeric(frame.get("ema_50", close), errors="coerce").astype(float)
    atr = pd.to_numeric(frame.get("atr_14", 1.0), errors="coerce").astype(float).replace(0.0, np.nan)
    atr_rank = pd.to_numeric(frame.get("atr_pct_rank_252", 0.5), errors="coerce").astype(float)
    slope = pd.to_numeric(frame.get("ema_slope_50", 0.0), errors="coerce").astype(float)
    bb_mid = pd.to_numeric(frame.get("bb_mid_20", close), errors="coerce").astype(float)

    comp_trend = np.clip(np.nan_to_num(np.abs(slope.to_numpy(dtype=float)) / 0.01, nan=0.0), 0.0, 1.0)
    comp_vol = np.clip(np.nan_to_num(atr_rank.to_numpy(dtype=float), nan=0.0), 0.0, 1.0)
    comp_dist = np.abs((close - ema50) / (atr + 1e-12))
    comp_mid = np.abs((close - bb_mid) / (atr + 1e-12))
    comp_revert = np.clip(np.nan_to_num(((comp_dist + comp_mid) / 2.0).to_numpy(dtype=float) / 3.0, nan=0.0), 0.0, 1.0)
    out = np.column_stack([comp_trend, comp_vol, comp_revert]).astype(float, copy=False)
    if out.shape != (n, 3):
        return np.zeros((n, 3), dtype=float)
    return out


def _soft_weight_series(
    *,
    frame: pd.DataFrame,
    strategy: StrategyVariant,
    stage12_3_cfg: dict[str, Any],
) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    idx = frame.index
    if not bool(stage12_3_cfg.get("enabled", False)):
        return pd.Series(np.ones(len(frame), dtype=float), index=idx, dtype=float)
    soft_cfg = dict(stage12_3_cfg.get("soft_weights", {}))
    if not bool(soft_cfg.get("enabled", True)):
        return pd.Series(np.ones(len(frame), dtype=float), index=idx, dtype=float)

    min_weight = float(soft_cfg.get("min_weight", 0.25))
    regime_mismatch_weight = float(soft_cfg.get("regime_mismatch_weight", 0.5))
    vol_mismatch_weight = float(soft_cfg.get("vol_mismatch_weight", 0.5))

    trend_score = pd.to_numeric(frame.get("score_trend", 0.5), errors="coerce").fillna(0.5).to_numpy(dtype=float)
    range_score = pd.to_numeric(frame.get("score_range", 0.5), errors="coerce").fillna(0.5).to_numpy(dtype=float)
    vol_rank = pd.to_numeric(frame.get("atr_pct_rank_252", 0.5), errors="coerce").fillna(0.5).to_numpy(dtype=float)
    family = _strategy_family_label(strategy=strategy)

    if family == "mean_reversion":
        regime_match = range_score >= trend_score
        vol_match = vol_rank <= 0.75
    elif family in {"trend", "breakout"}:
        regime_match = trend_score >= range_score
        vol_match = vol_rank >= 0.25
    else:
        regime_match = np.ones(len(frame), dtype=bool)
        vol_match = np.ones(len(frame), dtype=bool)

    regime_weight = np.where(regime_match, 1.0, regime_mismatch_weight)
    vol_weight = np.where(vol_match, 1.0, vol_mismatch_weight)
    final = np.clip(regime_weight * vol_weight, min_weight, 1.0)
    return pd.Series(final.astype(float), index=idx, dtype=float)


def _strategy_family_label(*, strategy: StrategyVariant) -> str:
    key = str(strategy.strategy_key)
    if key in {"BreakoutRetest", "MA_SlopePullback"}:
        return "trend"
    if key in {"VolCompressionBreakout", "Donchian_Breakout", "Range_Breakout_w/_EMA_Trend_Filter"}:
        return "breakout"
    if key in {"BollingerSnapBack", "ATR_DistanceRevert", "RangeFade", "RSI_Mean_Reversion"}:
        return "mean_reversion"
    if "Trend_Pullback" in key or key == "Trend_Pullback":
        return "trend"
    return "other"


def _usable_window(forward_metrics: dict[str, float], min_trades: float, min_exposure: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not bool(forward_metrics.get("finite", False)):
        reasons.append("non_finite_metrics")
    if float(forward_metrics.get("trade_count", 0.0)) < float(min_trades):
        reasons.append("min_trades")
    if float(forward_metrics.get("exposure_ratio", 0.0)) < float(min_exposure):
        reasons.append("min_exposure")
    return len(reasons) == 0, reasons


def _metrics_from_backtest(backtest_result: Any, frame: pd.DataFrame) -> dict[str, float]:
    metrics = dict(backtest_result.metrics)
    trade_count = _finite(metrics.get("trade_count", 0.0), default=0.0)
    expectancy = _finite(metrics.get("expectancy", 0.0), default=0.0)
    profit_factor = _finite(metrics.get("profit_factor", 0.0), default=0.0, clip=10.0)
    max_drawdown = _finite(metrics.get("max_drawdown", 0.0), default=0.0)
    pnl_values = pd.to_numeric(backtest_result.trades.get("pnl", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    exp_lcb = _exp_lcb(pnl_values)
    months = _estimate_months(frame)
    trades_per_month = float(trade_count / months) if months > 0 else 0.0
    return {
        "trade_count": trade_count,
        "expectancy": expectancy,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "exp_lcb": exp_lcb,
        "trades_per_month": trades_per_month,
    }


def _estimate_months(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return 0.0
    days = float((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400.0)
    return max(days / 30.0, 1e-6)


def _exp_lcb(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    mean = float(np.mean(values))
    if values.size <= 1:
        return mean
    std = float(np.std(values, ddof=0))
    return float(mean - std / math.sqrt(float(values.size)))


def _apply_cost_sensitivity_and_robust_score(leaderboard: pd.DataFrame, stage12_cfg: dict[str, Any]) -> pd.DataFrame:
    frame = leaderboard.copy()
    penalties = stage12_cfg.get("robustness", {}).get("instability_penalty", {})
    penalty_weight = float(stage12_cfg.get("robustness", {}).get("cost_sensitivity_penalty_weight", 1.0))
    group_cols = ["symbol", "timeframe", "strategy", "strategy_key", "strategy_source", "exit_type"]
    for _, group in frame.groupby(group_cols, sort=True):
        idx = list(group.index)
        ranked = group.assign(_rank=group["cost_level"].map(_COST_LEVEL_ORDER)).dropna(subset=["_rank"]).sort_values("_rank")
        x = ranked["_rank"].to_numpy(dtype=float)
        y = ranked["exp_lcb"].to_numpy(dtype=float)
        if len(x) >= 2:
            slope = float(abs(np.polyfit(x, y, 1)[0]))
        else:
            slope = 0.0
        median_exp_lcb = float(np.median(y)) if len(y) else 0.0
        instability = max(
            float(penalties.get(str(label), penalties.get("INSUFFICIENT_DATA", 1.0)))
            for label in group["stability_classification"].tolist()
        )
        robust_score = float(median_exp_lcb - (penalty_weight * slope) - instability)
        frame.loc[idx, "cost_sensitivity_slope"] = slope
        frame.loc[idx, "robust_score"] = robust_score
    return frame


def _apply_monte_carlo(
    *,
    leaderboard: pd.DataFrame,
    trade_pnls_by_combo: dict[str, np.ndarray],
    stage12_cfg: dict[str, Any],
    seed: int,
) -> tuple[pd.DataFrame, set[str]]:
    mc_cfg = stage12_cfg.get("monte_carlo", {})
    if not bool(mc_cfg.get("enabled", True)):
        return leaderboard, set()
    frame = leaderboard.copy()
    valid = frame.loc[frame["is_valid"] == True].copy()  # noqa: E712
    if valid.empty:
        return frame, set()
    valid = valid.sort_values("exp_lcb", ascending=False).reset_index(drop=True)
    top_pct = float(mc_cfg.get("top_pct", 0.2))
    top_n = max(1, int(math.ceil(len(valid) * top_pct)))
    selected_keys = valid.iloc[:top_n]["combo_key"].tolist()
    executed: set[str] = set()
    for idx, combo_key in enumerate(selected_keys):
        pnls = np.asarray(trade_pnls_by_combo.get(combo_key, np.asarray([], dtype=float)), dtype=float)
        if pnls.size == 0:
            continue
        paths = simulate_equity_paths(
            trade_pnls=pnls,
            n_paths=int(mc_cfg.get("n_paths", 5000)),
            method=str(mc_cfg.get("bootstrap", "block")),
            seed=int(seed) + int(idx),
            initial_equity=float(mc_cfg.get("initial_equity", 10000.0)),
            leverage=1.0,
            block_size_trades=int(mc_cfg.get("block_size_trades", 10)),
        )
        summary = summarize_mc(
            paths_results=paths,
            initial_equity=float(mc_cfg.get("initial_equity", 10000.0)),
            ruin_dd_threshold=float(mc_cfg.get("ruin_dd_threshold", 0.5)),
        )
        final_equity = pd.to_numeric(paths["final_equity"], errors="coerce").dropna().to_numpy(dtype=float)
        expected_log_growth = float(
            np.log(np.clip(final_equity, 1e-12, None) / float(mc_cfg.get("initial_equity", 10000.0))).mean()
        )
        frame.loc[frame["combo_key"] == combo_key, "MC_p_ruin"] = float(summary["tail_probabilities"]["p_ruin"])
        frame.loc[frame["combo_key"] == combo_key, "MC_p_return_negative"] = float(summary["tail_probabilities"]["p_return_lt_0"])
        frame.loc[frame["combo_key"] == combo_key, "MC_maxDD_p95"] = float(summary["max_drawdown"]["p95"])
        frame.loc[frame["combo_key"] == combo_key, "MC_expected_log_growth"] = expected_log_growth
        executed.add(str(combo_key))
    return frame, executed


def _final_verdict(leaderboard: pd.DataFrame) -> str:
    valid = leaderboard.loc[leaderboard["is_valid"] == True].copy()  # noqa: E712
    if valid.empty:
        return "NO ROBUST EDGE"
    stable = valid.loc[valid["stability_classification"] == "STABLE"].copy()
    if stable.empty:
        if float(valid["robust_score"].max()) > 0:
            return "WEAK EDGE"
        return "NO ROBUST EDGE"
    top = stable.sort_values("robust_score", ascending=False).iloc[0]
    p_ruin = float(top.get("MC_p_ruin", math.nan))
    if float(top.get("robust_score", 0.0)) > 0 and (math.isnan(p_ruin) or p_ruin < 0.01):
        return "STRONG EDGE FOUND"
    if float(top.get("robust_score", 0.0)) > 0:
        return "WEAK EDGE"
    return "NO ROBUST EDGE"


def _top_robust_rows(leaderboard: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    if leaderboard.empty:
        return []
    cols = [
        "symbol",
        "timeframe",
        "strategy",
        "exit_type",
        "cost_level",
        "exp_lcb",
        "PF",
        "expectancy",
        "robust_score",
        "cost_sensitivity_slope",
        "stability_classification",
        "MC_p_ruin",
        "MC_p_return_negative",
    ]
    top = leaderboard.sort_values(["robust_score", "exp_lcb"], ascending=[False, False]).head(limit)
    records = top.loc[:, cols].to_dict(orient="records")
    out: list[dict[str, Any]] = []
    for row in records:
        safe_row: dict[str, Any] = {}
        for key, value in row.items():
            if isinstance(value, (float, np.floating)) and not np.isfinite(float(value)):
                safe_row[key] = None
            else:
                safe_row[key] = value
        out.append(safe_row)
    return out


def _build_trade_rate_audit(
    *,
    leaderboard: pd.DataFrame,
    forensic_matrix: pd.DataFrame,
    min_usable_windows_valid: int,
) -> pd.DataFrame:
    joined = leaderboard.merge(
        forensic_matrix[
            [
                "combo_key",
                "invalid_reason",
                "walkforward_expected_windows",
                "walkforward_windows_count",
                "min_trades_required_per_window",
            ]
        ],
        on="combo_key",
        how="left",
    )
    joined["zero_trade"] = pd.to_numeric(joined["trade_count"], errors="coerce").fillna(0.0) <= 0.0
    joined["top_reject_reason"] = joined["invalid_reason"].fillna("VALID")
    joined["walkforward_usable_windows"] = pd.to_numeric(joined["usable_windows"], errors="coerce").fillna(0).astype(int)
    joined["walkforward_expected_windows"] = (
        pd.to_numeric(joined["walkforward_expected_windows"], errors="coerce").fillna(0).astype(int)
    )
    joined["min_trades_required_per_window"] = (
        pd.to_numeric(joined["min_trades_required_per_window"], errors="coerce").fillna(0).astype(int)
    )
    joined["avg_weight"] = pd.to_numeric(joined["avg_weight"], errors="coerce").fillna(1.0).astype(float)
    joined["usability_passed"] = joined["walkforward_usable_windows"] >= int(min_usable_windows_valid)

    cols = [
        "symbol",
        "timeframe",
        "strategy",
        "exit_type",
        "cost_level",
        "trade_count",
        "trades_per_month",
        "zero_trade",
        "top_reject_reason",
        "walkforward_expected_windows",
        "walkforward_usable_windows",
        "min_trades_required_per_window",
        "avg_weight",
        "usability_passed",
    ]
    out = joined.loc[:, cols].copy()
    out = out.rename(columns={"trade_count": "trade_count_total", "exit_type": "exit"})
    return out.sort_values(["symbol", "timeframe", "strategy", "exit", "cost_level"]).reset_index(drop=True)


def _build_stage12_3_metrics(
    *,
    forensic_summary: dict[str, Any],
    target_thresholds: dict[str, float],
) -> dict[str, Any]:
    zero_trade_pct = float(forensic_summary.get("zero_trade_pct", 0.0))
    walkforward_pct = float(forensic_summary.get("walkforward_executed_true_pct", 0.0))
    mc_trigger_rate = float(forensic_summary.get("mc_trigger_rate", 0.0))
    invalid_pct = float(forensic_summary.get("invalid_pct", 0.0))
    pass_map = {
        "zero_trade_pct": zero_trade_pct <= float(target_thresholds["zero_trade_pct_max"]),
        "walkforward_executed_true_pct": walkforward_pct >= float(target_thresholds["walkforward_executed_true_pct_min"]),
        "MC_trigger_rate": mc_trigger_rate >= float(target_thresholds["mc_trigger_rate_min"]),
        "invalid_pct": invalid_pct <= float(target_thresholds["invalid_pct_max"]),
    }
    return {
        "zero_trade_pct": zero_trade_pct,
        "walkforward_executed_true_pct": walkforward_pct,
        "MC_trigger_rate": mc_trigger_rate,
        "invalid_pct": invalid_pct,
        "targets": dict(target_thresholds),
        "target_pass_map": pass_map,
        "passed": bool(all(pass_map.values())),
        "status": "PASSED" if all(pass_map.values()) else "FAILED",
    }


def _write_stage12_3_docs(
    *,
    report_md: Path,
    report_json: Path,
    run_id: str,
    stage12_3_metrics: dict[str, Any],
    trade_rate_audit: pd.DataFrame,
    leaderboard: pd.DataFrame,
    baseline_stage12_1: Path,
) -> None:
    payload = dict(stage12_3_metrics)
    payload["run_id"] = str(run_id)
    report_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    baseline = {}
    if baseline_stage12_1.exists():
        try:
            baseline = json.loads(baseline_stage12_1.read_text(encoding="utf-8"))
        except Exception:
            baseline = {}

    reason_breakdown = (
        trade_rate_audit["top_reject_reason"].astype(str).value_counts(normalize=True).sort_index() * 100.0
        if not trade_rate_audit.empty
        else pd.Series(dtype=float)
    )
    valid_top = leaderboard.loc[leaderboard["is_valid"] == True].sort_values("robust_score", ascending=False).head(10)  # noqa: E712

    lines: list[str] = []
    lines.append("# Stage-12.3 Unblocking Report")
    lines.append("")
    lines.append(f"- run_id: `{run_id}`")
    lines.append(f"- status: `Stage-12.3 {payload['status']}`")
    lines.append("")
    lines.append("## Target Metrics")
    lines.append("| metric | value | target | pass |")
    lines.append("| --- | ---: | ---: | --- |")
    lines.append(
        f"| zero_trade_pct | {float(payload['zero_trade_pct']):.6f} | <= {float(payload['targets']['zero_trade_pct_max']):.6f} | {bool(payload['target_pass_map']['zero_trade_pct'])} |"
    )
    lines.append(
        f"| walkforward_executed_true_pct | {float(payload['walkforward_executed_true_pct']):.6f} | >= {float(payload['targets']['walkforward_executed_true_pct_min']):.6f} | {bool(payload['target_pass_map']['walkforward_executed_true_pct'])} |"
    )
    lines.append(
        f"| MC_trigger_rate | {float(payload['MC_trigger_rate']):.6f} | >= {float(payload['targets']['mc_trigger_rate_min']):.6f} | {bool(payload['target_pass_map']['MC_trigger_rate'])} |"
    )
    lines.append(
        f"| invalid_pct | {float(payload['invalid_pct']):.6f} | <= {float(payload['targets']['invalid_pct_max']):.6f} | {bool(payload['target_pass_map']['invalid_pct'])} |"
    )
    lines.append("")
    lines.append("## Before vs After (Stage-12.1 baseline, if available)")
    lines.append("| metric | baseline | stage12_3 |")
    lines.append("| --- | ---: | ---: |")
    for key in ("zero_trade_pct", "walkforward_executed_true_pct", "mc_trigger_rate", "invalid_pct"):
        baseline_value = baseline.get(key, None)
        if baseline_value is None:
            baseline_text = "N/A"
        else:
            baseline_text = f"{float(baseline_value):.6f}"
        lines.append(f"| {key} | {baseline_text} | {float(payload.get(key if key != 'mc_trigger_rate' else 'MC_trigger_rate', 0.0)):.6f} |")
    lines.append("")
    lines.append("## Reject Reason Breakdown")
    lines.append("| reason | pct |")
    lines.append("| --- | ---: |")
    for reason, pct in reason_breakdown.items():
        lines.append(f"| {reason} | {float(pct):.6f} |")
    lines.append("")
    lines.append("## Top 10 VALID by robust_score")
    if valid_top.empty:
        lines.append("- no valid combinations")
    else:
        lines.append("| symbol | timeframe | strategy | exit | cost_level | robust_score |")
        lines.append("| --- | --- | --- | --- | --- | ---: |")
        for row in valid_top.to_dict(orient="records"):
            lines.append(
                f"| {row.get('symbol','')} | {row.get('timeframe','')} | {row.get('strategy','')} | "
                f"{row.get('exit_type','')} | {row.get('cost_level','')} | {float(row.get('robust_score', 0.0)):.6f} |"
            )
    lines.append("")
    if not bool(payload["passed"]):
        lines.append("## Why Stage-12.3 failed and what will be changed in Stage-12.4")
        lines.append(
            "- Stage-12.3 soft weighting and adaptive usability improved diagnostics coverage but did not satisfy all target metrics. "
            "Stage-12.4 will add a deterministic score-based qualification wrapper with bounded search and explicit trade-rate constraints."
        )
    else:
        lines.append("## Conclusion")
        lines.append("- Stage-12.3 PASSED")

    report_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _build_stage12_4_metrics(
    *,
    stage12_4_table: pd.DataFrame,
    forensic_summary: dict[str, Any],
    leaderboard: pd.DataFrame,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {
            "enabled": False,
            "status": "SKIPPED",
            "zero_trade_pct": float(forensic_summary.get("zero_trade_pct", 0.0)),
            "walkforward_executed_true_pct": float(forensic_summary.get("walkforward_executed_true_pct", 0.0)),
            "MC_trigger_rate": float(forensic_summary.get("mc_trigger_rate", 0.0)),
            "invalid_pct": float(forensic_summary.get("invalid_pct", 0.0)),
            "threshold_distribution": {},
            "weight_distribution": {},
            "trade_rate_distribution": {},
            "cache_hit_rate": 0.0,
        }
    table = stage12_4_table.copy()
    if table.empty:
        return {
            "enabled": True,
            "status": "FAILED",
            "zero_trade_pct": float(forensic_summary.get("zero_trade_pct", 0.0)),
            "walkforward_executed_true_pct": float(forensic_summary.get("walkforward_executed_true_pct", 0.0)),
            "MC_trigger_rate": float(forensic_summary.get("mc_trigger_rate", 0.0)),
            "invalid_pct": float(forensic_summary.get("invalid_pct", 0.0)),
            "threshold_distribution": {},
            "weight_distribution": {},
            "trade_rate_distribution": {},
            "cache_hit_rate": 0.0,
        }
    threshold_dist = (
        pd.to_numeric(table["chosen_threshold"], errors="coerce")
        .round(4)
        .astype(str)
        .value_counts(normalize=True)
        .sort_index()
    )
    weight_dist = (
        table["chosen_weights"]
        .astype(str)
        .value_counts(normalize=True)
        .sort_index()
    )
    gated_count = pd.to_numeric(table["gated_signal_count"], errors="coerce").fillna(0.0)
    raw_count = pd.to_numeric(table["raw_signal_count"], errors="coerce").fillna(0.0)
    kept_ratio = np.where(raw_count > 0.0, gated_count / raw_count, 0.0)
    trade_rate_dist = (
        pd.Series(kept_ratio, dtype=float)
        .round(4)
        .astype(str)
        .value_counts(normalize=True)
        .sort_index()
    )
    cache_hit_rate = float(table["cache_hit"].astype(bool).mean())
    target_thresholds = {
        "zero_trade_pct_max": 25.0,
        "walkforward_executed_true_pct_min": 40.0,
        "mc_trigger_rate_min": 10.0,
        "invalid_pct_max": 60.0,
    }
    zero_trade_pct = float(forensic_summary.get("zero_trade_pct", 0.0))
    walkforward_pct = float(forensic_summary.get("walkforward_executed_true_pct", 0.0))
    mc_trigger_rate = float(forensic_summary.get("mc_trigger_rate", 0.0))
    invalid_pct = float(forensic_summary.get("invalid_pct", 0.0))
    pass_map = {
        "zero_trade_pct": zero_trade_pct <= float(target_thresholds["zero_trade_pct_max"]),
        "walkforward_executed_true_pct": walkforward_pct >= float(target_thresholds["walkforward_executed_true_pct_min"]),
        "MC_trigger_rate": mc_trigger_rate >= float(target_thresholds["mc_trigger_rate_min"]),
        "invalid_pct": invalid_pct <= float(target_thresholds["invalid_pct_max"]),
    }
    return {
        "enabled": True,
        "status": "PASSED" if all(pass_map.values()) else "FAILED",
        "zero_trade_pct": zero_trade_pct,
        "walkforward_executed_true_pct": walkforward_pct,
        "MC_trigger_rate": mc_trigger_rate,
        "invalid_pct": invalid_pct,
        "targets": target_thresholds,
        "target_pass_map": pass_map,
        "threshold_distribution": {str(k): float(v) for k, v in threshold_dist.items()},
        "weight_distribution": {str(k): float(v) for k, v in weight_dist.items()},
        "trade_rate_distribution": {str(k): float(v) for k, v in trade_rate_dist.items()},
        "cache_hit_rate": cache_hit_rate,
        "search_evaluations_max": int(pd.to_numeric(table["search_evaluations"], errors="coerce").fillna(0).max()),
        "search_evaluations_mean": float(pd.to_numeric(table["search_evaluations"], errors="coerce").fillna(0).mean()),
        "valid_combinations": int((leaderboard["is_valid"] == True).sum()),  # noqa: E712
    }


def _write_stage12_4_docs(
    *,
    report_md: Path,
    report_json: Path,
    run_id: str,
    stage12_4_metrics: dict[str, Any],
    stage12_3_metrics: dict[str, Any],
) -> None:
    payload = dict(stage12_4_metrics)
    payload["run_id"] = str(run_id)
    report_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Stage-12.4 Score-Based Signal Family Report")
    lines.append("")
    lines.append(f"- run_id: `{run_id}`")
    lines.append(f"- status: `{payload.get('status', 'SKIPPED')}`")
    lines.append("")
    lines.append("## Stage-12.3 vs Stage-12.4 Metrics")
    lines.append("| metric | stage12_3 | stage12_4 |")
    lines.append("| --- | ---: | ---: |")
    lines.append(f"| zero_trade_pct | {float(stage12_3_metrics.get('zero_trade_pct', 0.0)):.6f} | {float(payload.get('zero_trade_pct', 0.0)):.6f} |")
    lines.append(
        f"| walkforward_executed_true_pct | {float(stage12_3_metrics.get('walkforward_executed_true_pct', 0.0)):.6f} | {float(payload.get('walkforward_executed_true_pct', 0.0)):.6f} |"
    )
    lines.append(f"| MC_trigger_rate | {float(stage12_3_metrics.get('MC_trigger_rate', 0.0)):.6f} | {float(payload.get('MC_trigger_rate', 0.0)):.6f} |")
    lines.append(f"| invalid_pct | {float(stage12_3_metrics.get('invalid_pct', 0.0)):.6f} | {float(payload.get('invalid_pct', 0.0)):.6f} |")
    lines.append("")
    lines.append("## Threshold Distribution")
    lines.append("| threshold | share |")
    lines.append("| --- | ---: |")
    for key, value in sorted((payload.get("threshold_distribution") or {}).items()):
        lines.append(f"| {key} | {float(value):.6f} |")
    lines.append("")
    lines.append("## Weight Distribution")
    lines.append("| weights | share |")
    lines.append("| --- | ---: |")
    for key, value in sorted((payload.get("weight_distribution") or {}).items()):
        lines.append(f"| {key} | {float(value):.6f} |")
    lines.append("")
    lines.append("## Trade-Rate Distribution (kept_signal_ratio)")
    lines.append("| kept_ratio | share |")
    lines.append("| --- | ---: |")
    for key, value in sorted((payload.get("trade_rate_distribution") or {}).items()):
        lines.append(f"| {key} | {float(value):.6f} |")
    lines.append("")
    lines.append(f"- cache_hit_rate: `{float(payload.get('cache_hit_rate', 0.0)):.6f}`")
    lines.append(f"- search_evaluations_max: `{int(payload.get('search_evaluations_max', 0))}`")
    lines.append(f"- search_evaluations_mean: `{float(payload.get('search_evaluations_mean', 0.0)):.6f}`")

    report_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _reject_stage_label(*, invalid_reason: str | None, raw_trade_count: float, posthook_trade_count: float) -> str:
    reason = str(invalid_reason or "")
    if reason == "ZERO_TRADE":
        if float(raw_trade_count) <= 0:
            return "prehook"
        if float(posthook_trade_count) <= 0:
            return "posthook"
    if reason == "LOW_USABLE_WINDOWS":
        return "posthook"
    if reason:
        return "posthook"
    return "posthook"


def _family_display_name(family: str) -> str:
    mapping = {
        "BreakoutRetest": "Breakout Retest",
        "MA_SlopePullback": "MA SlopePullback",
        "VolCompressionBreakout": "Vol Compression Breakout",
        "BollingerSnapBack": "Bollinger SnapBack",
        "ATR_DistanceRevert": "ATR Distance Revert",
        "RangeFade": "Range Fade",
    }
    return mapping.get(str(family), str(family))


def _combo_key(*, symbol: str, timeframe: str, strategy: str, exit_type: str, cost_level: str) -> str:
    return stable_hash(
        {
            "symbol": str(symbol),
            "timeframe": str(timeframe),
            "strategy": str(strategy),
            "exit_type": str(exit_type),
            "cost_level": str(cost_level),
        },
        length=20,
    )


def _write_docs_report(leaderboard: pd.DataFrame, summary: dict[str, Any], report_md: Path, report_json: Path) -> None:
    validate_stage12_summary_schema(summary)
    report_json.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Stage-12 Full Price-Family Robustness Sweep")
    lines.append("")
    lines.append(f"- run_id: `{summary['run_id']}`")
    lines.append(f"- symbols: `{', '.join(summary['symbols'])}`")
    lines.append(f"- timeframes: `{', '.join(summary['timeframes'])}`")
    lines.append(f"- total_combinations: `{summary['total_combinations']}`")
    lines.append(f"- valid_combinations: `{summary['valid_combinations']}`")
    lines.append(f"- runtime_seconds: `{float(summary['runtime_seconds']):.3f}`")
    lines.append(f"- verdict: `{summary['verdict']}`")
    lines.append("")
    lines.append("## Runtime Breakdown by Timeframe")
    lines.append("| timeframe | runtime_seconds |")
    lines.append("| --- | ---: |")
    for timeframe, seconds in sorted(summary.get("runtime_by_timeframe", {}).items()):
        lines.append(f"| {timeframe} | {float(seconds):.4f} |")
    lines.append("")

    lines.append("## Top 10 Robust Combinations")
    lines.append("| symbol | timeframe | strategy | exit_type | cost_level | exp_lcb | PF | expectancy | robust_score | cost_sensitivity | stability | MC_p_ruin | MC_p_return_negative |")
    lines.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |")
    for row in summary.get("top_robust", []):
        lines.append(
            f"| {row.get('symbol','')} | {row.get('timeframe','')} | {row.get('strategy','')} | "
            f"{row.get('exit_type','')} | {row.get('cost_level','')} | {_safe_float(row.get('exp_lcb',0.0)):.6f} | "
            f"{_safe_float(row.get('PF',0.0)):.6f} | {_safe_float(row.get('expectancy',0.0)):.6f} | "
            f"{_safe_float(row.get('robust_score',0.0)):.6f} | {_safe_float(row.get('cost_sensitivity_slope',0.0)):.6f} | "
            f"{row.get('stability_classification','')} | {_safe_float(row.get('MC_p_ruin', math.nan)):.6f} | "
            f"{_safe_float(row.get('MC_p_return_negative', math.nan)):.6f} |"
        )
    lines.append("")

    lines.append("## Per-Timeframe Summary")
    tf_summary = (
        leaderboard.groupby("timeframe", dropna=False)
        .agg(
            combos=("combo_key", "count"),
            valid=("is_valid", "sum"),
            exp_lcb_median=("exp_lcb", "median"),
            robust_top=("robust_score", "max"),
        )
        .reset_index()
        .sort_values("timeframe")
    )
    lines.append("| timeframe | combos | valid | exp_lcb_median | best_robust_score |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in tf_summary.to_dict(orient="records"):
        lines.append(
            f"| {row['timeframe']} | {int(row['combos'])} | {int(row['valid'])} | "
            f"{float(row['exp_lcb_median']):.6f} | {float(row['robust_top']):.6f} |"
        )
    lines.append("")

    lines.append("## Per-Symbol Summary")
    sym_summary = (
        leaderboard.groupby("symbol", dropna=False)
        .agg(
            combos=("combo_key", "count"),
            valid=("is_valid", "sum"),
            exp_lcb_median=("exp_lcb", "median"),
            robust_top=("robust_score", "max"),
        )
        .reset_index()
        .sort_values("symbol")
    )
    lines.append("| symbol | combos | valid | exp_lcb_median | best_robust_score |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in sym_summary.to_dict(orient="records"):
        lines.append(
            f"| {row['symbol']} | {int(row['combos'])} | {int(row['valid'])} | "
            f"{float(row['exp_lcb_median']):.6f} | {float(row['robust_top']):.6f} |"
        )
    lines.append("")

    lines.append("## Stability Heatmap Table")
    heat = (
        leaderboard.pivot_table(
            index=["symbol", "timeframe"],
            columns="stability_classification",
            values="combo_key",
            aggfunc="count",
            fill_value=0,
        )
        .reset_index()
        .sort_values(["symbol", "timeframe"])
    )
    lines.append("| symbol | timeframe | STABLE | UNSTABLE | ZERO_TRADE | INVALID | INSUFFICIENT_DATA |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in heat.to_dict(orient="records"):
        lines.append(
            f"| {row.get('symbol','')} | {row.get('timeframe','')} | {int(row.get('STABLE',0))} | "
            f"{int(row.get('UNSTABLE',0))} | {int(row.get('ZERO_TRADE',0))} | {int(row.get('INVALID',0))} | {int(row.get('INSUFFICIENT_DATA',0))} |"
        )
    lines.append("")

    lines.append("## Cost Sensitivity Chart Data")
    lines.append("| symbol | timeframe | strategy | exit_type | cost_level | exp_lcb |")
    lines.append("| --- | --- | --- | --- | --- | ---: |")
    plot_rows = leaderboard[
        ["symbol", "timeframe", "strategy", "exit_type", "cost_level", "exp_lcb"]
    ].sort_values(["symbol", "timeframe", "strategy", "exit_type", "cost_level"])
    for row in plot_rows.to_dict(orient="records"):
        lines.append(
            f"| {row['symbol']} | {row['timeframe']} | {row['strategy']} | {row['exit_type']} | "
            f"{row['cost_level']} | {float(row['exp_lcb']):.6f} |"
        )
    lines.append("")

    mc_rows = leaderboard[leaderboard["MC_p_ruin"].notna()].copy()
    lines.append("## Monte Carlo Validation Summary (Top 20%)")
    if mc_rows.empty:
        lines.append("- no combinations selected for Monte Carlo")
    else:
        lines.append("| symbol | timeframe | strategy | exit_type | cost_level | MC_p_ruin | MC_p_return_negative | MC_maxDD_p95 | MC_expected_log_growth |")
        lines.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
        for row in mc_rows.sort_values("robust_score", ascending=False).to_dict(orient="records"):
            lines.append(
                f"| {row['symbol']} | {row['timeframe']} | {row['strategy']} | {row['exit_type']} | {row['cost_level']} | "
                f"{float(row['MC_p_ruin']):.6f} | {float(row['MC_p_return_negative']):.6f} | "
                f"{float(row['MC_maxDD_p95']):.6f} | {float(row['MC_expected_log_growth']):.6f} |"
            )
    lines.append("")
    lines.append("## Verdict")
    lines.append(f"- `{summary['verdict']}`")

    report_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _finite(value: Any, default: float = 0.0, clip: float | None = None) -> float:
    try:
        number = float(value)
    except Exception:
        number = float(default)
    if not np.isfinite(number):
        number = float(default)
    if clip is not None and number > float(clip):
        number = float(clip)
    return float(number)


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        number = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(number):
        return float(default)
    return float(number)


def _to_iso_utc(value: Any) -> str:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return ""
    return ts.isoformat()
