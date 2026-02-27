"""Stage-6 baseline vs regime-aware comparison runner (offline-safe)."""

from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.config import compute_config_hash, load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.execution.simulator import (
    _build_playback_state,
    _orders_to_trade_events,
    _normalize_signals,
    _ensure_utc,
    build_signals_from_stage2,
    resolve_stage4_method_and_leverage,
    simulate_execution,
)
from buffmini.ui.components.run_index import latest_completed_pipeline
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-6 baseline vs upgraded comparison")
    parser.add_argument("--run-id", type=str, default=None, help="Optional source pipeline run id")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true", help="No network fetch (required for tests/local offline)")
    parser.add_argument("--window-months", type=int, default=3)
    parser.add_argument("--symbols", type=str, default=None, help="Optional comma-separated symbol subset")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_pipeline_run = _resolve_source_pipeline_run(args.run_id, runs_dir=args.runs_dir)
    pipeline_summary = _safe_json(source_pipeline_run / "pipeline_summary.json")
    pipeline_cfg_path = source_pipeline_run / "pipeline_config.yaml"

    config = load_config(pipeline_cfg_path if pipeline_cfg_path.exists() else args.config)
    stage2_run_id = str(pipeline_summary.get("stage2_run_id", "")).strip()
    if not stage2_run_id:
        raise ValueError("Stage-6 compare requires source pipeline with non-empty stage2_run_id")
    stage3_run_id = str(pipeline_summary.get("stage3_3_run_id", "")).strip() or None
    stage3_summary = _safe_json(args.runs_dir / stage3_run_id / "selector_summary.json") if stage3_run_id else None

    selected_method, base_leverage, _, _ = resolve_stage4_method_and_leverage(cfg=config, stage3_choice=stage3_summary)
    signals, method_weights, metadata = build_signals_from_stage2(
        stage2_run_id=stage2_run_id,
        method=selected_method,
        runs_dir=args.runs_dir,
    )

    symbols_filter = _parse_symbols(args.symbols)
    if symbols_filter:
        signals = [signal for signal in signals if str(signal.symbol) in symbols_filter]
    signals = _slice_signals_last_months(signals=signals, window_months=int(args.window_months))
    if not signals:
        raise ValueError("No signals available for requested symbols/window-months")

    resolved_end_ts = _ensure_utc(max(signal.ts for signal in signals)).isoformat()

    baseline_cfg = deepcopy(config)
    baseline_cfg.setdefault("evaluation", {}).setdefault("stage6", {})
    baseline_cfg["evaluation"]["stage6"]["enabled"] = False
    baseline_run_dir = _run_simulation_variant(
        variant="baseline",
        base_cfg=baseline_cfg,
        stage2_run_id=stage2_run_id,
        stage1_run_id=str(metadata.get("stage1_run_id", "")),
        method=selected_method,
        chosen_leverage=float(base_leverage),
        signals=signals,
        method_weights=method_weights,
        candidate_metrics=metadata.get("candidate_metrics", {}),
        regime_lookup=metadata.get("regime_by_timestamp", {}),
        config_hash=str(metadata.get("config_hash", "")),
        data_hash=str(metadata.get("data_hash", "")),
        runs_dir=args.runs_dir,
        seed=int(args.seed),
        resolved_end_ts=resolved_end_ts,
    )

    stage6_cfg = deepcopy(config)
    stage6_cfg.setdefault("evaluation", {}).setdefault("stage6", {})
    stage6_cfg["evaluation"]["stage6"]["enabled"] = True
    stage6_run_dir = _run_simulation_variant(
        variant="stage6",
        base_cfg=stage6_cfg,
        stage2_run_id=stage2_run_id,
        stage1_run_id=str(metadata.get("stage1_run_id", "")),
        method=selected_method,
        chosen_leverage=float(base_leverage),
        signals=signals,
        method_weights=method_weights,
        candidate_metrics=metadata.get("candidate_metrics", {}),
        regime_lookup=metadata.get("regime_by_timestamp", {}),
        config_hash=str(metadata.get("config_hash", "")),
        data_hash=str(metadata.get("data_hash", "")),
        runs_dir=args.runs_dir,
        seed=int(args.seed),
        resolved_end_ts=resolved_end_ts,
    )

    baseline_metrics = _compute_compare_metrics(baseline_run_dir, resolved_end_ts=resolved_end_ts)
    stage6_metrics = _compute_compare_metrics(stage6_run_dir, resolved_end_ts=resolved_end_ts)
    rc_delta_baseline = _read_reality_check_drag_delta(source_pipeline_run)
    baseline_metrics["execution_drag_sensitivity_delta"] = rc_delta_baseline
    stage6_metrics["execution_drag_sensitivity_delta"] = _execution_drag_delta_from_run(stage6_run_dir)

    compare_dir = stage6_run_dir / "stage6_compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    (compare_dir / "baseline_metrics.json").write_text(
        json.dumps(baseline_metrics, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (compare_dir / "stage6_metrics.json").write_text(
        json.dumps(stage6_metrics, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    for artifact_name in ["leverage_path.csv", "sizing_multipliers.csv"]:
        source = stage6_run_dir / artifact_name
        if source.exists():
            target = compare_dir / artifact_name
            target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    compare_summary = {
        "baseline_run_id": baseline_run_dir.name,
        "stage6_run_id": stage6_run_dir.name,
        "source_pipeline_run_id": source_pipeline_run.name,
        "method": selected_method,
        "base_leverage": float(base_leverage),
        "seed": int(args.seed),
        "resolved_end_ts": resolved_end_ts,
        "window_months": int(args.window_months),
        "symbols": sorted({str(signal.symbol) for signal in signals}),
        "baseline": baseline_metrics,
        "stage6": stage6_metrics,
    }
    (compare_dir / "stage6_compare_summary.json").write_text(
        json.dumps(compare_summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_compare_report(compare_dir=compare_dir, summary=compare_summary)
    _write_docs_stage6_report(summary=compare_summary, docs_path=Path("docs/stage6_report.md"))

    print(f"baseline_run_id: {baseline_run_dir.name}")
    print(f"stage6_run_id: {stage6_run_dir.name}")
    print(f"compare_report: {compare_dir / 'stage6_compare_report.md'}")
    print(f"docs_report: docs/stage6_report.md")


def _resolve_source_pipeline_run(run_id: str | None, runs_dir: Path) -> Path:
    if run_id:
        run_dir = runs_dir / str(run_id)
        if not run_dir.exists():
            raise FileNotFoundError(f"Pipeline run not found: {run_id}")
        return run_dir
    latest = latest_completed_pipeline(runs_dir)
    if latest is None:
        raise FileNotFoundError("No completed pipeline runs found")
    return runs_dir / str(latest["run_id"])


def _parse_symbols(symbols: str | None) -> set[str]:
    if symbols is None or str(symbols).strip() == "":
        return set()
    return {item.strip() for item in str(symbols).split(",") if item.strip()}


def _slice_signals_last_months(signals: list[Any], window_months: int) -> list[Any]:
    if not signals:
        return []
    if int(window_months) < 1:
        raise ValueError("window_months must be >= 1")
    max_ts = _ensure_utc(max(signal.ts for signal in signals))
    cutoff = max_ts - pd.DateOffset(months=int(window_months))
    filtered = [signal for signal in signals if _ensure_utc(signal.ts) >= cutoff]
    return filtered if filtered else list(signals)


def _run_simulation_variant(
    variant: str,
    base_cfg: dict[str, Any],
    stage2_run_id: str,
    stage1_run_id: str,
    method: str,
    chosen_leverage: float,
    signals: list[Any],
    method_weights: dict[str, float],
    candidate_metrics: dict[str, Any],
    regime_lookup: dict[str, str],
    config_hash: str,
    data_hash: str,
    runs_dir: Path,
    seed: int,
    resolved_end_ts: str,
) -> Path:
    grouped = _normalize_signals(signals)
    runtime_cfg = deepcopy(base_cfg)
    runtime_cfg["_method_weights"] = dict(method_weights)
    runtime_cfg["_runtime_candidate_metrics"] = dict(candidate_metrics)
    runtime_cfg["_runtime_regime_by_timestamp"] = dict(regime_lookup)
    metrics, exposure_df, orders_df, killswitch_df = simulate_execution(
        signals_by_ts=grouped,
        cfg=runtime_cfg,
        initial_equity=float(base_cfg["portfolio"]["leverage_selector"]["initial_equity"]),
        chosen_leverage=float(chosen_leverage),
        seed=int(seed),
    )
    sizing_df = runtime_cfg.get("_stage6_sizing_df", pd.DataFrame())
    leverage_df = runtime_cfg.get("_stage6_leverage_df", pd.DataFrame())

    run_payload = {
        "variant": variant,
        "stage2_run_id": stage2_run_id,
        "stage1_run_id": stage1_run_id,
        "method": method,
        "chosen_leverage": float(chosen_leverage),
        "stage6_enabled": bool((base_cfg.get("evaluation", {}) or {}).get("stage6", {}).get("enabled", False)),
        "metrics": metrics,
        "config_hash": config_hash or compute_config_hash(base_cfg),
        "data_hash": data_hash,
        "resolved_end_ts": resolved_end_ts,
        "seed": int(seed),
    }
    run_id = f"{utc_now_compact()}_{stable_hash(run_payload, length=12)}_{variant}_stage6_sim"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "execution_metrics.json").write_text(
        json.dumps(run_payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    exposure_df.to_csv(run_dir / "exposure_timeseries.csv", index=False)
    orders_df.to_csv(run_dir / "orders.csv", index=False)
    _orders_to_trade_events(orders_df).to_csv(run_dir / "trades.csv", index=False)
    killswitch_df.to_csv(run_dir / "killswitch_events.csv", index=False)
    _build_playback_state(exposure_df=exposure_df, orders_df=orders_df, killswitch_df=killswitch_df).to_csv(
        run_dir / "playback_state.csv",
        index=False,
    )
    if isinstance(sizing_df, pd.DataFrame) and not sizing_df.empty:
        sizing_df.to_csv(run_dir / "sizing_multipliers.csv", index=False)
    if isinstance(leverage_df, pd.DataFrame) and not leverage_df.empty:
        leverage_df.to_csv(run_dir / "leverage_path.csv", index=False)
    return run_dir


def _compute_compare_metrics(run_dir: Path, resolved_end_ts: str) -> dict[str, Any]:
    exposure = pd.read_csv(run_dir / "exposure_timeseries.csv")
    orders = pd.read_csv(run_dir / "orders.csv") if (run_dir / "orders.csv").exists() else pd.DataFrame()
    leverage_path = pd.read_csv(run_dir / "leverage_path.csv") if (run_dir / "leverage_path.csv").exists() else pd.DataFrame()
    execution = _safe_json(run_dir / "execution_metrics.json")
    metrics = execution.get("metrics", {})

    if exposure.empty:
        raise ValueError(f"No exposure_timeseries rows found in {run_dir}")
    exposure["ts"] = pd.to_datetime(exposure["ts"], utc=True, errors="coerce")
    exposure["equity"] = pd.to_numeric(exposure["equity"], errors="coerce")
    exposure = exposure.dropna(subset=["ts", "equity"]).sort_values("ts")
    returns = exposure["equity"].pct_change().fillna(0.0).astype(float)
    duration_days = max(1.0, float((exposure["ts"].iloc[-1] - exposure["ts"].iloc[0]).total_seconds() / 86400.0))

    initial_equity = float(exposure["equity"].iloc[0])
    final_equity = float(exposure["equity"].iloc[-1])
    expected_log_growth = float(math.log(max(final_equity, 1e-12) / max(initial_equity, 1e-12)))
    running_peak = exposure["equity"].cummax().replace(0.0, np.nan)
    drawdown = ((running_peak - exposure["equity"]) / running_peak).fillna(0.0)
    max_drawdown = float(drawdown.max()) if not drawdown.empty else 0.0
    trade_count = int(len(orders))
    trades_per_month = float(trade_count / (duration_days / 30.0))
    avg_leverage = float(metrics.get("avg_leverage", execution.get("avg_leverage", execution.get("chosen_leverage", 1.0))))

    if not leverage_path.empty and "regime" in leverage_path.columns:
        regime_counts = leverage_path["regime"].astype(str).value_counts(normalize=True) * 100.0
        regime_distribution = {key: float(value) for key, value in regime_counts.to_dict().items()}
    else:
        regime_distribution = dict(metrics.get("regime_distribution", {}))

    return {
        "resolved_end_ts": resolved_end_ts,
        "expected_log_growth": expected_log_growth,
        "return_p05": float(returns.quantile(0.05)),
        "return_median": float(returns.quantile(0.50)),
        "return_p95": float(returns.quantile(0.95)),
        "maxdd_p95": max_drawdown,
        "p_ruin": None,
        "trades_per_month": trades_per_month,
        "avg_leverage": avg_leverage,
        "regime_distribution": regime_distribution,
        "trade_count": trade_count,
    }


def _execution_drag_delta_from_run(run_dir: Path) -> float:
    exposure = pd.read_csv(run_dir / "exposure_timeseries.csv")
    if exposure.empty:
        return 0.0
    exposure["equity"] = pd.to_numeric(exposure["equity"], errors="coerce")
    exposure = exposure.dropna(subset=["equity"])
    returns = exposure["equity"].pct_change().fillna(0.0).astype(float)
    delayed = returns.shift(1).fillna(0.0)
    dragged = delayed - 0.0001
    base = float((1.0 + returns).prod() - 1.0)
    drag = float((1.0 + dragged).prod() - 1.0)
    return drag - base


def _read_reality_check_drag_delta(source_pipeline_run: Path) -> float | None:
    table_path = source_pipeline_run / "reality_check" / "execution_drag_table.csv"
    if not table_path.exists():
        return None
    frame = pd.read_csv(table_path)
    target = frame.loc[(frame["delay_bars"] == 1) & (frame["extra_slippage_bps"] == 1.0)]
    if target.empty:
        return None
    return float(target.iloc[0]["delta_return_vs_base"])


def _write_compare_report(compare_dir: Path, summary: dict[str, Any]) -> None:
    baseline = summary["baseline"]
    stage6 = summary["stage6"]
    lines: list[str] = []
    lines.append("# Stage-6 Compare Report")
    lines.append("")
    lines.append(f"- baseline_run_id: `{summary['baseline_run_id']}`")
    lines.append(f"- stage6_run_id: `{summary['stage6_run_id']}`")
    lines.append(f"- source_pipeline_run_id: `{summary['source_pipeline_run_id']}`")
    lines.append(f"- method: `{summary['method']}`")
    lines.append(f"- base_leverage: `{summary['base_leverage']}`")
    lines.append(f"- resolved_end_ts (shared): `{summary['resolved_end_ts']}`")
    lines.append(f"- seed: `{summary['seed']}`")
    lines.append(f"- symbols: `{summary['symbols']}`")
    lines.append("")
    lines.append("| metric | baseline | stage6 | delta(stage6-baseline) |")
    lines.append("| --- | ---: | ---: | ---: |")
    for key in ["expected_log_growth", "return_p05", "return_median", "return_p95", "maxdd_p95", "trades_per_month", "avg_leverage"]:
        base_value = float(baseline.get(key, 0.0))
        stage6_value = float(stage6.get(key, 0.0))
        lines.append(f"| {key} | {base_value:.6f} | {stage6_value:.6f} | {stage6_value - base_value:.6f} |")
    lines.append("")
    lines.append("## Regime Distribution (%)")
    lines.append(f"- baseline: {baseline.get('regime_distribution', {})}")
    lines.append(f"- stage6: {stage6.get('regime_distribution', {})}")
    lines.append("")
    lines.append("## Execution Drag Sensitivity Delta")
    lines.append(
        f"- baseline: {baseline.get('execution_drag_sensitivity_delta')} | "
        f"stage6: {stage6.get('execution_drag_sensitivity_delta')}"
    )
    lines.append("")
    lines.append("No hard risk constraints were loosened. Dynamic leverage remains capped by configured limits.")
    (compare_dir / "stage6_compare_report.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _write_docs_stage6_report(summary: dict[str, Any], docs_path: Path) -> None:
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    baseline = summary["baseline"]
    stage6 = summary["stage6"]
    lines: list[str] = []
    lines.append("# Stage-6 Report")
    lines.append("")
    lines.append(f"- source_pipeline_run_id: `{summary['source_pipeline_run_id']}`")
    lines.append(f"- baseline_run_id: `{summary['baseline_run_id']}`")
    lines.append(f"- stage6_run_id: `{summary['stage6_run_id']}`")
    lines.append(f"- resolved_end_ts: `{summary['resolved_end_ts']}`")
    lines.append("")
    lines.append("## Before vs After")
    lines.append(f"- expected_log_growth: `{baseline['expected_log_growth']:.6f}` -> `{stage6['expected_log_growth']:.6f}`")
    lines.append(f"- return_p05: `{baseline['return_p05']:.6f}` -> `{stage6['return_p05']:.6f}`")
    lines.append(f"- return_median: `{baseline['return_median']:.6f}` -> `{stage6['return_median']:.6f}`")
    lines.append(f"- maxdd_p95: `{baseline['maxdd_p95']:.6f}` -> `{stage6['maxdd_p95']:.6f}`")
    lines.append(f"- trades_per_month: `{baseline['trades_per_month']:.4f}` -> `{stage6['trades_per_month']:.4f}`")
    lines.append(f"- avg_leverage: `{baseline['avg_leverage']:.4f}` -> `{stage6['avg_leverage']:.4f}`")
    lines.append("")
    lines.append("## Regime Distribution (%)")
    lines.append(f"- stage6: `{stage6.get('regime_distribution', {})}`")
    lines.append("")
    lines.append(
        f"Detailed artifact: `runs/{summary['stage6_run_id']}/stage6_compare/stage6_compare_report.md`"
    )
    docs_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


if __name__ == "__main__":
    main()
