"""Stage-27.3 rolling discovery harness with no-loss compute optimizations."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.config import compute_config_hash, load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.data.snapshot import snapshot_metadata_from_config
from buffmini.stage10.evaluate import _build_features
from buffmini.stage26.conditional_eval import ConditionalEvalParams, evaluate_rulelets_conditionally
from buffmini.stage26.context import ContextParams, classify_context
from buffmini.stage26.rulelets import build_rulelet_library
from buffmini.stage27.coverage_gate import evaluate_coverage_gate
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-27 rolling discovery")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--windows", type=str, default="3m,6m")
    parser.add_argument("--step", type=str, default="1m", help="Rolling step in months, format '<int>m'")
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--timeframes", type=str, default="15m,30m,1h,2h,4h")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-insufficient-data", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def _csv(value: str, default: list[str]) -> list[str]:
    out = [item.strip() for item in str(value).split(",") if item.strip()]
    return out or list(default)


def _parse_month_values(raw: str, default: list[int]) -> list[int]:
    values = []
    for item in _csv(raw, [f"{v}m" for v in default]):
        text = str(item).strip().lower()
        if not text.endswith("m"):
            raise ValueError(f"Invalid month token: {item}")
        values.append(int(text[:-1]))
    return [int(v) for v in values if int(v) > 0]


def _cost_rows(cfg: dict[str, Any], names: list[str]) -> list[dict[str, Any]]:
    stage12 = dict((cfg.get("evaluation", {}) or {}).get("stage12", {}))
    scenarios = dict(stage12.get("cost_scenarios", {}))
    costs = dict(cfg.get("costs", {}))
    base = {
        "round_trip_cost_pct": float(costs.get("round_trip_cost_pct", 0.1)),
        "slippage_pct": float(costs.get("slippage_pct", 0.0005)),
        "cost_model_cfg": cfg.get("cost_model", {}),
        "stop_atr_multiple": 1.5,
        "take_profit_atr_multiple": 3.0,
        "max_hold_bars": 24,
    }
    out: list[dict[str, Any]] = []
    for name in names:
        level = str(name)
        row = dict(base)
        row["name"] = level
        if level == "realistic":
            out.append(row)
            continue
        s = dict(scenarios.get(level, {}))
        v2 = dict((cfg.get("cost_model", {}) or {}).get("v2", {}))
        if s:
            for key in ("slippage_bps_base", "slippage_bps_vol_mult", "spread_bps", "delay_bars"):
                if key in s and not bool(s.get("use_config_default", False)):
                    v2[key] = s[key]
        row["cost_model_cfg"] = {**dict(cfg.get("cost_model", {})), "v2": v2}
        out.append(row)
    return out


def _rolling_windows(ts: pd.Series, window_months: int, step_months: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if ts.empty:
        return []
    start = pd.Timestamp(ts.iloc[0]).tz_convert("UTC")
    end = pd.Timestamp(ts.iloc[-1]).tz_convert("UTC")
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor < end:
        win_end = cursor + pd.DateOffset(months=int(window_months))
        if win_end > end:
            break
        windows.append((cursor, win_end))
        cursor = cursor + pd.DateOffset(months=int(step_months))
    return windows


def _effects_hash(df: pd.DataFrame) -> str:
    if df.empty:
        return stable_hash([], length=16)
    keep = [
        "rulelet",
        "family",
        "context",
        "context_occurrences",
        "trades_in_context",
        "expectancy",
        "exp_lcb",
        "max_drawdown",
        "classification",
    ]
    use_cols = [c for c in keep if c in df.columns]
    view = df.loc[:, use_cols].sort_values(use_cols).reset_index(drop=True)
    return stable_hash(view.to_dict(orient="records"), length=16)


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage-27 Research Engine Report",
        "",
        "## What Changed",
        "- Added rolling-window discovery (3m/6m windows with 1m step).",
        "- Enabled batch score computation in conditional evaluation.",
        "- Reused feature/context frames across windows (compute-once per symbol/timeframe).",
        "",
        "## Runtime",
        f"- semantic_sample_naive_seconds: `{float(payload.get('semantic_sample_naive_seconds', 0.0)):.6f}`",
        f"- semantic_sample_batch_seconds: `{float(payload.get('semantic_sample_batch_seconds', 0.0)):.6f}`",
        f"- rolling_discovery_runtime_seconds: `{float(payload.get('runtime_seconds', 0.0)):.6f}`",
        "",
        "## Semantic Equivalence Guard",
        f"- semantic_hash_equal: `{bool(payload.get('semantic_hash_equal', False))}`",
        f"- naive_hash: `{payload.get('semantic_sample_naive_hash', '')}`",
        f"- batch_hash: `{payload.get('semantic_sample_batch_hash', '')}`",
        "",
        "## Cache Reuse",
        f"- feature_compute_calls: `{int(payload.get('feature_compute_calls', 0))}`",
        f"- rolling_window_evaluations: `{int(payload.get('rolling_window_evaluations', 0))}`",
        f"- feature_cache_hit_rate_estimate: `{float(payload.get('feature_cache_hit_rate_estimate', 0.0)):.6f}`",
        "",
        "## Run Output",
        f"- run_id: `{payload.get('run_id', '')}`",
        f"- rows: `{int(payload.get('rows', 0))}`",
        f"- used_symbols: `{payload.get('used_symbols', [])}`",
    ]
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    stage26_cfg = dict((cfg.get("evaluation", {}) or {}).get("stage26", {}))
    symbols = _csv(args.symbols, default=list(stage26_cfg.get("symbols", ["BTC/USDT", "ETH/USDT"])))
    timeframes = _csv(args.timeframes, default=list(stage26_cfg.get("timeframes", ["15m", "30m", "1h", "2h", "4h"])))
    window_months = _parse_month_values(args.windows, [3, 6])
    step_months = int(_parse_month_values(args.step, [1])[0])
    seed = int(args.seed)
    started = time.perf_counter()

    gate = evaluate_coverage_gate(
        config=cfg,
        symbols=symbols,
        timeframe=str(stage26_cfg.get("base_timeframe", "1m")),
        data_dir=args.data_dir,
        allow_insufficient_data=bool(args.allow_insufficient_data),
        auto_btc_fallback=True,
    )
    if not gate.can_run:
        print(f"coverage_gate_status: {gate.status}")
        print(f"coverage_years_by_symbol: {gate.coverage_years_by_symbol}")
        raise SystemExit(2)
    used_symbols = list(gate.used_symbols)

    ctx_cfg = dict(stage26_cfg.get("context", {}))
    ctx_params = ContextParams(
        rank_window=int(ctx_cfg.get("rank_window", 252)),
        vol_window=int(ctx_cfg.get("vol_window", 24)),
        bb_window=int(ctx_cfg.get("bb_window", 20)),
        volume_window=int(ctx_cfg.get("volume_window", 120)),
        chop_window=int(ctx_cfg.get("chop_window", 48)),
        trend_lookback=int(ctx_cfg.get("trend_lookback", 24)),
    )
    cond_cfg = dict(stage26_cfg.get("conditional_eval", {}))
    cond_params = ConditionalEvalParams(
        bootstrap_samples=int(cond_cfg.get("bootstrap_samples", 500)),
        seed=seed,
        min_occurrences=int(cond_cfg.get("min_occurrences", 30)),
        min_trades=int(cond_cfg.get("min_trades", 30)),
        rare_min_trades=int(cond_cfg.get("rare_min_trades", 10)),
        rolling_months=tuple(int(v) for v in cond_cfg.get("rolling_months", [3, 6, 12])),
    )
    cost_rows = _cost_rows(cfg, list(stage26_cfg.get("cost_levels", ["realistic", "high"])))
    rulelets = build_rulelet_library()

    features_cache: dict[tuple[str, str], pd.DataFrame] = {}
    feature_compute_calls = 0
    for tf in timeframes:
        loaded = _build_features(
            config=cfg,
            symbols=used_symbols,
            timeframe=str(tf),
            dry_run=bool(args.dry_run),
            seed=seed,
            data_dir=args.data_dir,
            derived_dir=args.derived_dir,
        )
        for symbol, frame in loaded.items():
            features_cache[(str(symbol), str(tf))] = classify_context(frame, params=ctx_params)
            feature_compute_calls += 1

    semantic_naive_seconds = 0.0
    semantic_batch_seconds = 0.0
    semantic_naive_hash = ""
    semantic_batch_hash = ""
    semantic_hash_equal = True
    sample_done = False

    rows: list[dict[str, Any]] = []
    rolling_window_evals = 0
    for (symbol, tf), frame in sorted(features_cache.items()):
        ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce").dropna()
        if ts.empty:
            continue
        for months in window_months:
            windows = _rolling_windows(ts, window_months=int(months), step_months=int(step_months))
            for window_index, (start_ts, end_ts) in enumerate(windows):
                mask = (pd.to_datetime(frame["timestamp"], utc=True, errors="coerce") >= start_ts) & (
                    pd.to_datetime(frame["timestamp"], utc=True, errors="coerce") < end_ts
                )
                sliced = frame.loc[mask].reset_index(drop=True)
                if sliced.shape[0] < 300:
                    continue
                rolling_window_evals += 1
                effects, _ = evaluate_rulelets_conditionally(
                    frame=sliced,
                    rulelets=rulelets,
                    symbol=str(symbol),
                    timeframe=str(tf),
                    seed=seed,
                    cost_levels=cost_rows,
                    params=cond_params,
                    batch_mode=True,
                )
                if not sample_done:
                    sample = sliced.tail(min(2500, sliced.shape[0])).reset_index(drop=True)
                    t0 = time.perf_counter()
                    naive_df, _ = evaluate_rulelets_conditionally(
                        frame=sample,
                        rulelets=rulelets,
                        symbol=str(symbol),
                        timeframe=str(tf),
                        seed=seed,
                        cost_levels=cost_rows,
                        params=cond_params,
                        batch_mode=False,
                    )
                    semantic_naive_seconds = float(time.perf_counter() - t0)
                    t1 = time.perf_counter()
                    batch_df, _ = evaluate_rulelets_conditionally(
                        frame=sample,
                        rulelets=rulelets,
                        symbol=str(symbol),
                        timeframe=str(tf),
                        seed=seed,
                        cost_levels=cost_rows,
                        params=cond_params,
                        batch_mode=True,
                    )
                    semantic_batch_seconds = float(time.perf_counter() - t1)
                    semantic_naive_hash = _effects_hash(naive_df)
                    semantic_batch_hash = _effects_hash(batch_df)
                    semantic_hash_equal = bool(semantic_naive_hash == semantic_batch_hash)
                    sample_done = True
                if effects.empty:
                    continue
                ranked = effects.sort_values(["exp_lcb", "expectancy", "trades_in_context"], ascending=[False, False, False])
                top = ranked.iloc[0].to_dict()
                rows.append(
                    {
                        "symbol": str(symbol),
                        "timeframe": str(tf),
                        "window_months": int(months),
                        "window_index": int(window_index),
                        "window_start": start_ts.isoformat(),
                        "window_end": end_ts.isoformat(),
                        "best_rulelet": str(top.get("rulelet", "")),
                        "best_context": str(top.get("context", "")),
                        "best_family": str(top.get("family", "")),
                        "best_exp_lcb": float(top.get("exp_lcb", 0.0)),
                        "best_expectancy": float(top.get("expectancy", 0.0)),
                        "best_trades_in_context": int(top.get("trades_in_context", 0)),
                        "best_classification": str(top.get("classification", "")),
                    }
                )

    run_id = (
        f"{utc_now_compact()}_"
        f"{stable_hash({'seed': seed, 'symbols': used_symbols, 'timeframes': timeframes, 'windows': window_months, 'step_months': step_months, 'dry_run': bool(args.dry_run), 'cfg': compute_config_hash(cfg)}, length=12)}"
        "_stage27_roll"
    )
    out_dir = args.runs_dir / run_id / "stage27"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_df = pd.DataFrame(rows)
    rows_df.to_csv(out_dir / "rolling_results.csv", index=False)

    theoretical_hits = max(0, int(rolling_window_evals - feature_compute_calls))
    hit_rate = float(theoretical_hits / max(1, rolling_window_evals))
    payload = {
        "stage": "27.3",
        "run_id": run_id,
        "seed": seed,
        "dry_run": bool(args.dry_run),
        "requested_symbols": list(gate.requested_symbols),
        "used_symbols": list(used_symbols),
        "disabled_symbols": list(gate.disabled_symbols),
        "coverage_years_by_symbol": dict(gate.coverage_years_by_symbol),
        "timeframes": list(timeframes),
        "window_months": list(window_months),
        "step_months": int(step_months),
        "rows": int(rows_df.shape[0]),
        "feature_compute_calls": int(feature_compute_calls),
        "rolling_window_evaluations": int(rolling_window_evals),
        "feature_cache_hit_rate_estimate": float(hit_rate),
        "semantic_hash_equal": bool(semantic_hash_equal),
        "semantic_sample_naive_hash": semantic_naive_hash,
        "semantic_sample_batch_hash": semantic_batch_hash,
        "semantic_sample_naive_seconds": float(semantic_naive_seconds),
        "semantic_sample_batch_seconds": float(semantic_batch_seconds),
        "runtime_seconds": float(time.perf_counter() - started),
        "config_hash": compute_config_hash(cfg),
        **snapshot_metadata_from_config(cfg),
    }
    (out_dir / "rolling_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_md = docs_dir / "stage27_research_engine_report.md"
    report_json = docs_dir / "stage27_research_engine_summary.json"
    report_md.write_text(_render_report(payload), encoding="utf-8")
    report_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    print(f"run_id: {run_id}")
    print(f"rolling_csv: {out_dir / 'rolling_results.csv'}")
    print(f"rolling_json: {out_dir / 'rolling_results.json'}")
    print(f"report_md: {report_md}")
    print(f"report_json: {report_json}")


if __name__ == "__main__":
    main()
