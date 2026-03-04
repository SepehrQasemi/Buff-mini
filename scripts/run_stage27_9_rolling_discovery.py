"""Stage-27.9 expanded rolling contextual discovery runner."""

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
    parser = argparse.ArgumentParser(description="Run Stage-27.9 expanded rolling discovery")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--timeframes", type=str, default="15m,30m,1h,2h,4h")
    parser.add_argument("--windows", type=str, default="3m,6m")
    parser.add_argument("--step-size", type=str, default="1m")
    parser.add_argument("--max-windows", type=int, default=0, help="Safety cap, 0 means unlimited")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-insufficient-data", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def _csv(raw: str, default: list[str]) -> list[str]:
    out = [part.strip() for part in str(raw).split(",") if part.strip()]
    return out or list(default)


def _parse_month_token(raw: str) -> int:
    text = str(raw).strip().lower()
    if not text.endswith("m"):
        raise ValueError(f"Invalid month token: {raw}")
    return int(text[:-1])


def _parse_month_list(raw: str, default: list[int]) -> list[int]:
    tokens = _csv(raw, [f"{item}m" for item in default])
    out = [_parse_month_token(token) for token in tokens]
    return [int(value) for value in out if int(value) > 0]


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
    rows: list[dict[str, Any]] = []
    for name in names:
        level = str(name)
        row = dict(base)
        row["name"] = level
        if level == "realistic":
            rows.append(row)
            continue
        scenario = dict(scenarios.get(level, {}))
        v2 = dict((cfg.get("cost_model", {}) or {}).get("v2", {}))
        if scenario:
            for key in ("slippage_bps_base", "slippage_bps_vol_mult", "spread_bps", "delay_bars"):
                if key in scenario and not bool(scenario.get("use_config_default", False)):
                    v2[key] = scenario[key]
        row["cost_model_cfg"] = {**dict(cfg.get("cost_model", {})), "v2": v2}
        rows.append(row)
    return rows


def _rolling_windows(ts: pd.Series, *, window_months: int, step_months: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if ts.empty:
        return []
    start = pd.Timestamp(ts.iloc[0]).tz_convert("UTC")
    end = pd.Timestamp(ts.iloc[-1]).tz_convert("UTC")
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor < end:
        right = cursor + pd.DateOffset(months=int(window_months))
        if right > end:
            break
        out.append((cursor, right))
        cursor = cursor + pd.DateOffset(months=int(step_months))
    return out


def _render_report(payload: dict[str, Any]) -> str:
    counts = payload.get("window_counts", {})
    lines = [
        "# Stage-27.9 Rolling Discovery Summary",
        "",
        "## Config",
        f"- run_id: `{payload.get('run_id', '')}`",
        f"- seed: `{int(payload.get('seed', 42))}`",
        f"- dry_run: `{bool(payload.get('dry_run', False))}`",
        f"- symbols: `{payload.get('used_symbols', [])}`",
        f"- timeframes: `{payload.get('timeframes', [])}`",
        f"- windows: `{payload.get('window_months', [])}`",
        f"- step_months: `{int(payload.get('step_months', 1))}`",
        "",
        "## Window Coverage",
    ]
    for key in sorted(counts.keys(), key=lambda value: int(value)):
        row = dict(counts.get(key, {}))
        lines.append(
            f"- {key}m: generated={int(row.get('generated', 0))}, evaluated={int(row.get('evaluated', 0))}"
        )
    lines.extend(
        [
            "",
            "## Metrics",
            f"- rows: `{int(payload.get('rows', 0))}`",
            f"- positive_exp_lcb_rows: `{int(payload.get('positive_exp_lcb_rows', 0))}`",
            f"- runtime_seconds: `{float(payload.get('runtime_seconds', 0.0)):.6f}`",
            "",
            "## Top Contextual Rows",
        ]
    )
    top_rows = payload.get("top_rows", [])
    if top_rows:
        lines.append("| symbol | timeframe | window_months | context | rulelet | trade_count | exp | exp_lcb | pf |")
        lines.append("|---|---:|---:|---|---|---:|---:|---:|---:|")
        for row in top_rows:
            lines.append(
                "| {symbol} | {timeframe} | {window_months} | {context} | {rulelet} | {trade_count} | {exp:.6f} | {exp_lcb:.6f} | {pf:.6f} |".format(
                    symbol=row.get("symbol", ""),
                    timeframe=row.get("timeframe", ""),
                    window_months=int(row.get("window_months", 0)),
                    context=row.get("context", ""),
                    rulelet=row.get("rulelet", ""),
                    trade_count=int(row.get("trade_count", 0)),
                    exp=float(row.get("exp", 0.0)),
                    exp_lcb=float(row.get("exp_lcb", 0.0)),
                    pf=float(row.get("pf", 0.0)),
                )
            )
    else:
        lines.append("- No rows produced.")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    stage26 = dict((cfg.get("evaluation", {}) or {}).get("stage26", {}))
    seed = int(args.seed)

    symbols = _csv(args.symbols, default=list(stage26.get("symbols", ["BTC/USDT", "ETH/USDT"])))
    timeframes = _csv(args.timeframes, default=list(stage26.get("timeframes", ["15m", "30m", "1h", "2h", "4h"])))
    window_months = _parse_month_list(args.windows, [3, 6])
    step_months = int(_parse_month_list(args.step_size, [1])[0])
    max_windows = int(max(0, args.max_windows))

    started = time.perf_counter()

    gate = evaluate_coverage_gate(
        config=cfg,
        symbols=symbols,
        timeframe=str(stage26.get("base_timeframe", "1m")),
        data_dir=args.data_dir,
        allow_insufficient_data=bool(args.allow_insufficient_data),
        auto_btc_fallback=True,
    )
    if not gate.can_run:
        print(f"coverage_gate_status: {gate.status}")
        print(f"coverage_years_by_symbol: {gate.coverage_years_by_symbol}")
        raise SystemExit(2)

    used_symbols = list(gate.used_symbols)
    ctx_cfg = dict(stage26.get("context", {}))
    cond_cfg = dict(stage26.get("conditional_eval", {}))
    ctx_params = ContextParams(
        rank_window=int(ctx_cfg.get("rank_window", 252)),
        vol_window=int(ctx_cfg.get("vol_window", 24)),
        bb_window=int(ctx_cfg.get("bb_window", 20)),
        volume_window=int(ctx_cfg.get("volume_window", 120)),
        chop_window=int(ctx_cfg.get("chop_window", 48)),
        trend_lookback=int(ctx_cfg.get("trend_lookback", 24)),
    )
    cond_params = ConditionalEvalParams(
        bootstrap_samples=int(cond_cfg.get("bootstrap_samples", 500)),
        seed=seed,
        min_occurrences=int(cond_cfg.get("min_occurrences", 30)),
        min_trades=int(cond_cfg.get("min_trades", 30)),
        rare_min_trades=int(cond_cfg.get("rare_min_trades", 10)),
        rolling_months=tuple(int(value) for value in cond_cfg.get("rolling_months", [3, 6, 12])),
    )
    cost_rows = _cost_rows(cfg, list(stage26.get("cost_levels", ["realistic", "high"])))
    rulelets = build_rulelet_library()

    features_cache: dict[tuple[str, str], pd.DataFrame] = {}
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

    window_counts = {int(months): {"generated": 0, "evaluated": 0} for months in window_months}
    rows: list[dict[str, Any]] = []

    for (symbol, tf), frame in sorted(features_cache.items()):
        ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce").dropna()
        if ts.empty:
            continue
        ts_full = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        for months in window_months:
            windows = _rolling_windows(ts, window_months=int(months), step_months=int(step_months))
            window_counts[int(months)]["generated"] += int(len(windows))
            evaluated = 0
            for window_index, (start_ts, end_ts) in enumerate(windows):
                if max_windows > 0 and evaluated >= max_windows:
                    break
                mask = (ts_full >= start_ts) & (ts_full < end_ts)
                sliced = frame.loc[mask].reset_index(drop=True)
                if sliced.shape[0] < 300:
                    continue
                evaluated += 1
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
                if effects.empty:
                    continue
                for _, effect in effects.iterrows():
                    cost_items = effect.get("cost_rows", [])
                    cost_pf: list[float] = []
                    if isinstance(cost_items, list):
                        for item in cost_items:
                            if isinstance(item, dict):
                                try:
                                    cost_pf.append(float(item.get("profit_factor", item.get("pf", 0.0))))
                                except (TypeError, ValueError):
                                    cost_pf.append(0.0)
                    pf_value = float(pd.Series(cost_pf, dtype=float).median()) if cost_pf else float(effect.get("profit_factor", 0.0))
                    rows.append(
                        {
                            "symbol": str(symbol),
                            "timeframe": str(tf),
                            "window_months": int(months),
                            "window_index": int(window_index),
                            "window_start": start_ts.isoformat(),
                            "window_end": end_ts.isoformat(),
                            "context": str(effect.get("context", "")),
                            "rulelet": str(effect.get("rulelet", "")),
                            "trade_count": int(effect.get("trades_in_context", 0)),
                            "exp": float(effect.get("expectancy", 0.0)),
                            "exp_lcb": float(effect.get("exp_lcb", 0.0)),
                            "pf": float(pf_value),
                            "classification": str(effect.get("classification", "")),
                        }
                    )
            window_counts[int(months)]["evaluated"] += int(evaluated)

    config_hash = compute_config_hash(cfg)
    run_id = (
        f"{utc_now_compact()}_"
        f"{stable_hash({'seed': seed, 'symbols': used_symbols, 'timeframes': timeframes, 'windows': window_months, 'step_months': step_months, 'dry_run': bool(args.dry_run), 'cfg': config_hash}, length=12)}"
        "_stage27_9_roll"
    )
    out_dir = args.runs_dir / run_id / "stage27_9"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_df = pd.DataFrame(rows)
    rows_df.to_csv(out_dir / "rolling_results.csv", index=False)

    top_rows = (
        rows_df.sort_values(["exp_lcb", "exp", "trade_count"], ascending=[False, False, False]).head(15).to_dict(orient="records")
        if not rows_df.empty
        else []
    )
    payload = {
        "stage": "27.9.4",
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
        "window_counts": {str(key): value for key, value in sorted(window_counts.items())},
        "rows": int(rows_df.shape[0]),
        "positive_exp_lcb_rows": int((pd.to_numeric(rows_df.get("exp_lcb", 0.0), errors="coerce").fillna(0.0) > 0.0).sum()) if not rows_df.empty else 0,
        "top_rows": top_rows,
        "runtime_seconds": float(time.perf_counter() - started),
        "config_hash": config_hash,
        **snapshot_metadata_from_config(cfg),
    }
    (out_dir / "rolling_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    docs_md = docs_dir / "stage27_9_rolling_summary.md"
    docs_json = docs_dir / "stage27_9_rolling_summary.json"
    docs_md.write_text(_render_report(payload), encoding="utf-8")
    docs_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    print(f"run_id: {run_id}")
    print(f"rolling_csv: {out_dir / 'rolling_results.csv'}")
    print(f"rolling_json: {out_dir / 'rolling_results.json'}")
    print(f"docs_md: {docs_md}")
    print(f"docs_json: {docs_json}")


if __name__ == "__main__":
    main()
