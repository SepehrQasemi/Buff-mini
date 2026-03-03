"""Stage-26 orchestrator: coverage + contexts + conditional policy + global baseline."""

from __future__ import annotations

import argparse
import json
import math
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.backtest.engine import run_backtest
from buffmini.config import compute_config_hash, load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.stage10.evaluate import _build_features
from buffmini.stage26.conditional_eval import ConditionalEvalParams, bootstrap_lcb, evaluate_rulelets_conditionally
from buffmini.stage26.context import ContextParams, classify_context
from buffmini.stage26.policy import build_conditional_policy
from buffmini.stage26.replay import replay_conditional_policy
from buffmini.stage26.rulelets import build_rulelet_library
from buffmini.stage27.coverage_gate import evaluate_coverage_gate
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact
from buffmini.validation.walkforward_v2 import build_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-26 full pipeline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--timeframes", type=str, default="")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--allow-insufficient-data", action="store_true")
    return parser.parse_args()


def _csv(value: str, default: list[str]) -> list[str]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    return items or list(default)


def _frame_data_hash(frame: pd.DataFrame) -> str:
    cols = [c for c in ("timestamp", "open", "high", "low", "close", "volume") if c in frame.columns]
    if not cols:
        return stable_hash({"rows": int(frame.shape[0])}, length=16)
    return stable_hash(frame.loc[:, cols].to_dict(orient="list"), length=16)


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
            for k in ("slippage_bps_base", "slippage_bps_vol_mult", "spread_bps", "delay_bars"):
                if k in s and not bool(s.get("use_config_default", False)):
                    v2[k] = s[k]
        row["cost_model_cfg"] = {**dict(cfg.get("cost_model", {})), "v2": v2}
        out.append(row)
    return out


def _wf_mc_metrics(frame: pd.DataFrame, signal: pd.Series, trade_pnls: np.ndarray, trade_count: float, cfg: dict[str, Any]) -> dict[str, Any]:
    wf_cfg = (((cfg.get("evaluation", {}) or {}).get("stage8", {})) or {}).get("walkforward_v2", {})
    ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce").dropna()
    if ts.empty:
        return {
            "walkforward_executed_true": False,
            "usable_windows": 0,
            "expected_windows": 0,
            "mc_triggered": False,
        }
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
    usable = 0
    evaluated = 0
    signal_series = pd.to_numeric(signal, errors="coerce").fillna(0).astype(int)
    for w in windows:
        mask = (ts >= w.forward_start) & (ts < w.forward_end)
        if int(mask.sum()) <= 0:
            continue
        evaluated += 1
        f_sig = signal_series.loc[mask]
        trades_proxy = int((f_sig != 0).sum())
        exposure = float((f_sig != 0).mean()) if len(f_sig) else 0.0
        if trades_proxy >= min_trades and exposure >= min_exposure:
            usable += 1
    mc_min = int(max(10, (((cfg.get("evaluation", {}) or {}).get("stage12", {}) or {}).get("monte_carlo", {})).get("min_trades", 10)))
    arr = np.asarray(trade_pnls, dtype=float)
    mc_triggered = bool(int(trade_count) >= mc_min and arr.size >= 2 and np.isfinite(arr).all())
    return {
        "walkforward_executed_true": bool(evaluated > 0),
        "usable_windows": int(usable),
        "expected_windows": int(len(windows)),
        "mc_triggered": bool(mc_triggered),
    }


def _run_global_baseline(*, frame: pd.DataFrame, symbol: str, timeframe: str, seed: int, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rulelets = build_rulelet_library()
    score_bank: list[pd.Series] = []
    best_row: dict[str, Any] | None = None
    best_name = ""
    best_lcb = -1e18
    for name, rulelet in rulelets.items():
        score = pd.to_numeric(rulelet.compute_score(frame), errors="coerce").fillna(0.0)
        signal = pd.Series(0, index=frame.index, dtype=int)
        signal.loc[score >= float(rulelet.threshold)] = 1
        signal.loc[score <= -float(rulelet.threshold)] = -1
        signal = signal.shift(1).fillna(0).astype(int)
        result = run_backtest(
            frame=frame.assign(signal=signal),
            strategy_name=f"Stage26Global::{name}",
            symbol=symbol,
            signal_col="signal",
            stop_atr_multiple=1.5,
            take_profit_atr_multiple=3.0,
            max_hold_bars=24,
            round_trip_cost_pct=float(cfg.get("costs", {}).get("round_trip_cost_pct", 0.1)),
            slippage_pct=float(cfg.get("costs", {}).get("slippage_pct", 0.0005)),
            exit_mode="fixed_atr",
            cost_model_cfg=cfg.get("cost_model", {}),
        )
        pnls = pd.to_numeric(result.trades.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        exp_lcb = bootstrap_lcb(values=pnls, seed=int(seed), samples=500)
        wf_mc = _wf_mc_metrics(frame, signal, pnls, float(result.metrics.get("trade_count", 0.0)), cfg)
        trade_count = float(result.metrics.get("trade_count", 0.0))
        months = _months(frame)
        row = {
            "symbol": symbol,
            "timeframe": timeframe,
            "variant": "global_single",
            "rulelet": name,
            "trade_count": trade_count,
            "tpm": float(trade_count / max(1e-9, months)),
            "exposure_ratio": float((signal != 0).mean()),
            "PF_raw": float(result.metrics.get("profit_factor", 0.0)),
            "PF_clipped": float(np.clip(float(result.metrics.get("profit_factor", 0.0)), 0.0, 10.0)),
            "expectancy": float(result.metrics.get("expectancy", 0.0)),
            "exp_lcb": float(exp_lcb),
            "maxDD": float(result.metrics.get("max_drawdown", 0.0)),
            **wf_mc,
        }
        rows.append(row)
        score_bank.append(score)
        if float(row["exp_lcb"]) > best_lcb:
            best_lcb = float(row["exp_lcb"])
            best_name = str(name)
            best_row = row
    if score_bank:
        ensemble_score = pd.concat(score_bank, axis=1).mean(axis=1).fillna(0.0)
        signal = pd.Series(0, index=frame.index, dtype=int)
        signal.loc[ensemble_score >= 0.25] = 1
        signal.loc[ensemble_score <= -0.25] = -1
        signal = signal.shift(1).fillna(0).astype(int)
        result = run_backtest(
            frame=frame.assign(signal=signal),
            strategy_name="Stage26Global::ensemble",
            symbol=symbol,
            signal_col="signal",
            stop_atr_multiple=1.5,
            take_profit_atr_multiple=3.0,
            max_hold_bars=24,
            round_trip_cost_pct=float(cfg.get("costs", {}).get("round_trip_cost_pct", 0.1)),
            slippage_pct=float(cfg.get("costs", {}).get("slippage_pct", 0.0005)),
            exit_mode="fixed_atr",
            cost_model_cfg=cfg.get("cost_model", {}),
        )
        pnls = pd.to_numeric(result.trades.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        wf_mc = _wf_mc_metrics(frame, signal, pnls, float(result.metrics.get("trade_count", 0.0)), cfg)
        trade_count = float(result.metrics.get("trade_count", 0.0))
        months = _months(frame)
        rows.append(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "variant": "global_ensemble",
                "rulelet": "global_ensemble",
                "trade_count": trade_count,
                "tpm": float(trade_count / max(1e-9, months)),
                "exposure_ratio": float((signal != 0).mean()),
                "PF_raw": float(result.metrics.get("profit_factor", 0.0)),
                "PF_clipped": float(np.clip(float(result.metrics.get("profit_factor", 0.0)), 0.0, 10.0)),
                "expectancy": float(result.metrics.get("expectancy", 0.0)),
                "exp_lcb": float(bootstrap_lcb(values=pnls, seed=int(seed + 1), samples=500)),
                "maxDD": float(result.metrics.get("max_drawdown", 0.0)),
                **wf_mc,
            }
        )
    if best_row is not None:
        rows.append({**best_row, "variant": "global_best_single", "rulelet": best_name})
    return rows


def _months(frame: pd.DataFrame) -> float:
    ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce").dropna()
    if ts.empty:
        return 0.0
    return max(float((ts.iloc[-1] - ts.iloc[0]).total_seconds()) / 86400.0 / 30.0, 1e-9)


def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "trade_count": 0.0,
            "tpm": 0.0,
            "PF_clipped": 0.0,
            "expectancy": 0.0,
            "exp_lcb": 0.0,
            "maxDD": 0.0,
            "zero_trade_pct": 100.0,
            "invalid_pct": 100.0,
            "walkforward_executed_true_pct": 0.0,
            "usable_windows_count": 0,
            "mc_trigger_rate": 0.0,
        }
    df = pd.DataFrame(rows)
    for col in ("trade_count", "tpm", "PF_clipped", "expectancy", "exp_lcb", "maxDD"):
        df[col] = pd.to_numeric(df.get(col, 0.0), errors="coerce").fillna(0.0)
    invalid = (~np.isfinite(df["exp_lcb"])) | (~np.isfinite(df["PF_clipped"]))
    wf_raw = df["walkforward_executed_true"] if "walkforward_executed_true" in df.columns else pd.Series(False, index=df.index)
    mc_raw = df["mc_triggered"] if "mc_triggered" in df.columns else pd.Series(False, index=df.index)
    wf_true = pd.Series(wf_raw, index=df.index).fillna(False).astype(bool)
    mc_true = pd.Series(mc_raw, index=df.index).fillna(False).astype(bool)
    usable_raw = df["usable_windows"] if "usable_windows" in df.columns else pd.Series(0, index=df.index)
    out = {
        "trade_count": float(df["trade_count"].sum()),
        "tpm": float(df["tpm"].mean()),
        "PF_clipped": float(df["PF_clipped"].mean()),
        "expectancy": float(df["expectancy"].mean()),
        "exp_lcb": float(df["exp_lcb"].mean()),
        "maxDD": float(df["maxDD"].mean()),
        "zero_trade_pct": float((df["trade_count"] <= 0.0).mean() * 100.0),
        "invalid_pct": float(invalid.mean() * 100.0),
        "walkforward_executed_true_pct": float(wf_true.mean() * 100.0),
        "usable_windows_count": int(pd.to_numeric(pd.Series(usable_raw, index=df.index), errors="coerce").fillna(0).sum()),
        "mc_trigger_rate": float(mc_true.mean() * 100.0),
    }
    return out


def _classify_verdict(conditional_live: dict[str, Any], global_metrics: dict[str, Any], coverage_ok: bool) -> str:
    if not coverage_ok:
        return "INSUFFICIENT_DATA"
    if float(conditional_live.get("exp_lcb", 0.0)) <= 0.0 and float(global_metrics.get("exp_lcb", 0.0)) <= 0.0:
        return "NO_EDGE"
    if float(conditional_live.get("exp_lcb", 0.0)) > 0.0 and float(conditional_live.get("walkforward_executed_true_pct", 0.0)) >= 50.0:
        return "ROBUST_EDGE"
    if float(conditional_live.get("exp_lcb", 0.0)) > 0.0:
        return "WEAK_EDGE"
    return "NO_EDGE"


def _write_report_md(path: Path, payload: dict[str, Any]) -> None:
    comp = payload["comparison_delta"]
    lines = [
        "# Stage-26 Report",
        "",
        f"- head_commit: `{payload['head_commit']}`",
        f"- run_id: `{payload['run_id']}`",
        f"- seed: `{payload['seed']}`",
        f"- dry_run: `{payload['dry_run']}`",
        "",
        "## Data Coverage (4y)",
        f"- coverage_ok_all_symbols: `{payload['coverage_ok_all_symbols']}`",
        f"- required_years: `{payload['required_years']}`",
        "",
        "## Metrics",
        "| mode | trade_count | tpm | exp_lcb | maxDD | wf_executed_pct | mc_trigger_rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| conditional_research | {payload['conditional_policy_metrics_research']['trade_count']:.2f} | {payload['conditional_policy_metrics_research']['tpm']:.4f} | {payload['conditional_policy_metrics_research']['exp_lcb']:.6f} | {payload['conditional_policy_metrics_research']['maxDD']:.6f} | {payload['conditional_policy_metrics_research']['walkforward_executed_true_pct']:.2f} | {payload['conditional_policy_metrics_research']['mc_trigger_rate']:.2f} |",
        f"| conditional_live | {payload['conditional_policy_metrics_live']['trade_count']:.2f} | {payload['conditional_policy_metrics_live']['tpm']:.4f} | {payload['conditional_policy_metrics_live']['exp_lcb']:.6f} | {payload['conditional_policy_metrics_live']['maxDD']:.6f} | {payload['conditional_policy_metrics_live']['walkforward_executed_true_pct']:.2f} | {payload['conditional_policy_metrics_live']['mc_trigger_rate']:.2f} |",
        f"| global_baseline | {payload['global_baseline_metrics']['trade_count']:.2f} | {payload['global_baseline_metrics']['tpm']:.4f} | {payload['global_baseline_metrics']['exp_lcb']:.6f} | {payload['global_baseline_metrics']['maxDD']:.6f} | {payload['global_baseline_metrics']['walkforward_executed_true_pct']:.2f} | {payload['global_baseline_metrics']['mc_trigger_rate']:.2f} |",
        "",
        "## Conditional vs Global Delta",
        f"- delta_exp_lcb: `{float(comp['delta_exp_lcb']):.6f}`",
        f"- delta_maxDD: `{float(comp['delta_maxDD']):.6f}`",
        f"- delta_trade_count: `{float(comp['delta_trade_count']):.6f}`",
        "",
        "## Shadow Live",
        f"- shadow_live_reject_rate: `{float(payload['shadow_live_reject_rate']):.6f}`",
        f"- shadow_live_top_reasons: `{payload.get('shadow_live_top_reasons', {})}`",
        "",
        "## Verdict",
        f"- `{payload['verdict']}`",
        f"- next_bottleneck: `{payload['next_bottleneck']}`",
    ]
    path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.perf_counter()
    cfg = load_config(args.config)
    stage26 = dict((cfg.get("evaluation", {}) or {}).get("stage26", {}))
    # Stage-26 evaluates derived operational timeframes from base 1m cache.
    cfg.setdefault("universe", {})["base_timeframe"] = str(stage26.get("base_timeframe", "1m"))
    cfg.setdefault("data", {})["resample_source"] = "base"
    symbols = _csv(args.symbols, default=list(stage26.get("symbols", ["BTC/USDT", "ETH/USDT"])))
    timeframes = _csv(args.timeframes, default=list(stage26.get("timeframes", ["15m", "30m", "1h", "2h", "4h"])))
    seed = int(args.seed)
    dry_run = bool(args.dry_run)
    warnings: list[str] = []

    coverage_gate = evaluate_coverage_gate(
        config=cfg,
        symbols=symbols,
        timeframe=str(stage26.get("base_timeframe", "1m")),
        data_dir=args.data_dir,
        allow_insufficient_data=bool(args.allow_insufficient_data),
        auto_btc_fallback=True,
    )
    if not coverage_gate.can_run:
        print(f"coverage_gate_status: {coverage_gate.status}")
        print(f"coverage_years_by_symbol: {coverage_gate.coverage_years_by_symbol}")
        raise SystemExit(2)
    symbols = list(coverage_gate.used_symbols)
    coverage_rows = list(coverage_gate.rows)
    if coverage_gate.disabled_symbols:
        warnings.append(f"auto_disabled_symbols:{','.join(coverage_gate.disabled_symbols)}")
    if coverage_gate.notes:
        warnings.extend([str(note) for note in coverage_gate.notes])

    required_years = float(stage26.get("required_years", 4))
    coverage_ok = all(
        bool(row.get("exists", False)) and float(row.get("coverage_years", 0.0)) >= required_years
        for row in coverage_rows
        if str(row.get("symbol", "")) in symbols
    )

    ctx_cfg = dict(stage26.get("context", {}))
    ctx_params = ContextParams(
        rank_window=int(ctx_cfg.get("rank_window", 252)),
        vol_window=int(ctx_cfg.get("vol_window", 24)),
        bb_window=int(ctx_cfg.get("bb_window", 20)),
        volume_window=int(ctx_cfg.get("volume_window", 120)),
        chop_window=int(ctx_cfg.get("chop_window", 48)),
        trend_lookback=int(ctx_cfg.get("trend_lookback", 24)),
    )
    cond_cfg = dict(stage26.get("conditional_eval", {}))
    cond_params = ConditionalEvalParams(
        bootstrap_samples=int(cond_cfg.get("bootstrap_samples", 500)),
        seed=seed,
        min_occurrences=int(cond_cfg.get("min_occurrences", 30)),
        min_trades=int(cond_cfg.get("min_trades", 30)),
        rare_min_trades=int(cond_cfg.get("rare_min_trades", 10)),
        rolling_months=tuple(int(v) for v in cond_cfg.get("rolling_months", [3, 6, 12])),
    )
    policy_cfg = dict(stage26.get("policy", {}))
    cost_rows = _cost_rows(cfg, list(stage26.get("cost_levels", ["realistic", "high"])))
    rulelets = build_rulelet_library()

    frames_by_symbol_tf: dict[tuple[str, str], pd.DataFrame] = {}
    effects_rows: list[dict[str, Any]] = []
    data_hashes: dict[str, str] = {}
    resolved_ends: list[pd.Timestamp] = []
    context_distribution_rows: list[dict[str, Any]] = []
    for tf in timeframes:
        try:
            features_by_symbol = _build_features(
                config=cfg,
                symbols=symbols,
                timeframe=str(tf),
                dry_run=dry_run,
                seed=seed,
                data_dir=args.data_dir,
                derived_dir=args.derived_dir,
            )
        except FileNotFoundError as exc:
            warnings.append(f"missing_data:{tf}:{exc}")
            features_by_symbol = {}
        for symbol, frame in features_by_symbol.items():
            with_ctx = classify_context(frame, params=ctx_params)
            frames_by_symbol_tf[(str(symbol), str(tf))] = with_ctx
            data_hashes[f"{symbol}|{tf}"] = _frame_data_hash(with_ctx)
            ts = pd.to_datetime(with_ctx.get("timestamp"), utc=True, errors="coerce").dropna()
            if not ts.empty:
                resolved_ends.append(ts.max())
            dist = with_ctx["ctx_state"].astype(str).value_counts(normalize=True)
            for ctx, pct in dist.items():
                context_distribution_rows.append(
                    {
                        "symbol": str(symbol),
                        "timeframe": str(tf),
                        "context": str(ctx),
                        "pct": float(pct * 100.0),
                    }
                )
            effects_df, _ = evaluate_rulelets_conditionally(
                frame=with_ctx,
                rulelets=rulelets,
                symbol=str(symbol),
                timeframe=str(tf),
                seed=seed,
                cost_levels=cost_rows,
                params=cond_params,
            )
            if not effects_df.empty:
                effects_rows.extend(effects_df.to_dict(orient="records"))

    effects = pd.DataFrame(effects_rows)
    policy = build_conditional_policy(
        effects=effects,
        min_occurrences_per_context=int(policy_cfg.get("min_occurrences_per_context", 30)),
        min_trades_in_context=int(policy_cfg.get("min_trades_in_context", 30)),
        top_k=int(policy_cfg.get("top_k", 2)),
        w_min=float(policy_cfg.get("w_min", 0.05)),
        w_max=float(policy_cfg.get("w_max", 0.80)),
    )

    cfg_research = deepcopy(cfg)
    cfg_research.setdefault("evaluation", {}).setdefault("constraints", {})["mode"] = str(stage26.get("constraints_mode_discovery", "research"))
    research = replay_conditional_policy(
        frames_by_symbol_tf=frames_by_symbol_tf,
        effects=effects,
        config=cfg_research,
        seed=seed,
        mode="research",
    )
    cfg_live = deepcopy(cfg)
    cfg_live.setdefault("evaluation", {}).setdefault("constraints", {})["mode"] = str(stage26.get("constraints_mode_live", "live"))
    live = replay_conditional_policy(
        frames_by_symbol_tf=frames_by_symbol_tf,
        effects=effects,
        config=cfg_live,
        seed=seed,
        mode="live",
    )

    global_rows: list[dict[str, Any]] = []
    for (symbol, tf), frame in sorted(frames_by_symbol_tf.items()):
        global_rows.extend(_run_global_baseline(frame=frame, symbol=symbol, timeframe=tf, seed=seed, cfg=cfg))
    global_df = pd.DataFrame(global_rows)
    global_best = global_df.loc[global_df.get("variant", "").astype(str).eq("global_best_single")].copy()

    research_df = pd.DataFrame(research.metrics_rows)
    live_df = pd.DataFrame(live.metrics_rows)
    shadow_df = pd.DataFrame(research.shadow_live_rows)
    policy_trace_df = research.policy_trace.copy()
    policy_trace_df_live = live.policy_trace.copy()

    agg_research = _aggregate_rows(research.metrics_rows)
    agg_live = _aggregate_rows(live.metrics_rows)
    agg_global = _aggregate_rows(global_best.to_dict(orient="records") if not global_best.empty else global_rows)
    comparison_delta = {
        "delta_exp_lcb": float(agg_live["exp_lcb"] - agg_global["exp_lcb"]),
        "delta_maxDD": float(agg_live["maxDD"] - agg_global["maxDD"]),
        "delta_trade_count": float(agg_live["trade_count"] - agg_global["trade_count"]),
    }
    shadow_reject_rate = 0.0
    shadow_top_reasons: dict[str, int] = {}
    if not shadow_df.empty:
        reasons = shadow_df.get("live_reason", pd.Series(dtype=str)).astype(str)
        shadow_top_reasons = {str(k): int(v) for k, v in reasons.value_counts().head(5).items()}
        shadow_reject_rate = float((reasons != "VALID").mean())

    verdict = _classify_verdict(agg_live, agg_global, coverage_ok)
    next_bottleneck = "live_feasibility_constraints" if shadow_reject_rate > 0.3 else ("signal_quality" if float(agg_live["exp_lcb"]) <= 0 else "cost_drag")

    run_id = (
        f"{utc_now_compact()}_"
        f"{stable_hash({'seed': seed, 'symbols': symbols, 'timeframes': timeframes, 'dry_run': dry_run, 'config_hash': compute_config_hash(cfg), 'data_hash': stable_hash(data_hashes, length=16)}, length=12)}"
        "_stage26"
    )
    out_dir = args.runs_dir / run_id / "stage26"
    out_dir.mkdir(parents=True, exist_ok=True)

    effects.to_csv(out_dir / "conditional_effects.csv", index=False)
    (out_dir / "conditional_effects.json").write_text(
        json.dumps(_json_safe({"rows": effects.to_dict(orient="records")}), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    research_df.to_csv(out_dir / "replay_research.csv", index=False)
    (out_dir / "replay_research.json").write_text(
        json.dumps(_json_safe({"rows": research.metrics_rows, "policy": policy}), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    live_df.to_csv(out_dir / "replay_live.csv", index=False)
    (out_dir / "replay_live.json").write_text(
        json.dumps(_json_safe({"rows": live.metrics_rows, "policy": policy}), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    shadow_df.to_csv(out_dir / "shadow_live_rejects.csv", index=False)
    global_df.to_csv(out_dir / "global_baseline_results.csv", index=False)
    policy_trace_df.to_csv(out_dir / "policy_trace.csv", index=False)
    policy_trace_df_live.to_csv(out_dir / "policy_trace_live.csv", index=False)
    pd.DataFrame(context_distribution_rows).to_csv(out_dir / "context_distribution.csv", index=False)

    payload = {
        "stage": "26",
        "run_id": run_id,
        "seed": seed,
        "dry_run": dry_run,
        "requested_symbols": list(coverage_gate.requested_symbols),
        "used_symbols": list(symbols),
        "disabled_symbols": list(coverage_gate.disabled_symbols),
        "head_commit": _git_head(),
        "config_hash": compute_config_hash(cfg),
        "data_hash": stable_hash(data_hashes, length=16),
        "resolved_end_ts": max(resolved_ends).isoformat() if resolved_ends else None,
        "data_coverage_years_by_symbol": {
            str(row["symbol"]): float(row.get("coverage_years", 0.0)) for row in coverage_rows
        },
        "coverage_ok_all_symbols": bool(coverage_ok),
        "required_years": float(required_years),
        "timeframes_tested": list(timeframes),
        "contexts": sorted({str(item["context"]) for item in context_distribution_rows}),
        "top_rulelets_per_context": _top_rulelets_per_context(effects),
        "conditional_policy_metrics_research": agg_research,
        "conditional_policy_metrics_live": agg_live,
        "shadow_live_reject_rate": float(shadow_reject_rate),
        "shadow_live_top_reasons": shadow_top_reasons,
        "global_baseline_metrics": agg_global,
        "comparison_delta": comparison_delta,
        "verdict": verdict,
        "next_bottleneck": next_bottleneck,
        "warnings": warnings,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    (out_dir / "comparison_summary.json").write_text(
        json.dumps(_json_safe(payload), indent=2, allow_nan=False),
        encoding="utf-8",
    )

    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_md = docs_dir / "stage26_report.md"
    report_json = docs_dir / "stage26_report_summary.json"
    coverage_md = docs_dir / "stage26_data_coverage_4y.md"
    coverage_json = docs_dir / "stage26_data_coverage_4y.json"
    comparison_md = docs_dir / "stage26_global_vs_conditional_comparison.md"
    coverage_payload = {
        "stage": "26.1",
        "required_years": float(required_years),
        "coverage_ok_all_symbols": bool(coverage_ok),
        "rows": coverage_rows,
    }
    coverage_json.write_text(json.dumps(_json_safe(coverage_payload), indent=2, allow_nan=False), encoding="utf-8")
    coverage_md.write_text(_render_coverage_md(coverage_payload), encoding="utf-8")
    report_json.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")
    _write_report_md(report_md, payload)
    comparison_md.write_text(_render_comparison_md(agg_global, agg_live, comparison_delta, run_id), encoding="utf-8")

    print(f"run_id: {run_id}")
    print(f"stage26_dir: {out_dir}")
    print(f"report_md: {report_md}")
    print(f"report_json: {report_json}")
    print(f"verdict: {verdict}")


def _render_coverage_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage-26 Data Coverage (4-Year Audit)",
        "",
        f"- required_years: `{payload.get('required_years', 0)}`",
        f"- coverage_ok_all_symbols: `{payload.get('coverage_ok_all_symbols', False)}`",
        "",
        "| symbol | timeframe | exists | start_ts | end_ts | coverage_years | missing_bars_estimate |",
        "| --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for row in payload.get("rows", []):
        lines.append(
            f"| {row.get('symbol','')} | {row.get('timeframe','')} | {row.get('exists', False)} | {row.get('start_ts','')} | {row.get('end_ts','')} | {float(row.get('coverage_years',0.0)):.6f} | {int(row.get('missing_bars_estimate',0))} |"
        )
    return "\n".join(lines).strip() + "\n"


def _render_comparison_md(global_metrics: dict[str, Any], conditional_live: dict[str, Any], delta: dict[str, Any], run_id: str) -> str:
    lines = [
        "# Stage-26 Global vs Conditional Comparison",
        "",
        f"- stage26_run_id: `{run_id}`",
        "",
        "| mode | trade_count | exp_lcb | maxDD |",
        "| --- | ---: | ---: | ---: |",
        f"| global_baseline | {float(global_metrics.get('trade_count',0.0)):.2f} | {float(global_metrics.get('exp_lcb',0.0)):.6f} | {float(global_metrics.get('maxDD',0.0)):.6f} |",
        f"| conditional_live | {float(conditional_live.get('trade_count',0.0)):.2f} | {float(conditional_live.get('exp_lcb',0.0)):.6f} | {float(conditional_live.get('maxDD',0.0)):.6f} |",
        "",
        f"- delta_exp_lcb: `{float(delta.get('delta_exp_lcb',0.0)):.6f}`",
        f"- delta_maxDD: `{float(delta.get('delta_maxDD',0.0)):.6f}`",
        f"- delta_trade_count: `{float(delta.get('delta_trade_count',0.0)):.6f}`",
    ]
    return "\n".join(lines).strip() + "\n"


def _top_rulelets_per_context(effects: pd.DataFrame) -> list[dict[str, Any]]:
    if effects.empty:
        return []
    out: list[dict[str, Any]] = []
    work = effects.copy()
    work["exp_lcb"] = pd.to_numeric(work.get("exp_lcb", 0.0), errors="coerce").fillna(0.0)
    for context, grp in work.groupby("context", dropna=False):
        ranked = grp.sort_values("exp_lcb", ascending=False).head(3)
        for row in ranked.to_dict(orient="records"):
            out.append(
                {
                    "context": str(context),
                    "rulelet": str(row.get("rulelet", "")),
                    "family": str(row.get("family", "")),
                    "context_occurrences": int(row.get("context_occurrences", 0)),
                    "trades_in_context": int(row.get("trades_in_context", 0)),
                    "exp_lcb": float(row.get("exp_lcb", 0.0)),
                    "classification": str(row.get("classification", "")),
                }
            )
    return out


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


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _json_safe(value.tolist())
    if isinstance(value, pd.Timestamp):
        ts = value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")
        return ts.isoformat()
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        out = float(value)
        return out if np.isfinite(out) else 0.0
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


if __name__ == "__main__":
    main()
