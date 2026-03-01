"""Run Stage-17 exit engine v2 decomposition."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from buffmini.alpha_v2.contracts import AlphaRole, ClassicFamilyAdapter
from buffmini.alpha_v2.orchestrator import OrchestratorConfig, run_orchestrator
from buffmini.alpha_v2.reports import summary_hash, write_report_pair
from buffmini.backtest.engine import run_backtest
from buffmini.baselines.stage0 import generate_signals, trend_pullback
from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.signals.registry import build_families
from buffmini.stage10.evaluate import _build_features
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-17 exit-v2")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--symbols", type=str, default=None)
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    symbols = [item.strip() for item in str(args.symbols).split(",") if item.strip()] if args.symbols else None
    frames = _build_features(
        config=cfg,
        symbols=symbols or cfg.get("universe", {}).get("symbols", ["BTC/USDT", "ETH/USDT"]),
        timeframe=str(args.timeframe),
        dry_run=bool(args.dry_run),
        seed=int(args.seed),
        data_dir=args.data_dir,
        derived_dir=args.derived_dir,
    )
    if not frames:
        raise RuntimeError("Stage-17: no features")

    variant_rows: list[dict[str, object]] = []
    resolved_end_ts = None
    classic_fixed_rows = []
    classic_best_rows = []
    alpha_fixed_rows = []
    alpha_best_rows = []

    for symbol, frame in sorted(frames.items()):
        strat = trend_pullback()
        classic = frame.copy()
        classic["signal"] = generate_signals(classic, strategy=strat, gating_mode="none")
        alpha = frame.copy()
        families = build_families(enabled=["price", "volatility", "flow"], cfg=cfg)
        contracts = [
            ClassicFamilyAdapter(
                name=f"entry::{name}",
                family=name,
                role=AlphaRole.ENTRY,
                wrapped=family,
                params={"symbol": symbol, "timeframe": str(args.timeframe), "seed": int(args.seed), "config": cfg},
            )
            for name, family in families.items()
        ]
        intents, _ = run_orchestrator(
            frame=alpha,
            contracts=contracts,
            seed=int(args.seed),
            config=cfg,
            orchestrator_cfg=OrchestratorConfig(entry_threshold=0.25, min_confidence=0.05),
        )
        alpha["signal"] = intents["side"].shift(1).fillna(0).astype(int)

        variants = {
            "fixed_atr": {"exit_mode": "fixed_atr", "max_hold_bars": 24, "trailing_atr_k": 1.5, "partial_size": 0.5},
            "atr_trailing": {"exit_mode": "trailing_atr", "max_hold_bars": 24, "trailing_atr_k": 1.5, "partial_size": 0.5},
            "time_progress": {"exit_mode": "fixed_atr", "max_hold_bars": 12, "trailing_atr_k": 1.5, "partial_size": 0.5},
            "partial_runner": {"exit_mode": "partial_then_trail", "max_hold_bars": 24, "trailing_atr_k": 1.5, "partial_size": 0.5},
            "mae_mfe_tighten": {"exit_mode": "trailing_atr", "max_hold_bars": 24, "trailing_atr_k": 1.2, "partial_size": 0.5},
        }
        classic_variant_metrics: dict[str, dict[str, float]] = {}
        alpha_variant_metrics: dict[str, dict[str, float]] = {}
        for label, p in variants.items():
            classic_bt = _run_bt(frame=classic, symbol=symbol, cfg=cfg, **p)
            alpha_bt = _run_bt(frame=alpha, symbol=symbol, cfg=cfg, **p)
            classic_m = _metrics(classic_bt, classic)
            alpha_m = _metrics(alpha_bt, alpha)
            classic_variant_metrics[label] = classic_m
            alpha_variant_metrics[label] = alpha_m
            variant_rows.append(
                {
                    "symbol": symbol,
                    "variant": label,
                    "classic_exp_lcb": classic_m["exp_lcb"],
                    "alpha_exp_lcb": alpha_m["exp_lcb"],
                    "classic_trade_count": classic_m["trade_count"],
                    "alpha_trade_count": alpha_m["trade_count"],
                    "classic_maxdd": classic_m["max_drawdown"],
                    "alpha_maxdd": alpha_m["max_drawdown"],
                }
            )

        classic_fixed_rows.append(classic_variant_metrics["fixed_atr"])
        alpha_fixed_rows.append(alpha_variant_metrics["fixed_atr"])
        best_c = max(classic_variant_metrics.items(), key=lambda kv: (kv[1]["exp_lcb"], -kv[1]["max_drawdown"]))
        best_a = max(alpha_variant_metrics.items(), key=lambda kv: (kv[1]["exp_lcb"], -kv[1]["max_drawdown"]))
        classic_best_rows.append(best_c[1])
        alpha_best_rows.append(best_a[1])

        end_ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
        if not end_ts.empty:
            ts = end_ts.max().isoformat()
            resolved_end_ts = ts if resolved_end_ts is None else max(resolved_end_ts, ts)

    classic_fixed = _aggregate(classic_fixed_rows)
    alpha_fixed = _aggregate(alpha_fixed_rows)
    classic_best = _aggregate(classic_best_rows)
    alpha_best = _aggregate(alpha_best_rows)
    decomposition = {
        "entry_only_delta_exp_lcb": float(alpha_fixed["exp_lcb"] - classic_fixed["exp_lcb"]),
        "exit_only_delta_exp_lcb": float(classic_best["exp_lcb"] - classic_fixed["exp_lcb"]),
        "combined_delta_exp_lcb": float(alpha_best["exp_lcb"] - classic_fixed["exp_lcb"]),
    }
    status = "PASS" if (decomposition["combined_delta_exp_lcb"] > 0 or alpha_best["max_drawdown"] < classic_fixed["max_drawdown"]) else "FAILED"
    failures = [] if status == "PASS" else ["no_exit_variant_improvement_detected"]

    summary = {
        "run_id": f"{utc_now_compact()}_{stable_hash({'seed': int(args.seed), 'decomposition': decomposition}, length=12)}_stage17",
        "seed": int(args.seed),
        "config_hash": stable_hash(cfg, length=16),
        "data_hash": stable_hash(variant_rows, length=16),
        "resolved_end_ts": resolved_end_ts,
        "classic_fixed": classic_fixed,
        "classic_best_exit": classic_best,
        "alpha_fixed": alpha_fixed,
        "alpha_best_exit": alpha_best,
        "decomposition": decomposition,
        "status": status,
    }
    run_dir = args.runs_dir / summary["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(variant_rows).to_csv(run_dir / "stage17_exit_variants.csv", index=False)

    metrics = {
        "run_id": summary["run_id"],
        "seed": summary["seed"],
        "config_hash": summary["config_hash"],
        "data_hash": summary["data_hash"],
        "resolved_end_ts": summary["resolved_end_ts"],
        "trade_count": alpha_best["trade_count"],
        "trades_per_month": alpha_best["tpm"],
        "exposure_ratio": alpha_best["exposure_ratio"],
        "PF": alpha_best["PF"],
        "PF_raw": alpha_best["PF_raw"],
        "expectancy": alpha_best["expectancy"],
        "exp_lcb": alpha_best["exp_lcb"],
        "max_drawdown": alpha_best["max_drawdown"],
        "walkforward_executed_true_pct": 0.0,
        "usable_windows_count": 0,
        "mc_trigger_rate": 0.0,
        "invalid_pct": 0.0,
        "zero_trade_pct": float(100.0 if alpha_best["trade_count"] <= 0 else 0.0),
        "drag_sensitivity_delta_exp_lcb": decomposition["combined_delta_exp_lcb"],
        "runtime_seconds": 0.0,
        "cache_hit_rate": 0.0,
        "summary_hash": summary_hash(summary),
    }
    write_report_pair(
        report_md=Path("docs/stage17_report.md"),
        report_json=Path("docs/stage17_summary.json"),
        title="Stage-17 Report",
        how_to_run=[
            "dry-run: `python scripts/run_stage17.py --dry-run --seed 42`",
            "real-local: `python scripts/run_stage17.py --seed 42`",
        ],
        metrics=metrics,
        status=status,
        failures=failures,
        next_actions=[
            "Stage-18: test conditional effects by context state.",
            "Keep same-candle exit priority unchanged in engine core.",
        ],
        extras={"decomposition": decomposition},
    )
    print(f"run_id: {summary['run_id']}")
    print("stage17_summary: docs/stage17_summary.json")
    print("stage17_report: docs/stage17_report.md")
    print(f"status: {status}")


def _run_bt(
    *,
    frame: pd.DataFrame,
    symbol: str,
    cfg: dict,
    exit_mode: str,
    max_hold_bars: int,
    trailing_atr_k: float,
    partial_size: float,
):
    return run_backtest(
        frame=frame,
        strategy_name=f"stage17::{exit_mode}",
        symbol=symbol,
        signal_col="signal",
        max_hold_bars=int(max_hold_bars),
        stop_atr_multiple=1.5,
        take_profit_atr_multiple=3.0,
        exit_mode=str(exit_mode),
        trailing_atr_k=float(trailing_atr_k),
        partial_size=float(partial_size),
        round_trip_cost_pct=float(cfg.get("costs", {}).get("round_trip_cost_pct", 0.1)),
        slippage_pct=float(cfg.get("costs", {}).get("slippage_pct", 0.0005)),
        cost_model_cfg=cfg.get("cost_model", {}),
    )


def _metrics(bt, frame: pd.DataFrame) -> dict[str, float]:
    trade_count = float(bt.metrics.get("trade_count", 0.0))
    months = _months(frame)
    pnl = pd.to_numeric(bt.trades.get("pnl", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    exp_lcb = float(np.mean(pnl) - np.std(pnl, ddof=0) / max(1.0, np.sqrt(float(pnl.size)))) if pnl.size else 0.0
    exposure = float(
        pd.to_numeric(bt.trades.get("bars_held", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum() / max(1.0, float(len(frame)))
    ) if not bt.trades.empty else 0.0
    return {
        "trade_count": trade_count,
        "tpm": float(trade_count / months),
        "exposure_ratio": exposure,
        "PF_raw": float(bt.metrics.get("profit_factor", 0.0)),
        "PF": float(np.clip(float(bt.metrics.get("profit_factor", 0.0)), 0.0, 10.0)),
        "expectancy": float(bt.metrics.get("expectancy", 0.0)),
        "exp_lcb": exp_lcb,
        "max_drawdown": float(bt.metrics.get("max_drawdown", 0.0)),
    }


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = ["trade_count", "tpm", "exposure_ratio", "PF_raw", "PF", "expectancy", "exp_lcb", "max_drawdown"]
    if not rows:
        return {k: 0.0 for k in keys}
    return {k: float(np.mean([float(r[k]) for r in rows])) for k in keys}


def _months(frame: pd.DataFrame) -> float:
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return 1.0
    return max(float((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400.0) / 30.0, 1e-9)


if __name__ == "__main__":
    main()

