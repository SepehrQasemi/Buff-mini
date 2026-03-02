"""Run Stage-22 MTF policy integration with strict causal join."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from buffmini.alpha_v2.contracts import AlphaRole, ClassicFamilyAdapter
from buffmini.alpha_v2.mtf import MtfPolicyConfig, apply_mtf_policy, causal_join_bias
from buffmini.alpha_v2.orchestrator import OrchestratorConfig, run_orchestrator
from buffmini.alpha_v2.reports import summary_hash, write_report_pair
from buffmini.backtest.engine import run_backtest
from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.signals.registry import build_families
from buffmini.stage10.evaluate import _build_features
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-22 MTF policies")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--symbols", type=str, default=None)
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--conflict-mode", type=str, default="net", choices=["net", "hedge", "isolated"])
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
        raise RuntimeError("Stage-22: no features")

    baseline_rows = []
    mtf_rows = []
    stats_rows = []
    resolved_end_ts = None
    for symbol, frame in sorted(frames.items()):
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
            frame=frame,
            contracts=contracts,
            seed=int(args.seed),
            config=cfg,
            orchestrator_cfg=OrchestratorConfig(entry_threshold=0.25, min_confidence=0.05),
        )
        base = frame.copy()
        base["signal"] = intents["side"].shift(1).fillna(0).astype(int)
        bt_base = _run_bt(base, symbol, cfg)
        baseline_rows.append(_metrics(bt_base, base))

        bias_df = frame.iloc[::4][["timestamp", "ema_slope_50"]].copy()
        bias_df["bias_score"] = np.tanh(pd.to_numeric(bias_df["ema_slope_50"], errors="coerce").fillna(0.0) / 0.01)
        joined = causal_join_bias(base_df=frame[["timestamp"]].copy(), bias_df=bias_df[["timestamp", "bias_score"]], bias_col="bias_score")
        policy_df = frame.copy()
        policy_df["entry_score"] = pd.to_numeric(intents["score"], errors="coerce").fillna(0.0)
        policy_df["bias_score"] = pd.to_numeric(joined["bias_score"], errors="coerce").fillna(0.0)
        signal, stats = apply_mtf_policy(
            base_df=policy_df,
            entry_score_col="entry_score",
            bias_score_col="bias_score",
            cfg=MtfPolicyConfig(bias_threshold=0.02, entry_threshold=0.15, conflict_mode=str(args.conflict_mode)),
        )
        with_mtf = frame.copy()
        with_mtf["signal"] = signal.shift(1).fillna(0).astype(int)
        bt_mtf = _run_bt(with_mtf, symbol, cfg)
        mtf_rows.append(_metrics(bt_mtf, with_mtf))
        stats_rows.append({"symbol": symbol, **stats})

        end_ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
        if not end_ts.empty:
            ts = end_ts.max().isoformat()
            resolved_end_ts = ts if resolved_end_ts is None else max(resolved_end_ts, ts)

    baseline = _aggregate(baseline_rows)
    mtf = _aggregate(mtf_rows)
    stats_df = pd.DataFrame(stats_rows)
    delta = {
        "trade_count": float(mtf["trade_count"] - baseline["trade_count"]),
        "exp_lcb": float(mtf["exp_lcb"] - baseline["exp_lcb"]),
        "max_drawdown": float(mtf["max_drawdown"] - baseline["max_drawdown"]),
    }
    conflict_rate = float(stats_df["conflict_rate_pct"].mean()) if not stats_df.empty else 0.0
    bias_align = float(stats_df["bias_alignment_rate_pct"].mean()) if not stats_df.empty else 0.0
    measurable = abs(delta["trade_count"]) > 1e-9 or abs(delta["exp_lcb"]) > 1e-9 or conflict_rate > 0.0
    status = "PASS" if measurable else "FAILED"
    failures = [] if status == "PASS" else ["mtf_effect_not_measurable"]

    run_id = f"{utc_now_compact()}_{stable_hash({'seed': int(args.seed), 'delta': delta}, length=12)}_stage22"
    run_dir = args.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(run_dir / "stage22_mtf_stats.csv", index=False)

    metrics = {
        "run_id": run_id,
        "seed": int(args.seed),
        "config_hash": stable_hash(cfg, length=16),
        "data_hash": stable_hash({"baseline": baseline, "mtf": mtf}, length=16),
        "resolved_end_ts": resolved_end_ts,
        "trade_count": mtf["trade_count"],
        "trades_per_month": mtf["tpm"],
        "exposure_ratio": mtf["exposure_ratio"],
        "PF": mtf["PF"],
        "PF_raw": mtf["PF_raw"],
        "expectancy": mtf["expectancy"],
        "exp_lcb": mtf["exp_lcb"],
        "max_drawdown": mtf["max_drawdown"],
        "walkforward_executed_true_pct": 0.0,
        "usable_windows_count": 0,
        "mc_trigger_rate": 0.0,
        "invalid_pct": 0.0,
        "zero_trade_pct": float(100.0 if mtf["trade_count"] <= 0 else 0.0),
        "conflict_rate_pct": conflict_rate,
        "bias_alignment_rate_pct": bias_align,
        "delta_exp_lcb_vs_baseline": delta["exp_lcb"],
        "summary_hash": summary_hash({"run_id": run_id, "delta": delta, "stats": stats_rows}),
    }
    write_report_pair(
        report_md=Path("docs/stage22_report.md"),
        report_json=Path("docs/stage22_summary.json"),
        title="Stage-22 Report",
        how_to_run=[
            "dry-run: `python scripts/run_stage22.py --dry-run --seed 42`",
            "real-local: `python scripts/run_stage22.py --seed 42`",
        ],
        metrics=metrics,
        status=status,
        failures=failures,
        next_actions=[
            "Run Stage-15..22 master A/B summary.",
            "Inspect conflict mode differences (net/hedge/isolated) under same seed.",
        ],
        extras={"baseline": baseline, "mtf": mtf, "delta": delta, "mtf_stats": stats_rows},
    )
    print(f"run_id: {run_id}")
    print("stage22_summary: docs/stage22_summary.json")
    print("stage22_report: docs/stage22_report.md")
    print(f"status: {status}")


def _run_bt(frame: pd.DataFrame, symbol: str, cfg: dict):
    return run_backtest(
        frame=frame,
        strategy_name="stage22",
        symbol=symbol,
        signal_col="signal",
        exit_mode="fixed_atr",
        max_hold_bars=24,
        stop_atr_multiple=1.5,
        take_profit_atr_multiple=3.0,
        round_trip_cost_pct=float(cfg.get("costs", {}).get("round_trip_cost_pct", 0.1)),
        slippage_pct=float(cfg.get("costs", {}).get("slippage_pct", 0.0005)),
        cost_model_cfg=cfg.get("cost_model", {}),
    )


def _metrics(bt, frame: pd.DataFrame) -> dict[str, float]:
    trade_count = float(bt.metrics.get("trade_count", 0.0))
    tpm = float(trade_count / _months(frame))
    pnl = pd.to_numeric(bt.trades.get("pnl", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    exp_lcb = float(np.mean(pnl) - np.std(pnl, ddof=0) / max(1.0, np.sqrt(float(pnl.size)))) if pnl.size else 0.0
    exposure = float(
        pd.to_numeric(bt.trades.get("bars_held", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum() / max(1.0, float(len(frame)))
    ) if not bt.trades.empty else 0.0
    return {
        "trade_count": trade_count,
        "tpm": tpm,
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
