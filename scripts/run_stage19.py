"""Run Stage-19 state-transition signals integration."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from buffmini.alpha_v2.contracts import AlphaRole, ClassicFamilyAdapter
from buffmini.alpha_v2.orchestrator import OrchestratorConfig, run_orchestrator
from buffmini.alpha_v2.reports import summary_hash, write_report_pair
from buffmini.alpha_v2.transitions import combined_transition_score
from buffmini.backtest.engine import run_backtest
from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.signals.registry import build_families
from buffmini.stage10.evaluate import _build_features
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-19 transition signal integration")
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
        raise RuntimeError("Stage-19: no features")

    baseline_rows = []
    transition_rows = []
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
        baseline = frame.copy()
        baseline["signal"] = intents["side"].shift(1).fillna(0).astype(int)
        bt_base = _run_bt(baseline, symbol, cfg)
        baseline_rows.append(_metrics(bt_base, baseline))

        trans_score = combined_transition_score(frame)
        trans_side = np.where(trans_score >= 0.2, 1, np.where(trans_score <= -0.2, -1, 0))
        mix = np.sign(0.7 * intents["side"].to_numpy(dtype=float) + 0.3 * trans_side.astype(float)).astype(int)
        with_trans = frame.copy()
        with_trans["signal"] = pd.Series(mix, index=frame.index).shift(1).fillna(0).astype(int)
        bt_trans = _run_bt(with_trans, symbol, cfg)
        transition_rows.append(_metrics(bt_trans, with_trans))

        end_ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
        if not end_ts.empty:
            ts = end_ts.max().isoformat()
            resolved_end_ts = ts if resolved_end_ts is None else max(resolved_end_ts, ts)

    baseline = _aggregate(baseline_rows)
    transition = _aggregate(transition_rows)
    delta = {
        "trade_count": float(transition["trade_count"] - baseline["trade_count"]),
        "exp_lcb": float(transition["exp_lcb"] - baseline["exp_lcb"]),
        "max_drawdown": float(transition["max_drawdown"] - baseline["max_drawdown"]),
    }
    zero_trade_pct = float(100.0 if transition["trade_count"] <= 0 else 0.0)
    status = "PASS" if transition["trade_count"] >= baseline["trade_count"] * 0.8 else "FAILED"
    failures = [] if status == "PASS" else ["transition_integration_trade_density_drop_over_20pct"]

    run_id = f"{utc_now_compact()}_{stable_hash({'seed': int(args.seed), 'delta': delta}, length=12)}_stage19"
    run_dir = args.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"mode": "baseline", **baseline},
            {"mode": "with_transition", **transition},
        ]
    ).to_csv(run_dir / "stage19_compare.csv", index=False)

    metrics = {
        "run_id": run_id,
        "seed": int(args.seed),
        "config_hash": stable_hash(cfg, length=16),
        "data_hash": stable_hash({"baseline": baseline, "transition": transition}, length=16),
        "resolved_end_ts": resolved_end_ts,
        "trade_count": transition["trade_count"],
        "trades_per_month": transition["tpm"],
        "exposure_ratio": transition["exposure_ratio"],
        "PF": transition["PF"],
        "PF_raw": transition["PF_raw"],
        "expectancy": transition["expectancy"],
        "exp_lcb": transition["exp_lcb"],
        "max_drawdown": transition["max_drawdown"],
        "walkforward_executed_true_pct": 0.0,
        "usable_windows_count": 0,
        "mc_trigger_rate": 0.0,
        "invalid_pct": 0.0,
        "zero_trade_pct": zero_trade_pct,
        "delta_exp_lcb_vs_baseline": delta["exp_lcb"],
        "summary_hash": summary_hash({"run_id": run_id, "metrics": transition, "delta": delta}),
    }
    write_report_pair(
        report_md=Path("docs/stage19_report.md"),
        report_json=Path("docs/stage19_summary.json"),
        title="Stage-19 Report",
        how_to_run=[
            "dry-run: `python scripts/run_stage19.py --dry-run --seed 42`",
            "real-local: `python scripts/run_stage19.py --seed 42`",
        ],
        metrics=metrics,
        status=status,
        failures=failures,
        next_actions=[
            "Stage-20: rank candidates with robust objective constraints.",
            "Keep transition score bounded to avoid signal spam.",
        ],
        extras={"baseline": baseline, "with_transition": transition, "delta": delta},
    )
    print(f"run_id: {run_id}")
    print("stage19_summary: docs/stage19_summary.json")
    print("stage19_report: docs/stage19_report.md")
    print(f"status: {status}")


def _run_bt(frame: pd.DataFrame, symbol: str, cfg: dict):
    return run_backtest(
        frame=frame,
        strategy_name="stage19",
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

