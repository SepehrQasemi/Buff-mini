"""Run Stage-26 global (non-conditional) baseline strategies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.backtest.engine import run_backtest
from buffmini.config import compute_config_hash, load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.stage10.evaluate import _build_features
from buffmini.stage26.conditional_eval import bootstrap_lcb
from buffmini.stage26.context import ContextParams, classify_context
from buffmini.stage26.rulelets import build_rulelet_library
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-26 global baseline")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument("--timeframes", type=str, default="")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def _csv(value: str, default: list[str]) -> list[str]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    return items or list(default)


def _months(frame: pd.DataFrame) -> float:
    ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce").dropna()
    if ts.empty:
        return 0.0
    return max(float((ts.iloc[-1] - ts.iloc[0]).total_seconds()) / 86400.0 / 30.0, 1e-9)


def _eval_signal(
    frame: pd.DataFrame,
    signal: pd.Series,
    *,
    symbol: str,
    strategy: str,
    seed: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    costs = dict(config.get("costs", {}))
    result = run_backtest(
        frame=frame.assign(signal=signal),
        strategy_name=f"Stage26Global::{strategy}",
        symbol=str(symbol),
        signal_col="signal",
        stop_atr_multiple=1.5,
        take_profit_atr_multiple=3.0,
        max_hold_bars=24,
        round_trip_cost_pct=float(costs.get("round_trip_cost_pct", 0.1)),
        slippage_pct=float(costs.get("slippage_pct", 0.0005)),
        exit_mode="fixed_atr",
        cost_model_cfg=config.get("cost_model", {}),
    )
    pnls = pd.to_numeric(result.trades.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    exp_lcb = bootstrap_lcb(values=pnls, seed=int(seed), samples=500)
    trade_count = float(result.metrics.get("trade_count", 0.0))
    return {
        "trade_count": float(trade_count),
        "tpm": float(trade_count / max(1e-9, _months(frame))),
        "PF_raw": float(result.metrics.get("profit_factor", 0.0)),
        "expectancy": float(result.metrics.get("expectancy", 0.0)),
        "exp_lcb": float(exp_lcb),
        "maxDD": float(result.metrics.get("max_drawdown", 0.0)),
        "signal_density": float((signal != 0).mean()),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    stage26 = dict((cfg.get("evaluation", {}) or {}).get("stage26", {}))
    symbols = _csv(args.symbols, default=list(stage26.get("symbols", ["BTC/USDT", "ETH/USDT"])))
    timeframes = _csv(args.timeframes, default=list(stage26.get("timeframes", ["1h"])))
    ctx_cfg = dict(stage26.get("context", {}))
    params = ContextParams(
        rank_window=int(ctx_cfg.get("rank_window", 252)),
        vol_window=int(ctx_cfg.get("vol_window", 24)),
        bb_window=int(ctx_cfg.get("bb_window", 20)),
        volume_window=int(ctx_cfg.get("volume_window", 120)),
        chop_window=int(ctx_cfg.get("chop_window", 48)),
        trend_lookback=int(ctx_cfg.get("trend_lookback", 24)),
    )
    rulelets = build_rulelet_library()
    rows: list[dict[str, Any]] = []
    data_hashes: dict[str, str] = {}
    resolved_ends: list[pd.Timestamp] = []
    for tf in timeframes:
        frames = _build_features(
            config=cfg,
            symbols=symbols,
            timeframe=str(tf),
            dry_run=bool(args.dry_run),
            seed=int(args.seed),
            data_dir=args.data_dir,
            derived_dir=args.derived_dir,
        )
        for symbol, frame in frames.items():
            data_hashes[f"{symbol}|{tf}"] = _frame_data_hash(frame)
            ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce").dropna()
            if not ts.empty:
                resolved_ends.append(ts.max())
            with_ctx = classify_context(frame, params=params)
            best_name = ""
            best_exp_lcb = -1e18
            best_row: dict[str, Any] | None = None
            score_bank: list[pd.Series] = []
            for name, rulelet in rulelets.items():
                score = pd.to_numeric(rulelet.compute_score(with_ctx), errors="coerce").fillna(0.0)
                signal = pd.Series(0, index=with_ctx.index, dtype=int)
                signal.loc[score >= float(rulelet.threshold)] = 1
                signal.loc[score <= -float(rulelet.threshold)] = -1
                signal = signal.shift(1).fillna(0).astype(int)
                m = _eval_signal(
                    with_ctx,
                    signal,
                    symbol=str(symbol),
                    strategy=str(name),
                    seed=int(args.seed),
                    config=cfg,
                )
                row = {
                    "symbol": str(symbol),
                    "timeframe": str(tf),
                    "variant": "global_single",
                    "rulelet": str(name),
                    **m,
                }
                rows.append(row)
                score_bank.append(score)
                if float(m["exp_lcb"]) > best_exp_lcb:
                    best_exp_lcb = float(m["exp_lcb"])
                    best_name = str(name)
                    best_row = row
            if score_bank:
                ensemble_score = pd.concat(score_bank, axis=1).mean(axis=1).fillna(0.0)
                ens_signal = pd.Series(0, index=with_ctx.index, dtype=int)
                ens_signal.loc[ensemble_score >= 0.25] = 1
                ens_signal.loc[ensemble_score <= -0.25] = -1
                ens_signal = ens_signal.shift(1).fillna(0).astype(int)
                m_ens = _eval_signal(
                    with_ctx,
                    ens_signal,
                    symbol=str(symbol),
                    strategy="global_ensemble",
                    seed=int(args.seed),
                    config=cfg,
                )
                rows.append(
                    {
                        "symbol": str(symbol),
                        "timeframe": str(tf),
                        "variant": "global_ensemble",
                        "rulelet": "global_ensemble",
                        **m_ens,
                    }
                )
            if best_row is not None:
                rows.append(
                    {
                        "symbol": str(symbol),
                        "timeframe": str(tf),
                        "variant": "global_best_single",
                        "rulelet": best_name,
                        "trade_count": best_row["trade_count"],
                        "tpm": best_row["tpm"],
                        "PF_raw": best_row["PF_raw"],
                        "expectancy": best_row["expectancy"],
                        "exp_lcb": best_row["exp_lcb"],
                        "maxDD": best_row["maxDD"],
                        "signal_density": best_row["signal_density"],
                    }
                )

    out_df = pd.DataFrame(rows)
    run_id = f"{utc_now_compact()}_{stable_hash({'seed': int(args.seed), 'symbols': symbols, 'timeframes': timeframes, 'cfg': compute_config_hash(cfg), 'dry_run': bool(args.dry_run)}, length=12)}_stage26_global"
    out_dir = args.runs_dir / run_id / "stage26"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_dir / "global_baseline_results.csv", index=False)
    payload = {
        "stage": "26.6",
        "run_id": run_id,
        "seed": int(args.seed),
        "dry_run": bool(args.dry_run),
        "config_hash": compute_config_hash(cfg),
        "data_hash": stable_hash(data_hashes, length=16),
        "data_hashes_by_symbol_timeframe": data_hashes,
        "resolved_end_ts": max(resolved_ends).isoformat() if resolved_ends else None,
        "rows": out_df.to_dict(orient="records"),
    }
    (out_dir / "global_baseline_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    docs = docs_dir / "stage26_global_vs_conditional_comparison.md"
    lines = [
        "# Stage-26 Global vs Conditional Comparison",
        "",
        f"- global_run_id: `{run_id}`",
        "",
        "| symbol | timeframe | variant | rulelet | trade_count | exp_lcb | maxDD |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in out_df.sort_values(["symbol", "timeframe", "variant", "rulelet"]).to_dict(orient="records"):
        lines.append(
            f"| {row.get('symbol','')} | {row.get('timeframe','')} | {row.get('variant','')} | {row.get('rulelet','')} | {float(row.get('trade_count',0.0)):.2f} | {float(row.get('exp_lcb',0.0)):.6f} | {float(row.get('maxDD',0.0)):.6f} |"
        )
    docs.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    print(f"run_id: {run_id}")
    print(f"results_csv: {out_dir / 'global_baseline_results.csv'}")
    print(f"results_json: {out_dir / 'global_baseline_results.json'}")
    print(f"docs: {docs}")


def _frame_data_hash(frame: pd.DataFrame) -> str:
    cols = [c for c in ("timestamp", "open", "high", "low", "close", "volume") if c in frame.columns]
    if not cols:
        return stable_hash({"rows": int(frame.shape[0])}, length=16)
    payload = frame.loc[:, cols].to_dict(orient="list")
    return stable_hash(payload, length=16)


if __name__ == "__main__":
    main()
