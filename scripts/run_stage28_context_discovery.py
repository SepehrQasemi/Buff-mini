"""Run Stage-28 context-first discovery and emit candidate matrix artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from buffmini.config import compute_config_hash, load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.data.snapshot import snapshot_metadata_from_config
from buffmini.stage10.evaluate import _build_features
from buffmini.stage26.context import ContextParams, classify_context
from buffmini.stage28.context_discovery import evaluate_context_candidate_matrix
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-28 context-first discovery")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--timeframes", type=str, default="15m,30m,1h,2h,4h")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    return parser.parse_args()


def _csv(raw: str) -> list[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed = int(args.seed)
    symbols = _csv(args.symbols)
    timeframes = _csv(args.timeframes)
    if not symbols or not timeframes:
        raise SystemExit("symbols and timeframes cannot be empty")

    stage26_cfg = dict((cfg.get("evaluation", {}) or {}).get("stage26", {}))
    ctx_cfg = dict(stage26_cfg.get("context", {}))
    ctx_params = ContextParams(
        rank_window=int(ctx_cfg.get("rank_window", 252)),
        vol_window=int(ctx_cfg.get("vol_window", 24)),
        bb_window=int(ctx_cfg.get("bb_window", 20)),
        volume_window=int(ctx_cfg.get("volume_window", 120)),
        chop_window=int(ctx_cfg.get("chop_window", 48)),
        trend_lookback=int(ctx_cfg.get("trend_lookback", 24)),
    )
    costs = dict(cfg.get("costs", {}))
    cost_levels = [
        {
            "name": "realistic",
            "round_trip_cost_pct": float(costs.get("round_trip_cost_pct", 0.1)),
            "slippage_pct": float(costs.get("slippage_pct", 0.0005)),
            "cost_model_cfg": cfg.get("cost_model", {}),
            "stop_atr_multiple": 1.5,
            "take_profit_atr_multiple": 3.0,
            "max_hold_bars": 24,
        },
        {
            "name": "high",
            "round_trip_cost_pct": float(costs.get("round_trip_cost_pct", 0.1)) * 1.8,
            "slippage_pct": float(costs.get("slippage_pct", 0.0005)) * 1.8,
            "cost_model_cfg": cfg.get("cost_model", {}),
            "stop_atr_multiple": 1.5,
            "take_profit_atr_multiple": 3.0,
            "max_hold_bars": 24,
        },
    ]

    rows: list[pd.DataFrame] = []
    for timeframe in timeframes:
        loaded = _build_features(
            config=cfg,
            symbols=symbols,
            timeframe=str(timeframe),
            dry_run=bool(args.dry_run),
            seed=seed,
            data_dir=args.data_dir,
            derived_dir=args.derived_dir,
        )
        for symbol, frame in sorted(loaded.items()):
            with_ctx = classify_context(frame, params=ctx_params)
            matrix = evaluate_context_candidate_matrix(
                frame=with_ctx,
                symbol=str(symbol),
                timeframe=str(timeframe),
                seed=seed,
                contexts=None,
                cost_levels=cost_levels,
            )
            if not matrix.empty:
                rows.append(matrix)

    matrix_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    config_hash = compute_config_hash(cfg)
    run_id = (
        f"{utc_now_compact()}_"
        f"{stable_hash({'seed': seed, 'symbols': symbols, 'timeframes': timeframes, 'dry_run': bool(args.dry_run), 'cfg': config_hash}, length=12)}"
        "_stage28_ctx"
    )
    out_dir = args.runs_dir / run_id / "stage28"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "context_candidate_matrix.csv"
    json_path = out_dir / "context_candidate_matrix.json"
    matrix_df.to_csv(csv_path, index=False)

    payload = {
        "run_id": run_id,
        "seed": seed,
        "dry_run": bool(args.dry_run),
        "symbols": list(symbols),
        "timeframes": list(timeframes),
        "rows": int(matrix_df.shape[0]),
        "pass_count": int((matrix_df.get("classification", pd.Series(dtype=str)) == "PASS").sum()) if not matrix_df.empty else 0,
        "weak_count": int((matrix_df.get("classification", pd.Series(dtype=str)) == "WEAK").sum()) if not matrix_df.empty else 0,
        "rare_count": int((matrix_df.get("classification", pd.Series(dtype=str)) == "RARE").sum()) if not matrix_df.empty else 0,
        "fail_count": int((matrix_df.get("classification", pd.Series(dtype=str)) == "FAIL").sum()) if not matrix_df.empty else 0,
        "config_hash": config_hash,
        **snapshot_metadata_from_config(cfg),
    }
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"run_id: {run_id}")
    print(f"matrix_csv: {csv_path}")
    print(f"matrix_json: {json_path}")


if __name__ == "__main__":
    main()

