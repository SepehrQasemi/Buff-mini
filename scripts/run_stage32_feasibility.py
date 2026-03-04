"""Run Stage-32 feasibility envelope integration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.config import compute_config_hash, load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.data.snapshot import snapshot_metadata_from_config
from buffmini.stage10.evaluate import _build_features
from buffmini.stage26.context import ContextParams, classify_context
from buffmini.stage32.feasibility import candidate_feasibility_envelope
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-32 feasibility envelope")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--timeframes", type=str, default="15m,30m,1h,2h,4h")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    return parser.parse_args()


def _csv(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def _latest_stage32_validated(runs_dir: Path) -> pd.DataFrame:
    candidates = sorted(runs_dir.glob("*_stage32/stage32/validated.csv"))
    if not candidates:
        return pd.DataFrame()
    path = candidates[-1]
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    symbols = _csv(args.symbols)
    timeframes = _csv(args.timeframes)
    validated = _latest_stage32_validated(Path(args.runs_dir))
    candidate_ids = validated.get("candidate_id", pd.Series(dtype=str)).astype(str).dropna().unique().tolist() if not validated.empty else []
    if not candidate_ids:
        candidate_ids = ["policy_default"]

    rows: list[dict[str, Any]] = []
    data_hash_parts: dict[str, str] = {}
    for timeframe in timeframes:
        fmap = _build_features(
            config=cfg,
            symbols=symbols,
            timeframe=str(timeframe),
            dry_run=bool(args.dry_run),
            seed=int(args.seed),
            data_dir=args.data_dir,
            derived_dir=args.derived_dir,
        )
        for symbol, frame in sorted(fmap.items()):
            local = classify_context(frame, params=ContextParams())
            close = pd.to_numeric(local.get("close", 0.0), errors="coerce").replace(0.0, pd.NA)
            stop = pd.to_numeric(local.get("atr_pct", 0.0), errors="coerce")
            if stop.isna().all():
                hi = pd.to_numeric(local.get("high", 0.0), errors="coerce")
                lo = pd.to_numeric(local.get("low", 0.0), errors="coerce")
                stop = ((hi - lo).abs() / close).fillna(0.0)
            stop = stop.abs().fillna(0.0).clip(lower=0.0001)
            ctx = local.get("ctx_state", pd.Series("UNKNOWN", index=local.index)).astype(str)
            sample = pd.DataFrame(
                {
                    "symbol": str(symbol),
                    "timeframe": str(timeframe),
                    "context": ctx,
                    "stop_dist_pct": stop,
                }
            ).iloc[::10, :].reset_index(drop=True)
            for cid in candidate_ids:
                part = sample.copy()
                part["candidate_id"] = str(cid)
                rows.extend(part.to_dict(orient="records"))
            data_cols = [c for c in ("timestamp", "open", "high", "low", "close", "volume") if c in local.columns]
            data_hash_parts[f"{symbol}|{timeframe}"] = stable_hash(local.loc[:, data_cols].to_dict(orient="list"), length=16)

    eval_cfg = (cfg.get("evaluation", {}) or {})
    constraints_live = ((eval_cfg.get("constraints", {}) or {}).get("live", {}) or {})
    stage24_cfg = (eval_cfg.get("stage24", {}) or {})
    stage24_order_constraints = (stage24_cfg.get("order_constraints", {}) or {})
    stage24_sizing = (stage24_cfg.get("sizing", {}) or {})
    stage24_risk_ladder = (stage24_sizing.get("risk_ladder", {}) or {})

    envelope = candidate_feasibility_envelope(
        signals=pd.DataFrame(rows),
        equity_tiers=[100.0, 1000.0, 10000.0, 100000.0],
        min_notional=float(constraints_live.get("min_trade_notional", 10.0)),
        cost_rt_pct=float((cfg.get("costs", {}) or {}).get("round_trip_cost_pct", 0.1)),
        max_notional_pct=float(stage24_order_constraints.get("max_notional_pct_of_equity", 1.0)),
        risk_cap=float(stage24_risk_ladder.get("r_max", 0.20)),
    )

    run_id = (
        f"{utc_now_compact()}_"
        f"{stable_hash({'seed': int(args.seed), 'symbols': symbols, 'timeframes': timeframes, 'cands': sorted(candidate_ids), 'dry': bool(args.dry_run), 'cfg': compute_config_hash(cfg), 'data': data_hash_parts}, length=12)}"
        "_stage32_feas"
    )
    out_dir = Path(args.runs_dir) / run_id / "stage32"
    out_dir.mkdir(parents=True, exist_ok=True)
    envelope.to_csv(out_dir / "feasibility_envelope.csv", index=False)

    summary: dict[str, Any] = {
        "stage": "32.3",
        "run_id": run_id,
        "seed": int(args.seed),
        "symbols": symbols,
        "timeframes": timeframes,
        "candidate_count": int(len(candidate_ids)),
        "rows": int(envelope.shape[0]),
        "equity_tiers": [100, 1000, 10000, 100000],
        "avg_feasible_pct_by_equity": {
            str(int(eq)): float(val)
            for eq, val in envelope.groupby("equity", dropna=False)["feasible_pct"].mean().to_dict().items()
        }
        if not envelope.empty
        else {},
        "config_hash": compute_config_hash(cfg),
        "data_hash": stable_hash(data_hash_parts, length=16),
        **snapshot_metadata_from_config(cfg),
    }
    (out_dir / "feasibility_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"run_id: {run_id}")
    print(f"stage32_dir: {out_dir}")


if __name__ == "__main__":
    main()
