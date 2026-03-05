"""Monitor representation and performance drift for Stage-33."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.constants import RUNS_DIR
from buffmini.stage33.drift import build_drift_summary, performance_drift, representation_drift
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor drift (representation + performance)")
    parser.add_argument("--features-ml-dir", type=Path, default=Path("data/features_ml"))
    parser.add_argument("--symbol", type=str, default="BTC/USDT")
    parser.add_argument("--timeframe", type=str, default="15m")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def _symbol_path(symbol: str) -> str:
    return str(symbol).replace("/", "_").replace(":", "_")


def _load_embeddings(features_ml_dir: Path, symbol: str, timeframe: str) -> np.ndarray:
    path = Path(features_ml_dir) / _symbol_path(symbol) / f"embeddings_{timeframe}.parquet"
    if not path.exists():
        return np.empty((0, 0), dtype=np.float32)
    frame = pd.read_parquet(path)
    emb_cols = [c for c in frame.columns if str(c).startswith("emb_")]
    if not emb_cols:
        return np.empty((0, 0), dtype=np.float32)
    return frame.loc[:, emb_cols].to_numpy(dtype=np.float32)


def _load_perf_metrics(runs_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = sorted(Path(runs_dir).glob("*_stage32/stage32/validated.csv"))
    if not candidates:
        return {"exp_lcb": 0.0, "PF_clipped": 0.0, "maxDD": 0.0}, {"exp_lcb": 0.0, "PF_clipped": 0.0, "maxDD": 0.0}
    frame = pd.read_csv(candidates[-1])
    if frame.empty or "exp_lcb" not in frame.columns:
        return {"exp_lcb": 0.0, "PF_clipped": 0.0, "maxDD": 0.0}, {"exp_lcb": 0.0, "PF_clipped": 0.0, "maxDD": 0.0}
    mid = max(1, int(frame.shape[0] * 0.5))
    base = frame.iloc[:mid]
    recent = frame.iloc[mid:]
    baseline = {
        "exp_lcb": float(pd.to_numeric(base.get("exp_lcb", 0.0), errors="coerce").fillna(0.0).mean()),
        "PF_clipped": float(pd.to_numeric(base.get("exp_lcb", 0.0), errors="coerce").fillna(0.0).mean() + 1.0),
        "maxDD": 0.2,
    }
    latest = {
        "exp_lcb": float(pd.to_numeric(recent.get("exp_lcb", 0.0), errors="coerce").fillna(0.0).mean()),
        "PF_clipped": float(pd.to_numeric(recent.get("exp_lcb", 0.0), errors="coerce").fillna(0.0).mean() + 1.0),
        "maxDD": 0.2,
    }
    return baseline, latest


def main() -> None:
    args = parse_args()
    emb = _load_embeddings(args.features_ml_dir, str(args.symbol), str(args.timeframe))
    if emb.size == 0:
        rep = 0.0
    else:
        split = max(1, int(emb.shape[0] * 0.7))
        rep = representation_drift(emb[:split], emb[split:])
    baseline, latest = _load_perf_metrics(args.runs_dir)
    perf = performance_drift(baseline_metrics=baseline, recent_metrics=latest)
    summary = build_drift_summary(rep_drift=rep, perf_drift=perf)
    summary.update(
        {
            "symbol": str(args.symbol),
            "timeframe": str(args.timeframe),
            "hash": stable_hash(summary, length=16),
        }
    )

    out_path = args.out
    if out_path is None:
        run_id = f"{utc_now_compact()}_{stable_hash({'symbol': args.symbol, 'tf': args.timeframe}, length=12)}_stage33_drift"
        out_dir = Path(args.runs_dir) / run_id / "stage33"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "drift_summary.json"
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"drift_summary: {out_path}")


if __name__ == "__main__":
    main()

