"""Run Stage-8.1 walk-forward v2 on deterministic synthetic data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from buffmini.baselines.stage0 import stage06_strategies
from buffmini.config import compute_config_hash, load_config
from buffmini.data.features import calculate_features
from buffmini.utils.time import utc_now_compact
from buffmini.validation.leakage_harness import synthetic_ohlcv
from buffmini.validation.walkforward_v2 import (
    aggregate_windows,
    build_windows,
    evaluate_candidate_on_window,
    write_walkforward_v2_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-8 walk-forward v2")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--rows", type=int, default=24 * 420)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbol", type=str, default="BTC/USDT")
    parser.add_argument("--gating", type=str, default="vol+regime", choices=["none", "vol", "vol+regime"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    wf_cfg = config.get("evaluation", {}).get("stage8", {}).get("walkforward_v2", {})
    rows = int(max(24 * 365, int(args.rows)))
    raw = synthetic_ohlcv(rows=rows, seed=int(args.seed))
    data = calculate_features(raw)

    start_ts = data["timestamp"].iloc[0]
    end_ts = data["timestamp"].iloc[-1] + data["timestamp"].diff().dropna().median()
    windows = build_windows(
        start_ts=start_ts,
        end_ts=end_ts,
        train_days=int(wf_cfg.get("train_days", 180)),
        holdout_days=int(wf_cfg.get("holdout_days", 30)),
        forward_days=int(wf_cfg.get("forward_days", 30)),
        step_days=int(wf_cfg.get("step_days", 30)),
        reserve_tail_days=int(wf_cfg.get("reserve_tail_days", 0)),
    )

    if not windows:
        raise RuntimeError("No walk-forward windows generated; increase rows or adjust stage8.walkforward_v2 config")

    strategy = stage06_strategies()[0]
    candidate = {
        "strategy": strategy,
        "symbol": args.symbol,
        "gating_mode": args.gating,
    }
    metrics = [
        evaluate_candidate_on_window(
            candidate=candidate,
            data=data,
            window_triplet=window,
            cfg=config,
        )
        for window in windows
    ]
    summary = aggregate_windows(metrics, config)

    run_id = f"{utc_now_compact()}_{compute_config_hash(config)[:12]}_stage8_wf"
    out_dir = args.runs_dir / run_id
    write_walkforward_v2_artifacts(
        run_dir=out_dir,
        window_metrics=metrics,
        summary=summary,
        command="python scripts/run_stage8_walkforward.py",
    )

    meta = {
        "run_id": run_id,
        "synthetic_data": True,
        "strategy": strategy.name,
        "symbol": args.symbol,
        "gating": args.gating,
        "config_hash": compute_config_hash(config),
        "summary": summary,
    }
    (out_dir / "stage8_walkforward_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"run_id: {run_id}")
    print(f"classification: {summary['classification']}")
    print(f"usable_windows: {summary['usable_windows']} / {summary['total_windows']}")
    print(f"wrote: {out_dir}")


if __name__ == "__main__":
    main()
