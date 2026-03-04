"""Generate Stage-28 window calendar artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.stage10.evaluate import _build_features
from buffmini.stage28.window_calendar import generate_window_calendar
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Stage-28 rolling window calendar")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbols", type=str, default="BTC/USDT,ETH/USDT")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--window-months", type=int, default=3)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    symbols = [item.strip() for item in str(args.symbols).split(",") if item.strip()]
    if not symbols:
        raise SystemExit("No symbols provided")

    loaded = _build_features(
        config=cfg,
        symbols=symbols,
        timeframe=str(args.timeframe),
        dry_run=bool(args.dry_run),
        seed=int(args.seed),
        data_dir=args.data_dir,
        derived_dir=args.derived_dir,
    )
    first_symbol = sorted(loaded.keys())[0]
    frame = loaded[first_symbol]
    ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce").dropna()
    calendar = generate_window_calendar(
        ts,
        window_months=int(args.window_months),
        step_months=int(args.step_months),
    )

    run_id = (
        f"{utc_now_compact()}_"
        f"{stable_hash({'seed': int(args.seed), 'symbol': first_symbol, 'timeframe': str(args.timeframe), 'window_months': int(args.window_months), 'step_months': int(args.step_months), 'dry_run': bool(args.dry_run)}, length=12)}"
        "_stage28_win"
    )
    out_dir = args.runs_dir / run_id / "stage28"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "window_calendar.csv"
    json_path = out_dir / "window_calendar.json"
    calendar.to_csv(csv_path, index=False)
    payload = {
        "run_id": run_id,
        "seed": int(args.seed),
        "symbol": str(first_symbol),
        "timeframe": str(args.timeframe),
        "window_months": int(args.window_months),
        "step_months": int(args.step_months),
        "windows": int(calendar.shape[0]),
        "dry_run": bool(args.dry_run),
    }
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(f"run_id: {run_id}")
    print(f"window_csv: {csv_path}")
    print(f"window_json: {json_path}")


if __name__ == "__main__":
    main()

