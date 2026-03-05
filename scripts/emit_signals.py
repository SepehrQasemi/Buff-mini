"""Emit local policy signals (no exchange execution)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.stage10.evaluate import _build_features
from buffmini.stage26.context import ContextParams, classify_context
from buffmini.stage33.emitter import emit_signal_payload, load_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emit local strategy signal")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--policy-path", type=Path, default=None)
    parser.add_argument("--symbol", type=str, default="BTC/USDT")
    parser.add_argument("--timeframe", type=str, default="1h")
    parser.add_argument("--asof-ts", type=str, default="")
    parser.add_argument("--equity", type=float, default=1000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR)
    parser.add_argument("--derived-dir", type=Path, default=DERIVED_DATA_DIR)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def _find_latest_policy(runs_dir: Path) -> Path:
    candidates = sorted(Path(runs_dir).glob("*_stage33/stage33/policy.json"))
    if not candidates:
        raise FileNotFoundError("No policy.json found under runs/*_stage33/stage33/")
    return candidates[-1]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    policy_path = Path(args.policy_path) if args.policy_path else _find_latest_policy(Path(args.runs_dir))
    policy = load_policy(policy_path)
    frame_map = _build_features(
        config=cfg,
        symbols=[str(args.symbol)],
        timeframe=str(args.timeframe),
        dry_run=bool(args.dry_run),
        seed=int(args.seed),
        data_dir=args.data_dir,
        derived_dir=args.derived_dir,
    )
    frame = frame_map.get(str(args.symbol))
    if frame is None:
        raise SystemExit(f"No frame loaded for symbol={args.symbol}")
    frame_ctx = classify_context(frame, params=ContextParams())
    payload = emit_signal_payload(
        frame=frame_ctx,
        policy=policy,
        symbol=str(args.symbol),
        timeframe=str(args.timeframe),
        asof_ts=str(args.asof_ts).strip() or None,
        equity=float(args.equity),
    )
    text = json.dumps(payload, indent=2, allow_nan=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

