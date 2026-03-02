"""Run Stage-16 context engine and persistence audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from buffmini.alpha_v2.ab_runner import run_ab_compare
from buffmini.alpha_v2.context import compute_context_states, context_distribution, context_persistence_summary
from buffmini.alpha_v2.reports import summary_hash, write_report_pair
from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.stage10.evaluate import _build_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-16 context persistence")
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
        raise RuntimeError("Stage-16: no features")

    per_symbol_rows: list[dict[str, object]] = []
    transition_rows: list[dict[str, object]] = []
    dist_rows: list[dict[str, object]] = []
    resolved_end_ts = None
    for symbol, frame in sorted(frames.items()):
        with_ctx = compute_context_states(frame)
        durations, trans = context_persistence_summary(with_ctx)
        dist = context_distribution(with_ctx)
        dist_rows.append({"symbol": symbol, **dist})
        for row in durations.to_dict(orient="records"):
            per_symbol_rows.append({"symbol": symbol, **row})
        for from_state, values in trans.iterrows():
            for to_state, prob in values.items():
                transition_rows.append(
                    {"symbol": symbol, "from_state": from_state, "to_state": to_state, "probability": float(prob)}
                )
        end_ts = pd.to_datetime(with_ctx["timestamp"], utc=True, errors="coerce").dropna()
        if not end_ts.empty:
            ts = end_ts.max().isoformat()
            resolved_end_ts = ts if resolved_end_ts is None else max(resolved_end_ts, ts)

    ab = run_ab_compare(
        config=cfg,
        seed=int(args.seed),
        dry_run=bool(args.dry_run),
        symbols=symbols,
        timeframe=str(args.timeframe),
        runs_root=args.runs_dir,
        data_dir=args.data_dir,
        derived_dir=args.derived_dir,
        alpha_enabled=True,
        context_enabled=True,
    )["summary"]

    run_dir = args.runs_dir / ab["run_id"]
    pd.DataFrame(per_symbol_rows).to_csv(run_dir / "context_persistence.csv", index=False)
    pd.DataFrame(transition_rows).to_csv(run_dir / "context_transition_matrix.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(run_dir / "context_distribution.csv", index=False)

    distribution_max = 0.0
    for row in dist_rows:
        distribution_max = max(distribution_max, float(max(v for k, v in row.items() if k != "symbol")))
    failures = []
    if distribution_max > 95.0:
        failures.append("context_distribution_degenerate_over_95pct")
    status = "PASS" if not failures else "FAILED"
    metrics = {
        "run_id": ab["run_id"],
        "seed": ab["seed"],
        "config_hash": ab["config_hash"],
        "data_hash": ab["data_hash"],
        "resolved_end_ts": resolved_end_ts or ab.get("resolved_end_ts"),
        "classic_trade_count": ab["classic"]["trade_count"],
        "alpha_trade_count": ab["alpha_v2"]["trade_count"],
        "classic_exp_lcb": ab["classic"]["exp_lcb"],
        "alpha_exp_lcb": ab["alpha_v2"]["exp_lcb"],
        "max_state_share_pct": distribution_max,
        "summary_hash": summary_hash(ab),
    }
    report_md = Path("docs/stage16_report.md")
    report_json = Path("docs/stage16_summary.json")
    write_report_pair(
        report_md=report_md,
        report_json=report_json,
        title="Stage-16 Report",
        how_to_run=[
            "dry-run: `python scripts/run_stage16.py --dry-run --seed 42`",
            "real-local: `python scripts/run_stage16.py --seed 42`",
        ],
        metrics=metrics,
        status=status,
        failures=failures,
        next_actions=[
            "Stage-17: evaluate exit-v2 variants with fixed entries.",
            "Use transition matrix to weight soft context routing only.",
        ],
        stage_type="non_trading",
        expect_walkforward=False,
        expect_mc=False,
        extras={
            "context_distribution": dist_rows,
            "artifacts": {
                "context_persistence": str((run_dir / "context_persistence.csv").as_posix()),
                "context_transition_matrix": str((run_dir / "context_transition_matrix.csv").as_posix()),
            },
        },
    )
    print(f"run_id: {ab['run_id']}")
    print(f"stage16_summary: {report_json}")
    print(f"stage16_report: {report_md}")
    print(f"max_state_share_pct: {distribution_max:.6f}")


if __name__ == "__main__":
    main()
