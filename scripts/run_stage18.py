"""Run Stage-18 conditional edge framework."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from buffmini.alpha_v2.conditional_tests import (
    ConditionalTestConfig,
    apply_falsification_rules,
    conditional_effects_table,
    suggest_context_policy,
)
from buffmini.alpha_v2.context import compute_context_states
from buffmini.alpha_v2.reports import summary_hash, write_report_pair
from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.signals.families.price import PriceStructureFamily
from buffmini.signals.family_base import FamilyContext
from buffmini.stage10.evaluate import _build_features
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-18 conditional edge tests")
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
        raise RuntimeError("Stage-18: no features")

    cfg_test = ConditionalTestConfig(bootstrap_samples=600, seed=int(args.seed), min_samples=30, ci_alpha=0.05)
    family = PriceStructureFamily(params={"entry_threshold": 0.3})
    tables = []
    policy_rows = []
    resolved_end_ts = None
    for symbol, frame in sorted(frames.items()):
        with_ctx = compute_context_states(frame)
        ctx = FamilyContext(symbol=symbol, timeframe=str(args.timeframe), seed=int(args.seed), config=cfg, params={})
        score = family.compute_scores(with_ctx, ctx)
        work = with_ctx.copy()
        work["stage18_signal"] = (pd.to_numeric(score, errors="coerce").fillna(0.0) > 0.30).astype(int)
        work["forward_return_24"] = pd.to_numeric(work["close"], errors="coerce").pct_change(24).shift(-24).fillna(0.0)
        tbl = conditional_effects_table(
            frame=work,
            signal_col="stage18_signal",
            context_col="ctx_state",
            forward_return_col="forward_return_24",
            cfg=cfg_test,
        )
        tbl = apply_falsification_rules(table=tbl, min_samples=cfg_test.min_samples)
        tbl["symbol"] = symbol
        tables.append(tbl)
        policy = suggest_context_policy(tbl)
        for st, weight in policy.items():
            policy_rows.append({"symbol": symbol, "context": st, "weight": float(weight)})
        end_ts = pd.to_datetime(work["timestamp"], utc=True, errors="coerce").dropna()
        if not end_ts.empty:
            ts = end_ts.max().isoformat()
            resolved_end_ts = ts if resolved_end_ts is None else max(resolved_end_ts, ts)

    table = pd.concat(tables, axis=0, ignore_index=True) if tables else pd.DataFrame()
    accepted_count = int(table["accepted"].astype(bool).sum()) if not table.empty else 0
    status = "PASS" if not table.empty else "FAILED"
    failures = []
    if table.empty:
        failures.append("no_conditional_rows")
    if not table.empty and accepted_count == 0:
        failures.append("no_accepted_conditional_effects")

    run_id = f"{utc_now_compact()}_{stable_hash({'seed': int(args.seed), 'rows': len(table)}, length=12)}_stage18"
    run_dir = args.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(run_dir / "stage18_conditional_table.csv", index=False)
    pd.DataFrame(policy_rows).to_csv(run_dir / "stage18_policy_suggestions.csv", index=False)

    metrics = {
        "run_id": run_id,
        "seed": int(args.seed),
        "config_hash": stable_hash(cfg, length=16),
        "data_hash": stable_hash(table.to_dict(orient="records"), length=16),
        "resolved_end_ts": resolved_end_ts,
        "trade_count": 0.0,
        "trades_per_month": 0.0,
        "exposure_ratio": 0.0,
        "PF": 0.0,
        "PF_raw": 0.0,
        "expectancy": 0.0,
        "exp_lcb": float(pd.to_numeric(table["median_diff"], errors="coerce").fillna(0.0).mean()) if not table.empty else 0.0,
        "max_drawdown": 0.0,
        "walkforward_executed_true_pct": 0.0,
        "usable_windows_count": 0,
        "mc_trigger_rate": 0.0,
        "invalid_pct": float(100.0 if table.empty else 0.0),
        "zero_trade_pct": 0.0,
        "accepted_effects_count": accepted_count,
        "summary_hash": summary_hash({"run_id": run_id, "accepted": accepted_count, "status": status}),
    }
    write_report_pair(
        report_md=Path("docs/stage18_report.md"),
        report_json=Path("docs/stage18_summary.json"),
        title="Stage-18 Report",
        how_to_run=[
            "dry-run: `python scripts/run_stage18.py --dry-run --seed 42`",
            "real-local: `python scripts/run_stage18.py --seed 42`",
        ],
        metrics=metrics,
        status=status,
        failures=failures,
        next_actions=[
            "Stage-19: map accepted contexts into transition entry components.",
            "Keep falsification rules strict when sample sizes shrink.",
        ],
        extras={
            "policy_suggestions": policy_rows,
            "artifacts": {
                "conditional_table": str((run_dir / "stage18_conditional_table.csv").as_posix()),
                "policy_table": str((run_dir / "stage18_policy_suggestions.csv").as_posix()),
            },
        },
    )
    print(f"run_id: {run_id}")
    print("stage18_summary: docs/stage18_summary.json")
    print("stage18_report: docs/stage18_report.md")
    print(f"accepted_effects_count: {accepted_count}")


if __name__ == "__main__":
    main()

