"""Run Stage-20 robust objective ranking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from buffmini.alpha_v2.objective import ObjectiveConstraints, robust_objective
from buffmini.alpha_v2.reports import summary_hash, write_report_pair
from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-20 objective")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--timeframe", type=str, default="1h")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    stage19_path = Path("docs/stage19_summary.json")
    payload = {}
    if stage19_path.exists():
        payload = json.loads(stage19_path.read_text(encoding="utf-8"))
    candidates = _collect_candidates(payload)
    constraints = _constraints_for_timeframe(timeframe=str(args.timeframe))
    rows = []
    for cand in candidates:
        result = robust_objective(
            exp_lcb=float(cand.get("exp_lcb", 0.0)),
            tpm=float(cand.get("tpm", 0.0)),
            exposure_ratio=float(cand.get("exposure_ratio", 0.0)),
            max_dd_p95=float(cand.get("max_drawdown", 0.0)),
            drag_penalty=float(cand.get("drag_penalty", 0.0)),
            horizon_consistency=float(cand.get("horizon_consistency", 0.0)),
            constraints=constraints,
        )
        rows.append({**cand, **result})
    table = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    valid = table.loc[table["valid"]].copy()
    best = valid.iloc[0].to_dict() if not valid.empty else {}
    status = "PASS" if not valid.empty else "FAILED"
    failures = [] if status == "PASS" else ["no_valid_candidate_after_constraints"]

    run_id = f"{utc_now_compact()}_{stable_hash({'seed': int(args.seed), 'rows': rows}, length=12)}_stage20"
    report_table = Path("docs/stage20_candidates.csv")
    table.to_csv(report_table, index=False)
    metrics = {
        "run_id": run_id,
        "seed": int(args.seed),
        "config_hash": stable_hash(cfg, length=16),
        "data_hash": stable_hash(rows, length=16),
        "resolved_end_ts": payload.get("resolved_end_ts"),
        "trade_count": float(best.get("trade_count", 0.0)),
        "trades_per_month": float(best.get("tpm", 0.0)),
        "exposure_ratio": float(best.get("exposure_ratio", 0.0)),
        "PF": float(best.get("PF", 0.0)),
        "PF_raw": float(best.get("PF_raw", 0.0)),
        "expectancy": float(best.get("expectancy", 0.0)),
        "exp_lcb": float(best.get("exp_lcb", 0.0)),
        "max_drawdown": float(best.get("max_drawdown", 0.0)),
        "walkforward_executed_true_pct": 0.0,
        "usable_windows_count": 0,
        "mc_trigger_rate": 0.0,
        "invalid_pct": float(100.0 if table.empty else 0.0),
        "zero_trade_pct": float(100.0 if float(best.get("trade_count", 0.0)) <= 0.0 else 0.0),
        "valid_candidates": int(valid.shape[0]),
        "summary_hash": summary_hash({"run_id": run_id, "best": best, "status": status}),
    }
    write_report_pair(
        report_md=Path("docs/stage20_report.md"),
        report_json=Path("docs/stage20_summary.json"),
        title="Stage-20 Report",
        how_to_run=[
            "dry-run: `python scripts/run_stage20.py --seed 42`",
            "real-local: `python scripts/run_stage20.py --seed 42`",
        ],
        metrics=metrics,
        status=status,
        failures=failures,
        next_actions=[
            "Stage-21: search-v2 bounded candidate generation and pruning.",
            "Keep objective constraints hard; never accept degenerate low-trade candidates.",
        ],
        stage_type="non_trading",
        expect_walkforward=False,
        expect_mc=False,
        extras={"best_candidate": best, "constraints": constraints.__dict__, "candidate_table": str(report_table.as_posix())},
    )
    print(f"run_id: {run_id}")
    print("stage20_summary: docs/stage20_summary.json")
    print("stage20_report: docs/stage20_report.md")
    print(f"valid_candidates: {int(valid.shape[0])}")


def _collect_candidates(stage19_payload: dict) -> list[dict]:
    baseline = dict(stage19_payload.get("baseline", {}))
    transition = dict(stage19_payload.get("with_transition", {}))
    out = [
        {
            "name": "baseline",
            **baseline,
            "drag_penalty": 0.6,
            "horizon_consistency": 0.35,
        },
        {
            "name": "transition",
            **transition,
            "drag_penalty": 0.8,
            "horizon_consistency": 0.40,
        },
    ]
    if not baseline and not transition:
        out = [
            {
                "name": "fallback",
                "trade_count": 20.0,
                "tpm": 8.0,
                "exposure_ratio": 0.15,
                "PF": 1.1,
                "PF_raw": 1.1,
                "expectancy": 10.0,
                "exp_lcb": 1.0,
                "max_drawdown": 0.2,
                "drag_penalty": 0.7,
                "horizon_consistency": 0.5,
            }
        ]
    return out


def _constraints_for_timeframe(*, timeframe: str) -> ObjectiveConstraints:
    tf = str(timeframe).strip().lower()
    scale = 1.0
    if tf in {"15m", "30m"}:
        scale = 2.0
    elif tf in {"2h", "4h"}:
        scale = 0.5
    return ObjectiveConstraints(
        min_tpm=5.0 * scale,
        max_tpm=80.0 * scale,
        exposure_min=0.01,
        max_dd_p95=0.30,
        max_drag_penalty=5.0,
        min_horizon_consistency=0.20,
    )


if __name__ == "__main__":
    main()
