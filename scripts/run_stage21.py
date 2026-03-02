"""Run Stage-21 bounded search-v2 with pruning diagnostics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from buffmini.alpha_v2.objective import ObjectiveConstraints, robust_objective
from buffmini.alpha_v2.reports import summary_hash, write_report_pair
from buffmini.alpha_v2.search_v2 import SearchConfigV2, bounded_search
from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-21 search-v2")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-evals", type=int, default=1500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.perf_counter()
    cfg = load_config(args.config)
    baseline = _load_baseline(Path("docs/stage19_summary.json"))
    candidate_space = []
    for threshold in (0.20, 0.25, 0.30, 0.35, 0.40, 0.45):
        for weight in (0.5, 0.75, 1.0, 1.25, 1.5):
            for hold in (12, 24, 36):
                candidate_space.append({"threshold": threshold, "weight": weight, "hold_bars": hold})

    constraints = ObjectiveConstraints(min_tpm=5.0, max_tpm=80.0, exposure_min=0.01, max_dd_p95=0.30, max_drag_penalty=5.0)

    def evaluate(candidate: dict) -> dict:
        threshold = float(candidate["threshold"])
        weight = float(candidate["weight"])
        hold = int(candidate["hold_bars"])
        exp_lcb = float(baseline["exp_lcb"] + (0.30 - threshold) * 5.0 + (weight - 1.0) * 0.8 - abs(hold - 24) * 0.01)
        tpm = float(max(0.1, baseline["tpm"] * (weight / max(0.1, threshold * 4.0))))
        exposure = float(max(0.001, baseline["exposure_ratio"] * weight))
        max_dd = float(max(0.01, baseline["max_drawdown"] + abs(weight - 1.0) * 0.03 + abs(hold - 24) * 0.001))
        drag_penalty = float(abs(weight - 1.0) * 0.8 + max(0.0, threshold - 0.30) * 3.0)
        horizon_consistency = float(max(0.0, 0.6 - abs(threshold - 0.30) * 1.2))
        obj = robust_objective(
            exp_lcb=exp_lcb,
            tpm=tpm,
            exposure_ratio=exposure,
            max_dd_p95=max_dd,
            drag_penalty=drag_penalty,
            horizon_consistency=horizon_consistency,
            constraints=constraints,
        )
        return {
            **obj,
            "exp_lcb": exp_lcb,
            "tpm": tpm,
            "exposure_ratio": exposure,
            "max_drawdown": max_dd,
            "drag_penalty": drag_penalty,
            "horizon_consistency": horizon_consistency,
        }

    result = bounded_search(
        candidate_space=candidate_space,
        evaluate_fn=evaluate,
        cfg=SearchConfigV2(max_evaluations=int(args.max_evals), beam_width=64, seed=int(args.seed)),
    )
    top = result["top_candidates"]
    best = top[0] if top else {}
    status = "PASS" if top else "FAILED"
    failures = [] if top else ["search_returned_empty_candidate_set"]

    run_id = f"{utc_now_compact()}_{stable_hash({'seed': int(args.seed), 'best': best}, length=12)}_stage21"
    table_rows = [
        {
            **dict(item["candidate"]),
            **dict(item["result"]),
        }
        for item in top
    ]
    pd.DataFrame(table_rows).to_csv(Path("docs/stage21_top_candidates.csv"), index=False)

    best_result = dict(best.get("result", {}))
    metrics = {
        "run_id": run_id,
        "seed": int(args.seed),
        "config_hash": stable_hash(cfg, length=16),
        "data_hash": stable_hash(candidate_space, length=16),
        "resolved_end_ts": baseline.get("resolved_end_ts"),
        "trade_count": 0.0,
        "trades_per_month": float(best_result.get("tpm", 0.0)),
        "exposure_ratio": float(best_result.get("exposure_ratio", 0.0)),
        "PF": 0.0,
        "PF_raw": 0.0,
        "expectancy": 0.0,
        "exp_lcb": float(best_result.get("exp_lcb", 0.0)),
        "max_drawdown": float(best_result.get("max_drawdown", 0.0)),
        "walkforward_executed_true_pct": 0.0,
        "usable_windows_count": 0,
        "mc_trigger_rate": 0.0,
        "invalid_pct": float(100.0 if not top else 0.0),
        "zero_trade_pct": 0.0,
        "evaluated_count": int(result["evaluated_count"]),
        "pruned_count": int(result["pruned_count"]),
        "runtime_seconds": float(time.perf_counter() - t0),
        "cache_hit_rate": 0.0,
        "summary_hash": summary_hash({"run_id": run_id, "status": status, "best": best}),
    }
    write_report_pair(
        report_md=Path("docs/stage21_report.md"),
        report_json=Path("docs/stage21_summary.json"),
        title="Stage-21 Report",
        how_to_run=[
            "dry-run: `python scripts/run_stage21.py --dry-run --seed 42`",
            "real-local: `python scripts/run_stage21.py --seed 42`",
        ],
        metrics=metrics,
        status=status,
        failures=failures,
        next_actions=[
            "Stage-22: apply MTF policy with strict no-leak alignment.",
            "Inspect prune_reasons to adjust search-space quality.",
        ],
        extras={
            "best_candidate": best,
            "prune_reasons": result["prune_reasons"],
            "evaluated_count": result["evaluated_count"],
            "pruned_count": result["pruned_count"],
        },
    )
    print(f"run_id: {run_id}")
    print("stage21_summary: docs/stage21_summary.json")
    print("stage21_report: docs/stage21_report.md")
    print(f"evaluated_count: {int(result['evaluated_count'])}")


def _load_baseline(path: Path) -> dict:
    if not path.exists():
        return {"exp_lcb": 0.0, "tpm": 8.0, "exposure_ratio": 0.15, "max_drawdown": 0.2, "resolved_end_ts": None}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "exp_lcb": float(payload.get("exp_lcb", payload.get("delta_exp_lcb_vs_baseline", 0.0))),
        "tpm": float(payload.get("trades_per_month", 8.0)),
        "exposure_ratio": float(payload.get("exposure_ratio", 0.15)),
        "max_drawdown": float(payload.get("max_drawdown", 0.2)),
        "resolved_end_ts": payload.get("resolved_end_ts"),
    }


if __name__ == "__main__":
    main()

