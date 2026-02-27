"""Stage-3.2 leverage frontier evaluation built on Stage-3.1 Monte Carlo."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.constants import RAW_DATA_DIR, RUNS_DIR
from buffmini.portfolio.monte_carlo import load_portfolio_trades, simulate_equity_paths, summarize_mc
from buffmini.utils.hashing import stable_hash
from buffmini.utils.logging import get_logger
from buffmini.utils.time import utc_now_compact


logger = get_logger(__name__)

DEFAULT_METHODS = ["equal", "vol", "corr-min"]


def run_stage3_leverage_frontier(
    stage2_run_id: str,
    leverage_levels: list[float],
    n_paths: int = 20_000,
    bootstrap: str = "block",
    block_size_trades: int = 10,
    initial_equity: float = 10_000.0,
    ruin_dd_threshold: float = 0.5,
    max_p_ruin: float = 0.01,
    max_dd_p95: float = 0.25,
    min_return_p05: float = 0.0,
    methods: list[str] | None = None,
    seed: int = 42,
    runs_dir: Path = RUNS_DIR,
    data_dir: Path = RAW_DATA_DIR,
) -> Path:
    """Evaluate leverage frontier per Stage-2 portfolio method."""

    method_keys = [str(item).strip() for item in (methods or DEFAULT_METHODS) if str(item).strip()]
    levels = sorted({float(level) for level in leverage_levels})
    if not levels:
        raise ValueError("leverage_levels must include at least one value")
    if int(n_paths) < 1:
        raise ValueError("n_paths must be >= 1")
    if float(initial_equity) <= 0.0:
        raise ValueError("initial_equity must be > 0")
    if int(block_size_trades) < 1:
        raise ValueError("block_size_trades must be >= 1")
    if not 0.0 < float(max_p_ruin) < 1.0:
        raise ValueError("max_p_ruin must be between 0 and 1")
    if float(max_dd_p95) <= 0.0:
        raise ValueError("max_dd_p95 must be > 0")

    rows: list[dict[str, Any]] = []
    method_summaries: dict[str, dict[str, Any]] = {}

    for method_idx, method in enumerate(method_keys):
        trades = load_portfolio_trades(
            stage2_run_id=stage2_run_id,
            method=method,
            runs_dir=runs_dir,
            data_dir=data_dir,
        )
        if trades.empty:
            raise ValueError(f"No trades available for method {method} in stage2 run {stage2_run_id}")
        trade_count_source = int(len(trades))
        trade_pnls = trades["pnl"].astype(float)

        method_rows: list[dict[str, Any]] = []
        for lev_idx, leverage in enumerate(levels):
            paths = simulate_equity_paths(
                trade_pnls=trade_pnls,
                n_paths=int(n_paths),
                method=str(bootstrap).strip().lower(),
                seed=int(seed) + (method_idx * 1000) + lev_idx,
                initial_equity=float(initial_equity),
                leverage=float(leverage),
                block_size_trades=int(block_size_trades),
            )
            summary = summarize_mc(
                paths_results=paths,
                initial_equity=float(initial_equity),
                ruin_dd_threshold=float(ruin_dd_threshold),
            )
            p_ruin = float(summary["tail_probabilities"]["p_ruin"])
            dd_p95 = float(summary["max_drawdown"]["p95"])
            ret_p05 = float(summary["return_pct"]["p05"])

            pass_p_ruin = bool(p_ruin <= float(max_p_ruin))
            pass_dd = bool(dd_p95 <= float(max_dd_p95))
            pass_return = bool(ret_p05 >= float(min_return_p05))
            pass_all = bool(pass_p_ruin and pass_dd and pass_return)

            fail_constraints: list[str] = []
            if not pass_p_ruin:
                fail_constraints.append("max_p_ruin")
            if not pass_dd:
                fail_constraints.append("max_dd_p95")
            if not pass_return:
                fail_constraints.append("min_return_p05")

            row = {
                "method": method,
                "leverage": float(leverage),
                "trade_count_source": trade_count_source,
                "return_p05": ret_p05,
                "return_median": float(summary["return_pct"]["median"]),
                "return_p95": float(summary["return_pct"]["p95"]),
                "maxdd_p95": dd_p95,
                "maxdd_p99": float(summary["max_drawdown"]["p99"]),
                "p_ruin": p_ruin,
                "p_return_lt_0": float(summary["tail_probabilities"]["p_return_lt_0"]),
                "pass_all_constraints": pass_all,
                "pass_max_p_ruin": pass_p_ruin,
                "pass_max_dd_p95": pass_dd,
                "pass_min_return_p05": pass_return,
                "failed_constraints": ",".join(fail_constraints),
            }
            rows.append(row)
            method_rows.append(row)

        safe_levels = [float(item["leverage"]) for item in method_rows if bool(item["pass_all_constraints"])]
        chosen_safe = max(safe_levels) if safe_levels else None
        first_failure_row = next((item for item in method_rows if not bool(item["pass_all_constraints"])), None)
        method_summaries[method] = {
            "chosen_safe_leverage": chosen_safe,
            "first_failure_leverage": float(first_failure_row["leverage"]) if first_failure_row is not None else None,
            "first_failure_constraints": (
                str(first_failure_row["failed_constraints"]).split(",")
                if first_failure_row is not None and str(first_failure_row["failed_constraints"]).strip()
                else []
            ),
            "trade_count_source": trade_count_source,
        }

    rows_df = pd.DataFrame(rows).sort_values(["method", "leverage"]).reset_index(drop=True)
    payload = {
        "stage2_run_id": stage2_run_id,
        "seed": int(seed),
        "n_paths": int(n_paths),
        "bootstrap": str(bootstrap).strip().lower(),
        "block_size_trades": int(block_size_trades),
        "initial_equity": float(initial_equity),
        "ruin_dd_threshold": float(ruin_dd_threshold),
        "constraints": {
            "max_p_ruin": float(max_p_ruin),
            "max_dd_p95": float(max_dd_p95),
            "min_return_p05": float(min_return_p05),
        },
        "leverage_levels": [float(level) for level in levels],
        "methods": method_summaries,
        "rows": rows_df.to_dict(orient="records"),
    }
    run_hash = stable_hash(payload, length=12)
    run_id = f"{utc_now_compact()}_{run_hash}_stage3_2"
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload["run_id"] = run_id

    rows_df.to_csv(run_dir / "leverage_frontier.csv", index=False)
    (run_dir / "stage3_2_summary.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    logger.info("Saved Stage-3.2 leverage frontier artifacts to %s", run_dir)
    return run_dir

