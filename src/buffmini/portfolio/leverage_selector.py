"""Stage-3.3 automatic leverage selector with hard constraints and log-utility."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.config import compute_config_hash
from buffmini.constants import RAW_DATA_DIR, RUNS_DIR
from buffmini.portfolio.monte_carlo import (
    _load_stage2_context,
    _normalize_method_key,
    _reconstruct_method_trade_frame,
    simulate_equity_paths,
    summarize_mc,
)
from buffmini.utils.hashing import stable_hash
from buffmini.utils.logging import get_logger
from buffmini.utils.time import utc_now_compact


logger = get_logger(__name__)


def compute_log_growth(final_equity: float, initial_equity: float, epsilon: float) -> float:
    """Compute log-growth utility with numeric floor protection."""

    safe_final = max(float(final_equity), float(epsilon))
    return float(math.log(safe_final / float(initial_equity)))


def summarize_utility(
    paths_final_equity: pd.Series | list[float] | np.ndarray,
    initial_equity: float,
    epsilon: float,
) -> dict[str, float]:
    """Summarize expected log-growth utility from Monte Carlo terminal equities."""

    values = np.asarray(pd.Series(paths_final_equity, dtype=float).dropna().to_numpy(), dtype=float)
    if values.size == 0:
        raise ValueError("paths_final_equity must be non-empty")

    log_growth = np.log(np.maximum(values, float(epsilon)) / float(initial_equity))
    return {
        "expected_log_growth": float(np.mean(log_growth)),
        "log_growth_p05": float(np.quantile(log_growth, 0.05)),
        "log_growth_p50": float(np.quantile(log_growth, 0.50)),
        "log_growth_p95": float(np.quantile(log_growth, 0.95)),
        "probability_log_growth_negative": float(np.mean(log_growth < 0.0)),
    }


def evaluate_leverage_candidate(
    method: str,
    leverage: float,
    cfg: dict[str, Any],
    seed: int,
    context: Any | None = None,
    trade_pnls: pd.Series | None = None,
) -> dict[str, Any]:
    """Evaluate one method/leverage candidate using Stage-3.1 Monte Carlo primitives."""

    method_key = _normalize_method_key(method)
    runtime_cfg = deepcopy(cfg)
    resolved_context = context
    if trade_pnls is None:
        if resolved_context is None:
            resolved_context = _load_stage2_context(
                stage2_run_id=str(runtime_cfg["stage2_run_id"]),
                runs_dir=Path(runtime_cfg.get("runs_dir", RUNS_DIR)),
                data_dir=Path(runtime_cfg.get("data_dir", RAW_DATA_DIR)),
            )
        trades = _reconstruct_method_trade_frame(context=resolved_context, method=method_key)
        trade_pnls = trades["pnl"].astype(float)
    else:
        trade_pnls = pd.Series(trade_pnls, dtype=float)

    if trade_pnls.empty:
        raise ValueError(f"No trade PnL data available for method {method_key}")

    constraints = runtime_cfg["constraints"]
    utility_cfg = runtime_cfg["utility"]
    initial_equity = float(runtime_cfg["initial_equity"])
    ruin_dd_threshold = float(runtime_cfg["ruin_dd_threshold"])

    paths = simulate_equity_paths(
        trade_pnls=trade_pnls,
        n_paths=int(runtime_cfg["n_paths"]),
        method=str(runtime_cfg["bootstrap"]),
        seed=int(seed),
        initial_equity=initial_equity,
        leverage=float(leverage),
        block_size_trades=int(runtime_cfg["block_size_trades"]),
    )
    mc_summary = summarize_mc(
        paths_results=paths,
        initial_equity=initial_equity,
        ruin_dd_threshold=ruin_dd_threshold,
    )
    utility = summarize_utility(
        paths_final_equity=paths["final_equity"].astype(float),
        initial_equity=initial_equity,
        epsilon=float(utility_cfg["epsilon"]),
    )

    p_ruin = float(mc_summary["tail_probabilities"]["p_ruin"])
    dd_p95 = float(mc_summary["max_drawdown"]["p95"])
    ret_p05 = float(mc_summary["return_pct"]["p05"])
    pass_p_ruin = bool(p_ruin <= float(constraints["max_p_ruin"]))
    pass_dd = bool(dd_p95 <= float(constraints["max_dd_p95"]))
    pass_return = bool(ret_p05 >= float(constraints["min_return_p05"]))

    failed_constraints: list[str] = []
    if not pass_p_ruin:
        failed_constraints.append("max_p_ruin")
    if not pass_dd:
        failed_constraints.append("max_dd_p95")
    if not pass_return:
        failed_constraints.append("min_return_p05")

    return {
        "method": method_key,
        "leverage": float(leverage),
        "trade_count_source": int(len(trade_pnls)),
        "expected_log_growth": float(utility["expected_log_growth"]),
        "log_growth_p05": float(utility["log_growth_p05"]),
        "log_growth_p50": float(utility["log_growth_p50"]),
        "log_growth_p95": float(utility["log_growth_p95"]),
        "p_log_growth_negative": float(utility["probability_log_growth_negative"]),
        "return_p05": ret_p05,
        "return_median": float(mc_summary["return_pct"]["median"]),
        "return_p95": float(mc_summary["return_pct"]["p95"]),
        "maxdd_p95": dd_p95,
        "maxdd_p99": float(mc_summary["max_drawdown"]["p99"]),
        "p_return_lt_0": float(mc_summary["tail_probabilities"]["p_return_lt_0"]),
        "p_ruin": p_ruin,
        "pass_all_constraints": bool(pass_p_ruin and pass_dd and pass_return),
        "pass_max_p_ruin": pass_p_ruin,
        "pass_max_dd_p95": pass_dd,
        "pass_min_return_p05": pass_return,
        "failed_constraints": ",".join(failed_constraints),
    }


def choose_best_leverage(results_rows: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    """Choose best leverage for one method: feasible set + utility maximization."""

    if not results_rows:
        return {
            "status": "NO_DATA",
            "chosen_leverage": None,
            "chosen_row": None,
            "feasible_count": 0,
            "binding_constraints": [],
            "margins_to_limit": {},
            "failure_summary": {},
            "first_failure_leverage": None,
            "first_failure_constraints": [],
        }

    constraints = cfg["constraints"]
    ordered = sorted(results_rows, key=lambda row: float(row["leverage"]))
    feasible = [row for row in ordered if bool(row["pass_all_constraints"])]

    failure_counts = {
        "max_p_ruin": int(sum(not bool(row["pass_max_p_ruin"]) for row in ordered)),
        "max_dd_p95": int(sum(not bool(row["pass_max_dd_p95"]) for row in ordered)),
        "min_return_p05": int(sum(not bool(row["pass_min_return_p05"]) for row in ordered)),
    }
    first_failure = next((row for row in ordered if not bool(row["pass_all_constraints"])), None)

    if not feasible:
        return {
            "status": "NO_FEASIBLE_LEVERAGE",
            "chosen_leverage": None,
            "chosen_row": None,
            "feasible_count": 0,
            "binding_constraints": [],
            "margins_to_limit": {},
            "failure_summary": failure_counts,
            "first_failure_leverage": float(first_failure["leverage"]) if first_failure else None,
            "first_failure_constraints": (
                str(first_failure["failed_constraints"]).split(",")
                if first_failure and str(first_failure["failed_constraints"]).strip()
                else []
            ),
        }

    ranked = sorted(
        feasible,
        key=lambda row: (
            -float(row["expected_log_growth"]),
            -float(row["return_p05"]),
            float(row["maxdd_p95"]),
            float(row["p_ruin"]),
        ),
    )
    chosen = ranked[0]
    margins = {
        "max_p_ruin_margin": float(constraints["max_p_ruin"]) - float(chosen["p_ruin"]),
        "max_dd_p95_margin": float(constraints["max_dd_p95"]) - float(chosen["maxdd_p95"]),
        "min_return_p05_margin": float(chosen["return_p05"]) - float(constraints["min_return_p05"]),
    }
    normalized = {
        "max_p_ruin_margin": margins["max_p_ruin_margin"] / max(float(constraints["max_p_ruin"]), 1e-12),
        "max_dd_p95_margin": margins["max_dd_p95_margin"] / max(float(constraints["max_dd_p95"]), 1e-12),
        "min_return_p05_margin": margins["min_return_p05_margin"] / max(abs(float(constraints["min_return_p05"])), 1.0),
    }
    min_norm = min(float(value) for value in normalized.values())
    binding = [
        key.replace("_margin", "")
        for key, value in normalized.items()
        if abs(float(value) - min_norm) <= 1e-12
    ]

    return {
        "status": "OK",
        "chosen_leverage": float(chosen["leverage"]),
        "chosen_row": chosen,
        "feasible_count": int(len(feasible)),
        "binding_constraints": binding,
        "margins_to_limit": margins,
        "failure_summary": failure_counts,
        "first_failure_leverage": float(first_failure["leverage"]) if first_failure else None,
        "first_failure_constraints": (
            str(first_failure["failed_constraints"]).split(",")
            if first_failure and str(first_failure["failed_constraints"]).strip()
            else []
        ),
    }


def choose_best_method(method_summaries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Choose overall best method among methods with feasible leverage."""

    feasible = [
        {"method": method, **summary["chosen_row"]}
        for method, summary in method_summaries.items()
        if summary.get("chosen_row") is not None
    ]
    if not feasible:
        return {
            "status": "NO_FEASIBLE_METHOD",
            "method": None,
            "chosen_leverage": None,
            "chosen_row": None,
        }

    ranked = sorted(
        feasible,
        key=lambda row: (
            -float(row["expected_log_growth"]),
            -float(row["return_p05"]),
            float(row["maxdd_p95"]),
            float(row["p_ruin"]),
        ),
    )
    best = ranked[0]
    return {
        "status": "OK",
        "method": str(best["method"]),
        "chosen_leverage": float(best["leverage"]),
        "chosen_row": best,
    }


def run_stage3_leverage_selector(
    stage2_run_id: str,
    selector_cfg: dict[str, Any],
    methods: list[str] | None = None,
    leverage_levels: list[float] | None = None,
    seed: int | None = None,
    n_paths: int | None = None,
    bootstrap: str | None = None,
    block_size_trades: int | None = None,
    initial_equity: float | None = None,
    ruin_dd_threshold: float | None = None,
    max_p_ruin: float | None = None,
    max_dd_p95: float | None = None,
    min_return_p05: float | None = None,
    runs_dir: Path = RUNS_DIR,
    data_dir: Path = RAW_DATA_DIR,
    run_id: str | None = None,
    cli_command: str | None = None,
) -> Path:
    """Run Stage-3.3 leverage selection and produce audit-grade artifacts."""

    cfg = deepcopy(selector_cfg)
    cfg["stage2_run_id"] = stage2_run_id
    cfg["runs_dir"] = Path(runs_dir)
    cfg["data_dir"] = Path(data_dir)
    if methods is not None:
        cfg["methods"] = [str(item).strip() for item in methods if str(item).strip()]
    if leverage_levels is not None:
        cfg["leverage_levels"] = [float(item) for item in leverage_levels]
    if seed is not None:
        cfg["seed"] = int(seed)
    if n_paths is not None:
        cfg["n_paths"] = int(n_paths)
    if bootstrap is not None:
        cfg["bootstrap"] = str(bootstrap).strip().lower()
    if block_size_trades is not None:
        cfg["block_size_trades"] = int(block_size_trades)
    if initial_equity is not None:
        cfg["initial_equity"] = float(initial_equity)
    if ruin_dd_threshold is not None:
        cfg["ruin_dd_threshold"] = float(ruin_dd_threshold)
    if max_p_ruin is not None:
        cfg["constraints"]["max_p_ruin"] = float(max_p_ruin)
    if max_dd_p95 is not None:
        cfg["constraints"]["max_dd_p95"] = float(max_dd_p95)
    if min_return_p05 is not None:
        cfg["constraints"]["min_return_p05"] = float(min_return_p05)

    method_keys = [_normalize_method_key(item) for item in cfg["methods"]]
    leverage_list = sorted({float(item) for item in cfg["leverage_levels"]})
    if not method_keys:
        raise ValueError("Stage-3.3 requires at least one method")
    if not leverage_list:
        raise ValueError("Stage-3.3 requires at least one leverage level")

    context = _load_stage2_context(stage2_run_id=stage2_run_id, runs_dir=Path(runs_dir), data_dir=Path(data_dir))
    all_rows: list[dict[str, Any]] = []
    method_choices: dict[str, dict[str, Any]] = {}

    for method_idx, method in enumerate(method_keys):
        trades = _reconstruct_method_trade_frame(context=context, method=method)
        trade_pnls = trades["pnl"].astype(float)
        method_rows: list[dict[str, Any]] = []
        for lev_idx, leverage in enumerate(leverage_list):
            row = evaluate_leverage_candidate(
                method=method,
                leverage=float(leverage),
                cfg=cfg,
                seed=int(cfg["seed"]) + (method_idx * 1000) + lev_idx,
                context=context,
                trade_pnls=trade_pnls,
            )
            method_rows.append(row)
            all_rows.append(row)
        method_choices[method] = choose_best_leverage(method_rows, cfg)

    overall_choice = choose_best_method(method_choices)
    recommendation = (
        f"Selected `{overall_choice['method']}` at `{overall_choice['chosen_leverage']}x` leverage by expected log-growth within hard constraints."
        if overall_choice["status"] == "OK"
        else "No feasible leverage under current hard constraints; do not proceed to leverage modeling."
    )

    table_df = pd.DataFrame(all_rows).sort_values(["method", "leverage"]).reset_index(drop=True)
    feasible_df = table_df[table_df["pass_all_constraints"]].copy().reset_index(drop=True)

    summary_payload = {
        "stage2_run_id": stage2_run_id,
        "stage1_run_id": context.stage1_run_id,
        "settings": {
            "methods": method_keys,
            "leverage_levels": [float(level) for level in leverage_list],
            "seed": int(cfg["seed"]),
            "n_paths": int(cfg["n_paths"]),
            "bootstrap": str(cfg["bootstrap"]),
            "block_size_trades": int(cfg["block_size_trades"]),
            "initial_equity": float(cfg["initial_equity"]),
            "ruin_dd_threshold": float(cfg["ruin_dd_threshold"]),
            "constraints": {
                "max_p_ruin": float(cfg["constraints"]["max_p_ruin"]),
                "max_dd_p95": float(cfg["constraints"]["max_dd_p95"]),
                "min_return_p05": float(cfg["constraints"]["min_return_p05"]),
            },
            "utility": {
                "objective": str(cfg["utility"]["objective"]),
                "epsilon": float(cfg["utility"]["epsilon"]),
                "use_final_equity": bool(cfg["utility"]["use_final_equity"]),
                "penalize_dd": bool(cfg["utility"]["penalize_dd"]),
            },
            "reporting": {
                "include_per_leverage_tables": bool(cfg["reporting"]["include_per_leverage_tables"]),
                "include_binding_constraints": bool(cfg["reporting"]["include_binding_constraints"]),
            },
            "command": cli_command,
        },
        "config_hash": compute_config_hash(context.config),
        "data_hash": context.data_hash,
        "method_choices": method_choices,
        "overall_choice": overall_choice,
        "recommendation": recommendation,
    }
    run_hash = stable_hash(summary_payload, length=12)
    resolved_run_id = run_id or f"{utc_now_compact()}_{run_hash}_stage3_3_selector"
    run_dir = Path(runs_dir) / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_payload["run_id"] = resolved_run_id

    table_df.to_csv(run_dir / "selector_table.csv", index=False)
    feasible_df.to_csv(run_dir / "feasible_only.csv", index=False)
    (run_dir / "chosen_leverage_per_method.json").write_text(
        json.dumps(method_choices, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (run_dir / "chosen_overall.json").write_text(
        json.dumps(overall_choice, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (run_dir / "selector_summary.json").write_text(
        json.dumps(summary_payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    _write_selector_report(
        run_dir=run_dir,
        summary=summary_payload,
        table=table_df,
    )

    logger.info("Saved Stage-3.3 selector artifacts to %s", run_dir)
    return run_dir


def _write_selector_report(run_dir: Path, summary: dict[str, Any], table: pd.DataFrame) -> None:
    settings = summary["settings"]
    lines: list[str] = []
    lines.append("# Stage-3.3 Leverage Selector Report")
    lines.append("")
    lines.append("## Section 1 - Provenance")
    lines.append(f"- Stage-2 run_id: `{summary['stage2_run_id']}`")
    lines.append(f"- Stage-1 run_id: `{summary['stage1_run_id']}`")
    lines.append(f"- CLI command: `{settings['command']}`")
    lines.append(f"- seed: `{settings['seed']}`")
    lines.append(f"- n_paths: `{settings['n_paths']}`")
    lines.append(f"- bootstrap: `{settings['bootstrap']}`")
    lines.append(f"- block_size_trades: `{settings['block_size_trades']}`")
    lines.append(f"- leverage_levels: `{settings['leverage_levels']}`")
    lines.append(f"- constraints: `{settings['constraints']}`")
    lines.append(f"- initial_equity: `{settings['initial_equity']}`")
    lines.append(f"- ruin_dd_threshold: `{settings['ruin_dd_threshold']}`")
    lines.append(f"- config_hash: `{summary['config_hash']}`")
    lines.append(f"- data_hash: `{summary['data_hash']}`")
    lines.append("")

    lines.append("## Section 2 - Method-by-method Results")
    for method in settings["methods"]:
        method_table = table[table["method"] == method].copy().sort_values("leverage")
        lines.append(f"### {method}")
        if bool(settings["reporting"]["include_per_leverage_tables"]):
            lines.append("| leverage | exp_log_growth | lg_p05 | lg_p50 | lg_p95 | P(lg<0) | ret_p05 | ret_med | ret_p95 | maxDD_p95 | maxDD_p99 | P(ret<0) | P(ruin) | pass_all | failed_constraints |")
            lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
            for _, row in method_table.iterrows():
                lines.append(
                    f"| {float(row['leverage']):.2f} | {float(row['expected_log_growth']):.6f} | {float(row['log_growth_p05']):.6f} | "
                    f"{float(row['log_growth_p50']):.6f} | {float(row['log_growth_p95']):.6f} | {float(row['p_log_growth_negative']):.6f} | "
                    f"{float(row['return_p05']):.6f} | {float(row['return_median']):.6f} | {float(row['return_p95']):.6f} | "
                    f"{float(row['maxdd_p95']):.6f} | {float(row['maxdd_p99']):.6f} | {float(row['p_return_lt_0']):.6f} | "
                    f"{float(row['p_ruin']):.6f} | {bool(row['pass_all_constraints'])} | {str(row['failed_constraints']) or '-'} |"
                )
            lines.append("")

        feasible = method_table[method_table["pass_all_constraints"]]
        lines.append(f"- feasible_count: `{len(feasible)}`")
        if not feasible.empty:
            lines.append("- feasible_leverages: `" + ", ".join(f"{float(item):g}" for item in feasible["leverage"].tolist()) + "`")

        choice = summary["method_choices"][method]
        lines.append(f"- chosen_status: `{choice['status']}`")
        if choice["chosen_row"] is not None:
            chosen = choice["chosen_row"]
            lines.append(f"- chosen_L: `{float(chosen['leverage']):g}`")
            lines.append(f"- expected_log_growth@chosen: `{float(chosen['expected_log_growth']):.6f}`")
            if bool(settings["reporting"]["include_binding_constraints"]):
                lines.append(f"- binding_constraints: `{choice['binding_constraints']}`")
                lines.append(f"- margins_to_limit: `{choice['margins_to_limit']}`")
        else:
            lines.append(
                f"- no feasible leverage. first_failure_leverage: `{choice['first_failure_leverage']}`, "
                f"first_failure_constraints: `{choice['first_failure_constraints']}`"
            )
            lines.append(f"- failure_summary: `{choice['failure_summary']}`")
        lines.append("")

    lines.append("## Section 3 - Overall Selection")
    overall = summary["overall_choice"]
    lines.append(f"- status: `{overall['status']}`")
    if overall["status"] == "OK":
        chosen = overall["chosen_row"]
        lines.append(f"- best_method: `{overall['method']}`")
        lines.append(f"- best_leverage: `{float(overall['chosen_leverage']):g}`")
        lines.append(f"- expected_log_growth: `{float(chosen['expected_log_growth']):.6f}`")
        lines.append(
            "- rationale: selected by highest expected_log_growth within feasible set; "
            "tie-breakers are return_p05 (higher), maxDD_p95 (lower), P(ruin) (lower)."
        )
    lines.append(f"- recommendation: {summary['recommendation']}")
    lines.append("")

    lines.append("## Section 4 - Warnings")
    lines.append("- Results are conditional on historical trade-PnL distribution and block-bootstrap assumptions.")
    lines.append("- Re-evaluate periodically as market regimes shift; monitor drift in drawdown and ruin probabilities.")

    (run_dir / "selector_report.md").write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

