"""Layered robustness summaries for evaluation-stage decision support."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from buffmini.validation import evaluate_candidate_walkforward


ROBUSTNESS_LEVELS = {
    0: "rejected",
    1: "mechanism_viability",
    2: "forward_plausibility",
    3: "full_robustness",
}


def evaluate_split_perturbation(
    *,
    candidate: dict[str, Any],
    config: dict[str, Any],
    symbol: str,
    frame,
    market_meta: dict[str, Any],
) -> dict[str, Any]:
    """Re-run walk-forward with perturbed split geometry as a robustness stress."""

    cfg = deepcopy(config)
    stage8 = cfg.setdefault("evaluation", {}).setdefault("stage8", {}).setdefault("walkforward_v2", {})
    train_days = int(stage8.get("train_days", 180))
    holdout_days = int(stage8.get("holdout_days", 30))
    forward_days = int(stage8.get("forward_days", 30))
    step_days = int(stage8.get("step_days", 30))
    stage8["train_days"] = max(60, int(round(train_days * 0.80)))
    stage8["holdout_days"] = max(14, int(round(holdout_days * 1.20)))
    stage8["forward_days"] = max(14, int(round(forward_days * 0.80)))
    stage8["step_days"] = max(14, int(round(step_days * 0.80)))
    result = evaluate_candidate_walkforward(
        candidate=candidate,
        config=cfg,
        symbol=symbol,
        frame=frame,
        market_meta=market_meta,
    )
    return {
        "execution_status": str(result.get("execution_status", "BLOCKED")),
        "validation_state": str(result.get("validation_state", "")),
        "decision_use_allowed": bool(result.get("decision_use_allowed", False)),
        "summary": dict(result.get("summary", {})),
        "window_metrics": list(result.get("window_metrics", [])),
        "effective_split": {
            "train_days": int(stage8["train_days"]),
            "holdout_days": int(stage8["holdout_days"]),
            "forward_days": int(stage8["forward_days"]),
            "step_days": int(stage8["step_days"]),
        },
    }


def summarize_layered_robustness(
    *,
    replay_metrics: dict[str, Any],
    walkforward_summary: dict[str, Any],
    monte_carlo: dict[str, Any],
    perturbation: dict[str, Any],
    split_perturbation: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Map validation evidence to layered robustness levels."""

    replay_gate = dict(config.get("promotion_gates", {}).get("replay", {}))
    walkforward_gate = dict(config.get("promotion_gates", {}).get("walkforward", {}))
    cross_gate = dict(config.get("promotion_gates", {}).get("cross_seed", {}))

    mechanism_ok = bool(
        int(replay_metrics.get("trade_count", 0)) > 0
        and float(replay_metrics.get("exp_lcb", -1.0)) >= float(replay_gate.get("min_exp_lcb", -0.001))
        and float(replay_metrics.get("maxDD", 1.0)) <= float(replay_gate.get("max_drawdown", 0.20))
    )
    forward_ok = bool(
        int(walkforward_summary.get("usable_windows", 0)) >= int(walkforward_gate.get("min_usable_windows", 3))
        and float(walkforward_summary.get("p10_forward_expectancy", -1.0)) >= -0.0025
        and float(walkforward_summary.get("positive_window_fraction", 0.0)) >= 0.35
        and float(walkforward_summary.get("degradation_vs_holdout", -1.0)) >= -0.01
    )
    mc_rows = list(monte_carlo.get("scenario_rows", []))
    mc_ok = bool(
        str(monte_carlo.get("execution_status", "")) == "EXECUTED"
        and float(monte_carlo.get("conservative_downside_bound", -1.0)) >= -0.03
        and float(monte_carlo.get("worst_case_drawdown_p95", 1.0)) <= 0.40
        and all(bool(row.get("passed", False)) for row in mc_rows)
    )
    perturb_rows = list(perturbation.get("rows", []))
    perturb_ok = bool(
        str(perturbation.get("execution_status", "")) == "EXECUTED"
        and int(perturbation.get("surviving_seeds", 0)) >= int(cross_gate.get("min_passing_seeds", 3))
    )
    split_summary = dict(split_perturbation.get("summary", {}))
    split_ok = bool(
        str(split_perturbation.get("execution_status", "")) == "EXECUTED"
        and int(split_summary.get("usable_windows", 0)) >= int(max(1, walkforward_gate.get("min_usable_windows", 3) - 1))
        and float(split_summary.get("p10_forward_expectancy", -1.0)) >= -0.003
    )

    level = 0
    stop_reason = "mechanism_not_viable"
    if mechanism_ok:
        level = 1
        stop_reason = "forward_plausibility_not_met"
    if mechanism_ok and forward_ok:
        level = 2
        stop_reason = "full_robustness_not_met"
    if mechanism_ok and forward_ok and mc_ok and perturb_ok and split_ok:
        level = 3
        stop_reason = ""

    return {
        "level_reached": int(level),
        "level_name": str(ROBUSTNESS_LEVELS[int(level)]),
        "stop_reason": str(stop_reason),
        "levels": {
            "level_1_mechanism_viability": bool(mechanism_ok),
            "level_2_forward_plausibility": bool(mechanism_ok and forward_ok),
            "level_3_full_robustness": bool(mechanism_ok and forward_ok and mc_ok and perturb_ok and split_ok),
        },
        "forward_metrics": {
            "worst_window_expectancy": float(walkforward_summary.get("worst_window_expectancy", 0.0)),
            "p10_forward_expectancy": float(walkforward_summary.get("p10_forward_expectancy", 0.0)),
            "p10_forward_profit_factor": float(walkforward_summary.get("p10_forward_profit_factor", 0.0)),
            "positive_window_fraction": float(walkforward_summary.get("positive_window_fraction", 0.0)),
            "dispersion": dict(walkforward_summary.get("dispersion", {})),
            "degradation_vs_holdout": float(walkforward_summary.get("degradation_vs_holdout", 0.0)),
        },
        "stress_results": {
            "monte_carlo_passed": bool(mc_ok),
            "cross_perturbation_passed": bool(perturb_ok),
            "split_perturbation_passed": bool(split_ok),
            "monte_carlo_scenarios": mc_rows,
            "cross_perturbation_rows": perturb_rows,
            "split_perturbation_summary": split_summary,
        },
    }
