"""Stage-1 random-search optimization funnel."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from buffmini.backtest.engine import run_backtest
from buffmini.baselines.stage0 import generate_signals
from buffmini.config import compute_config_hash
from buffmini.constants import RAW_DATA_DIR, RUNS_DIR
from buffmini.data.features import calculate_features
from buffmini.data.storage import load_parquet
from buffmini.data.window import slice_last_n_months
from buffmini.discovery.generator import (
    Candidate,
    candidate_to_strategy_spec,
    complexity_penalty,
    perturb_candidate,
    sample_candidate,
)
from buffmini.types import ConfigDict
from buffmini.utils.hashing import stable_hash
from buffmini.utils.logging import get_logger
from buffmini.utils.time import utc_now_compact


logger = get_logger(__name__)

DEFAULT_SMALL_EXPECTANCY_THRESHOLD = 1e-6


@dataclass
class CandidateEval:
    """Candidate evaluation output."""

    candidate: Candidate
    stage: str
    score: float
    expectancy: float
    profit_factor: float
    max_drawdown: float
    trade_count: float
    final_equity: float
    return_pct: float
    complexity_penalty: float
    instability_penalty: float
    date_range: str
    rejected: bool = False
    rejection_reason: str = ""
    metrics_validation: dict[str, float | str] | None = None
    metrics_holdout: dict[str, float | str] | None = None
    metrics_combined: dict[str, float | str] | None = None
    holdout_symbol_metrics: dict[str, dict[str, float | str]] | None = None
    cagr_approx_holdout: float = 0.0
    trades_per_month_holdout: float = 0.0
    low_signal_penalty: float = 0.0
    penalty_relief_applied: bool = False
    pf_adj_holdout: float = 0.0
    exp_lcb_holdout: float = 0.0
    effective_edge: float = 0.0
    exposure_ratio: float = 0.0
    exposure_penalty: float = 0.0
    validation_exposure_ratio: float = 0.0
    validation_active_days: float = 0.0
    validation_evidence_passed: bool = False

    def to_row(self, rank: int | None = None) -> dict[str, Any]:
        row = {
            "stage": self.stage,
            "candidate_id": self.candidate.candidate_id,
            "family": self.candidate.family,
            "gating_mode": self.candidate.gating_mode,
            "exit_mode": self.candidate.exit_mode,
            "score": self.score,
            "expectancy": self.expectancy,
            "profit_factor": self.profit_factor,
            "max_drawdown": self.max_drawdown,
            "trade_count": self.trade_count,
            "final_equity": self.final_equity,
            "return_pct": self.return_pct,
            "complexity_penalty": self.complexity_penalty,
            "instability_penalty": self.instability_penalty,
            "date_range": self.date_range,
            "rejected": self.rejected,
            "rejection_reason": self.rejection_reason,
            "params": json.dumps(self.candidate.params, sort_keys=True),
        }
        if rank is not None:
            row["rank"] = int(rank)
        if self.metrics_validation is not None:
            row["trade_count_validation"] = float(self.metrics_validation["trade_count"])
            row["profit_factor_validation"] = float(self.metrics_validation["profit_factor"])
            row["expectancy_validation"] = float(self.metrics_validation["expectancy"])
        if self.metrics_holdout is not None:
            row["trade_count_holdout"] = float(self.metrics_holdout["trade_count"])
            row["profit_factor_holdout"] = float(self.metrics_holdout["profit_factor"])
            row["expectancy_holdout"] = float(self.metrics_holdout["expectancy"])
            row["return_pct_holdout"] = float(self.metrics_holdout["return_pct"])
        if self.metrics_combined is not None:
            row["max_drawdown_combined"] = float(self.metrics_combined["max_drawdown"])
        row["cagr_approx_holdout"] = float(self.cagr_approx_holdout)
        row["trades_per_month_holdout"] = float(self.trades_per_month_holdout)
        row["low_signal_penalty"] = float(self.low_signal_penalty)
        row["penalty_relief_applied"] = bool(self.penalty_relief_applied)
        row["pf_adj_holdout"] = float(self.pf_adj_holdout)
        row["exp_lcb_holdout"] = float(self.exp_lcb_holdout)
        row["effective_edge"] = float(self.effective_edge)
        row["exposure_ratio"] = float(self.exposure_ratio)
        row["exposure_penalty"] = float(self.exposure_penalty)
        row["validation_exposure_ratio"] = float(self.validation_exposure_ratio)
        row["validation_active_days"] = float(self.validation_active_days)
        row["validation_evidence_passed"] = bool(self.validation_evidence_passed)
        return row


def run_stage1_optimization(
    config: ConfigDict,
    config_path: Path,
    dry_run: bool = False,
    run_id: str | None = None,
    runs_dir: Path = RUNS_DIR,
    data_dir: Path = RAW_DATA_DIR,
    candidate_count: int | None = None,
    seed: int | None = None,
    cost_pct: float | None = None,
    stage_a_months: int | None = None,
    stage_b_months: int | None = None,
    holdout_months: int | None = None,
    docs_report_path: Path | None = None,
) -> Path:
    """Run Stage-1 random-search funnel and save artifacts."""

    started_at = time.time()

    stage1 = config["evaluation"]["stage1"]
    search_space = stage1["search_space"]
    weights = stage1["weights"]

    resolved_seed = int(seed if seed is not None else config["search"]["seed"])
    resolved_candidate_count = int(candidate_count if candidate_count is not None else stage1["candidate_count"])
    resolved_top_k = int(stage1["top_k"])
    resolved_top_m = int(stage1["top_m"])
    resolved_stage_a_months = int(stage_a_months if stage_a_months is not None else stage1["stage_a_months"])
    resolved_stage_b_months = int(stage_b_months if stage_b_months is not None else stage1["stage_b_months"])
    resolved_holdout_months = int(holdout_months if holdout_months is not None else stage1["holdout_months"])
    early_stop_patience = int(stage1["early_stop_patience"])
    min_stage_a_evals = int(stage1["min_stage_a_evals"])
    split_mode = str(stage1["split_mode"])
    min_holdout_trades = int(stage1["min_holdout_trades"])
    recent_weight = float(stage1["recent_weight"])
    small_expectancy_threshold = float(stage1.get("small_expectancy_threshold", DEFAULT_SMALL_EXPECTANCY_THRESHOLD))
    target_trades_per_month_holdout = float(stage1["target_trades_per_month_holdout"])
    low_signal_penalty_weight = float(stage1["low_signal_penalty_weight"])
    min_trades_per_month_floor = float(stage1["min_trades_per_month_floor"])
    min_validation_exposure_ratio = float(stage1["min_validation_exposure_ratio"])
    min_validation_active_days = float(stage1["min_validation_active_days"])
    allow_rare_if_high_expectancy = bool(stage1["allow_rare_if_high_expectancy"])
    rare_expectancy_threshold = float(stage1["rare_expectancy_threshold"])
    rare_penalty_relief = float(stage1["rare_penalty_relief"])

    costs = config["costs"]
    round_trip_cost_pct = float(cost_pct if cost_pct is not None else costs["round_trip_cost_pct"])
    slippage_pct = float(costs["slippage_pct"])

    risk = config["risk"]
    initial_capital = 10_000.0 * float(risk["max_concurrent_positions"])

    universe_symbols = list(config["universe"]["symbols"])
    required_symbols = [s for s in ["BTC/USDT", "ETH/USDT"] if s in universe_symbols]
    if len(required_symbols) < 2:
        raise ValueError("Stage-1 requires BTC/USDT and ETH/USDT in config.universe.symbols")

    raw_data = _load_stage1_data(
        symbols=required_symbols,
        timeframe=config["universe"]["timeframe"],
        data_dir=data_dir,
        dry_run=dry_run,
        start=config["universe"]["start"],
        seed=resolved_seed,
    )
    feature_data = {symbol: calculate_features(frame) for symbol, frame in raw_data.items()}
    data_hash = _compute_data_hash(feature_data)

    config_hash = compute_config_hash(config)
    resolved_run_id = run_id or f"{utc_now_compact()}_{config_hash}_stage1"
    run_dir = runs_dir / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    search_meta = {
        "seed": resolved_seed,
        "candidate_count": resolved_candidate_count,
        "top_k": resolved_top_k,
        "top_m": resolved_top_m,
        "stage_a_months": resolved_stage_a_months,
        "stage_b_months": resolved_stage_b_months,
        "holdout_months": resolved_holdout_months,
        "early_stop_patience": early_stop_patience,
        "min_stage_a_evals": min_stage_a_evals,
        "split_mode": split_mode,
        "min_holdout_trades": min_holdout_trades,
        "recent_weight": recent_weight,
        "small_expectancy_threshold": small_expectancy_threshold,
        "target_trades_per_month_holdout": target_trades_per_month_holdout,
        "low_signal_penalty_weight": low_signal_penalty_weight,
        "min_trades_per_month_floor": min_trades_per_month_floor,
        "min_validation_exposure_ratio": min_validation_exposure_ratio,
        "min_validation_active_days": min_validation_active_days,
        "allow_rare_if_high_expectancy": allow_rare_if_high_expectancy,
        "rare_expectancy_threshold": rare_expectancy_threshold,
        "rare_penalty_relief": rare_penalty_relief,
        "round_trip_cost_pct": round_trip_cost_pct,
        "slippage_pct": slippage_pct,
        "weights": weights,
        "search_space": search_space,
        "data_hash": data_hash,
        "dry_run": dry_run,
        "config_path": str(config_path),
    }
    with (run_dir / "search_space.json").open("w", encoding="utf-8") as handle:
        json.dump(search_meta, handle, indent=2)

    stage_a_data = {
        "BTC/USDT": _slice_by_months(feature_data["BTC/USDT"], resolved_stage_a_months),
    }
    stage_b_data = {
        symbol: _slice_by_months(feature_data[symbol], resolved_stage_b_months)
        for symbol in required_symbols
    }
    stage_c_data = {
        symbol: frame.copy().sort_values("timestamp").reset_index(drop=True)
        for symbol, frame in feature_data.items()
    }
    stage_c_splits = _build_temporal_splits(stage_c_data=stage_c_data, split_mode=split_mode)

    rng = np.random.default_rng(resolved_seed)
    leaderboard_rows: list[dict[str, Any]] = []

    # Stage A: random search on BTC, short window
    stage_a_results: list[CandidateEval] = []
    best_score = -math.inf
    no_improve = 0

    for idx in range(resolved_candidate_count):
        candidate = sample_candidate(index=idx + 1, rng=rng, search_space=search_space)
        metrics = _evaluate_candidate_metrics(
            candidate=candidate,
            data_by_symbol=stage_a_data,
            round_trip_cost_pct=round_trip_cost_pct,
            slippage_pct=slippage_pct,
            initial_capital=initial_capital,
        )

        instability = 0.0
        score = _compute_score(
            metrics=metrics,
            candidate=candidate,
            weights=weights,
            instability_penalty=instability,
        )
        result = CandidateEval(
            candidate=candidate,
            stage="A",
            score=score,
            expectancy=metrics["expectancy"],
            profit_factor=metrics["profit_factor"],
            max_drawdown=metrics["max_drawdown"],
            trade_count=metrics["trade_count"],
            final_equity=metrics["final_equity"],
            return_pct=metrics["return_pct"],
            complexity_penalty=complexity_penalty(candidate),
            instability_penalty=instability,
            date_range=metrics["date_range"],
        )
        stage_a_results.append(result)

        if score > best_score:
            best_score = score
            no_improve = 0
        else:
            no_improve += 1

        if (idx + 1) >= min_stage_a_evals and no_improve >= early_stop_patience:
            logger.info("Stage A early stop at %s candidates", idx + 1)
            break

    stage_a_sorted = sorted(stage_a_results, key=lambda x: x.score, reverse=True)
    top_k_candidates = [row.candidate for row in stage_a_sorted[:resolved_top_k]]

    for rank, item in enumerate(stage_a_sorted[:resolved_top_k], start=1):
        leaderboard_rows.append(item.to_row(rank=rank))

    # Stage B: medium data, top K re-evaluation with instability penalty
    stage_b_results: list[CandidateEval] = []
    instability_symbol = {"BTC/USDT": stage_b_data["BTC/USDT"]}

    for candidate in top_k_candidates:
        metrics = _evaluate_candidate_metrics(
            candidate=candidate,
            data_by_symbol=stage_b_data,
            round_trip_cost_pct=round_trip_cost_pct,
            slippage_pct=slippage_pct,
            initial_capital=initial_capital,
        )

        instability = _instability_penalty(
            candidate=candidate,
            base_metrics=metrics,
            data_by_symbol=instability_symbol,
            search_space=search_space,
            round_trip_cost_pct=round_trip_cost_pct,
            slippage_pct=slippage_pct,
            initial_capital=initial_capital,
            weights=weights,
        )
        score = _compute_score(metrics=metrics, candidate=candidate, weights=weights, instability_penalty=instability)

        stage_b_results.append(
            CandidateEval(
                candidate=candidate,
                stage="B",
                score=score,
                expectancy=metrics["expectancy"],
                profit_factor=metrics["profit_factor"],
                max_drawdown=metrics["max_drawdown"],
                trade_count=metrics["trade_count"],
                final_equity=metrics["final_equity"],
                return_pct=metrics["return_pct"],
                complexity_penalty=complexity_penalty(candidate),
                instability_penalty=instability,
                date_range=metrics["date_range"],
            )
        )

    stage_b_sorted = sorted(stage_b_results, key=lambda x: x.score, reverse=True)
    top_m_candidates = [row.candidate for row in stage_b_sorted[:resolved_top_m]]

    for rank, item in enumerate(stage_b_sorted[:resolved_top_m], start=1):
        leaderboard_rows.append(item.to_row(rank=rank))

    # Stage C: robust temporal evaluation (60/20/20 split).
    stage_c_results: list[CandidateEval] = []
    for candidate in top_m_candidates:
        temporal_metrics = _evaluate_temporal_candidate_metrics(
            candidate=candidate,
            splits=stage_c_splits,
            round_trip_cost_pct=round_trip_cost_pct,
            slippage_pct=slippage_pct,
            initial_capital=initial_capital,
        )
        validation_metrics = temporal_metrics["validation"]
        holdout_metrics = temporal_metrics["holdout"]
        combined_metrics = temporal_metrics["combined"]
        holdout_symbol_metrics = temporal_metrics["holdout_by_symbol"]

        validation_exposure_ratio = float(validation_metrics["exposure_ratio"])
        validation_active_days = float(validation_metrics["active_days"])
        validation_evidence_passed = _passes_validation_evidence(
            validation_exposure_ratio=validation_exposure_ratio,
            validation_active_days=validation_active_days,
            min_validation_exposure_ratio=min_validation_exposure_ratio,
            min_validation_active_days=min_validation_active_days,
        )

        pf_adj_holdout = _compute_pf_adjusted(
            profit_factor_holdout=float(holdout_metrics["profit_factor"]),
            holdout_trades=float(holdout_metrics["trade_count"]),
        )
        exp_lcb_holdout = _compute_expectancy_lcb(
            mean_holdout=float(holdout_metrics["expectancy"]),
            std_holdout=float(holdout_metrics["expectancy_std"]),
            holdout_trades=float(holdout_metrics["trade_count"]),
        )
        trades_per_month_holdout = _compute_trades_per_month(
            trade_count=float(holdout_metrics["trade_count"]),
            duration_days=float(holdout_metrics["duration_days"]),
        )
        low_signal_penalty, penalty_relief_applied = _compute_low_signal_penalty(
            trades_per_month_holdout=trades_per_month_holdout,
            target_trades_per_month_holdout=target_trades_per_month_holdout,
            expectancy_holdout=float(holdout_metrics["expectancy"]),
            allow_rare_if_high_expectancy=allow_rare_if_high_expectancy,
            rare_expectancy_threshold=rare_expectancy_threshold,
            rare_penalty_relief=rare_penalty_relief,
        )
        effective_edge = float(exp_lcb_holdout) * float(trades_per_month_holdout)
        exposure_ratio = float(holdout_metrics["exposure_ratio"])
        exposure_penalty = _compute_exposure_penalty(exposure_ratio)

        rejection_reason = _candidate_rejection_reason(
            holdout_metrics=holdout_metrics,
            trades_per_month_holdout=trades_per_month_holdout,
            min_trades_per_month_floor=min_trades_per_month_floor,
            allow_rare_if_high_expectancy=allow_rare_if_high_expectancy,
            rare_expectancy_threshold=rare_expectancy_threshold,
            validation_evidence_passed=validation_evidence_passed,
        )
        rejected = bool(rejection_reason)

        base_score = _compute_temporal_score(
            effective_edge=effective_edge,
            max_drawdown_holdout=float(holdout_metrics["max_drawdown"]),
            complexity=complexity_penalty(candidate),
            instability=0.0,
            exposure_penalty=exposure_penalty,
        )

        instability = 0.0
        score = -1e9 if rejected else base_score
        if not rejected:
            instability = _instability_penalty_temporal(
                candidate=candidate,
                base_score=base_score,
                splits=stage_c_splits,
                search_space=search_space,
                round_trip_cost_pct=round_trip_cost_pct,
                slippage_pct=slippage_pct,
                initial_capital=initial_capital,
            )
            score = _compute_temporal_score(
                effective_edge=effective_edge,
                max_drawdown_holdout=float(holdout_metrics["max_drawdown"]),
                complexity=complexity_penalty(candidate),
                instability=instability,
                exposure_penalty=exposure_penalty,
            )

        stage_c_results.append(
            CandidateEval(
                candidate=candidate,
                stage="C",
                score=score,
                expectancy=float(holdout_metrics["expectancy"]),
                profit_factor=float(holdout_metrics["profit_factor"]),
                max_drawdown=float(combined_metrics["max_drawdown"]),
                trade_count=float(holdout_metrics["trade_count"]),
                final_equity=float(holdout_metrics["final_equity"]),
                return_pct=float(holdout_metrics["return_pct"]),
                complexity_penalty=complexity_penalty(candidate),
                instability_penalty=instability,
                date_range=str(holdout_metrics["date_range"]),
                rejected=rejected,
                rejection_reason=rejection_reason,
                metrics_validation=validation_metrics,
                metrics_holdout=holdout_metrics,
                metrics_combined=combined_metrics,
                holdout_symbol_metrics=holdout_symbol_metrics,
                cagr_approx_holdout=_cagr_from_metrics(holdout_metrics),
                trades_per_month_holdout=trades_per_month_holdout,
                low_signal_penalty=low_signal_penalty,
                penalty_relief_applied=penalty_relief_applied,
                pf_adj_holdout=pf_adj_holdout,
                exp_lcb_holdout=exp_lcb_holdout,
                effective_edge=effective_edge,
                exposure_ratio=exposure_ratio,
                exposure_penalty=exposure_penalty,
                validation_exposure_ratio=validation_exposure_ratio,
                validation_active_days=validation_active_days,
                validation_evidence_passed=validation_evidence_passed,
            )
        )

    stage_c_sorted = sorted(stage_c_results, key=lambda x: x.score, reverse=True)
    accepted = [row for row in stage_c_sorted if not row.rejected]
    rejected_rows = [row for row in stage_c_sorted if row.rejected]
    top_three = accepted[:3]
    if len(top_three) < 3:
        top_three.extend(rejected_rows[: 3 - len(top_three)])
    top_five = stage_c_sorted[:5]

    for rank, item in enumerate(stage_c_sorted, start=1):
        leaderboard_rows.append(item.to_row(rank=rank))

    leaderboard = pd.DataFrame(leaderboard_rows)
    leaderboard.to_csv(run_dir / "leaderboard.csv", index=False)

    strategies_payload: list[dict[str, Any]] = []
    for rank, item in enumerate(top_three, start=1):
        spec = candidate_to_strategy_spec(item.candidate)
        strategies_payload.append(
            {
                "rank": rank,
                "candidate_id": item.candidate.candidate_id,
                "family": item.candidate.family,
                "strategy_name": spec.name,
                "gating_mode": item.candidate.gating_mode,
                "exit_mode": item.candidate.exit_mode,
                "rules": {
                    "entry": spec.entry_rules,
                    "exit": spec.exit_rules,
                },
                "parameters": item.candidate.params,
                "rejected": item.rejected,
                "rejection_reason": item.rejection_reason,
                "metrics_validation": {
                    "profit_factor": float(item.metrics_validation["profit_factor"]) if item.metrics_validation else 0.0,
                    "expectancy": float(item.metrics_validation["expectancy"]) if item.metrics_validation else 0.0,
                    "trade_count": float(item.metrics_validation["trade_count"]) if item.metrics_validation else 0.0,
                    "exposure_ratio": float(item.metrics_validation["exposure_ratio"]) if item.metrics_validation else 0.0,
                    "active_days": float(item.metrics_validation["active_days"]) if item.metrics_validation else 0.0,
                    "evidence_passed": bool(item.validation_evidence_passed),
                    "date_range": str(item.metrics_validation["date_range"]) if item.metrics_validation else "n/a",
                },
                "validation_evidence": {
                    "exposure_ratio": float(item.validation_exposure_ratio),
                    "active_days": float(item.validation_active_days),
                    "passed": bool(item.validation_evidence_passed),
                },
                "metrics_holdout": {
                    "score": item.score,
                    "profit_factor": item.profit_factor,
                    "pf_adj": item.pf_adj_holdout,
                    "expectancy": item.expectancy,
                    "exp_lcb": item.exp_lcb_holdout,
                    "effective_edge": item.effective_edge,
                    "max_drawdown": float(item.metrics_combined["max_drawdown"]) if item.metrics_combined else item.max_drawdown,
                    "trade_count": item.trade_count,
                    "trades_per_month": item.trades_per_month_holdout,
                    "exposure_ratio": item.exposure_ratio,
                    "exposure_penalty": item.exposure_penalty,
                    "low_signal_penalty": item.low_signal_penalty,
                    "penalty_relief_applied": item.penalty_relief_applied,
                    "final_equity": item.final_equity,
                    "return_pct": item.return_pct,
                    "date_range": item.date_range,
                    "cagr_approx": item.cagr_approx_holdout,
                },
                "metrics_combined": {
                    "max_drawdown": float(item.metrics_combined["max_drawdown"]) if item.metrics_combined else 0.0,
                    "trade_count": float(item.metrics_combined["trade_count"]) if item.metrics_combined else 0.0,
                    "date_range": str(item.metrics_combined["date_range"]) if item.metrics_combined else "n/a",
                },
                "metrics_holdout_by_symbol": item.holdout_symbol_metrics or {},
            }
        )

    with (run_dir / "strategies.json").open("w", encoding="utf-8") as handle:
        json.dump(strategies_payload, handle, indent=2)

    stage_a_trade_counts = [float(row.trade_count) for row in stage_a_results]
    stage_b_trade_counts = [float(row.trade_count) for row in stage_b_results]
    stage_c_trade_counts = [
        float(row.metrics_combined["trade_count"]) if row.metrics_combined else float(row.trade_count)
        for row in stage_c_results
    ]
    diagnostics = {
        "run_id": resolved_run_id,
        "seed": resolved_seed,
        "split_mode": split_mode,
        "stages": {
            "A": _build_stage_diagnostics(stage_a_trade_counts),
            "B": _build_stage_diagnostics(stage_b_trade_counts),
            "C": _build_stage_diagnostics(stage_c_trade_counts),
        },
    }
    stage_a_warning = diagnostics["stages"]["A"]["avg_trades_per_candidate"] < 30.0
    diagnostics["warnings"] = []
    if stage_a_warning:
        diagnostics["warnings"].append("Search space too restrictive; insufficient trade sampling.")

    with (run_dir / "diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)

    best_accepted_payload: dict[str, Any] | None = None
    if accepted:
        best_acc = accepted[0]
        best_acc_spec = candidate_to_strategy_spec(best_acc.candidate)
        best_accepted_payload = {
            "candidate_id": best_acc.candidate.candidate_id,
            "family": best_acc.candidate.family,
            "strategy_name": best_acc_spec.name,
            "gating_mode": best_acc.candidate.gating_mode,
            "exit_mode": best_acc.candidate.exit_mode,
            "score": float(best_acc.score),
            "validation_evidence": {
                "exposure_ratio": float(best_acc.validation_exposure_ratio),
                "active_days": float(best_acc.validation_active_days),
                "passed": bool(best_acc.validation_evidence_passed),
            },
            "metrics_holdout": {
                "profit_factor": float(best_acc.profit_factor),
                "pf_adj": float(best_acc.pf_adj_holdout),
                "expectancy": float(best_acc.expectancy),
                "exp_lcb": float(best_acc.exp_lcb_holdout),
                "effective_edge": float(best_acc.effective_edge),
                "trade_count": float(best_acc.trade_count),
                "trades_per_month": float(best_acc.trades_per_month_holdout),
                "exposure_ratio": float(best_acc.exposure_ratio),
                "exposure_penalty": float(best_acc.exposure_penalty),
                "low_signal_penalty": float(best_acc.low_signal_penalty),
            },
            "metrics_holdout_by_symbol": best_acc.holdout_symbol_metrics or {},
        }

    duration_sec = time.time() - started_at
    summary = {
        "run_id": resolved_run_id,
        "stage_version": "stage1",
        "seed": resolved_seed,
        "config_hash": config_hash,
        "data_hash": data_hash,
        "candidate_count_stage_a": len(stage_a_results),
        "candidate_count_stage_b": len(stage_b_results),
        "candidate_count_stage_c": len(stage_c_results),
        "accepted_stage_c_count": len(accepted),
        "rejected_stage_c_count": len(rejected_rows),
        "rejected_due_validation_evidence_count": sum(
            1 for row in stage_c_results if row.rejection_reason == "insufficient_validation_evidence"
        ),
        "top_k": resolved_top_k,
        "top_m": resolved_top_m,
        "top_n": len(top_three),
        "round_trip_cost_pct": round_trip_cost_pct,
        "stage_a_months": resolved_stage_a_months,
        "stage_b_months": resolved_stage_b_months,
        "holdout_months": resolved_holdout_months,
        "split_mode": split_mode,
        "min_holdout_trades": min_holdout_trades,
        "recent_weight": recent_weight,
        "target_trades_per_month_holdout": target_trades_per_month_holdout,
        "low_signal_penalty_weight": low_signal_penalty_weight,
        "min_trades_per_month_floor": min_trades_per_month_floor,
        "min_validation_exposure_ratio": min_validation_exposure_ratio,
        "min_validation_active_days": min_validation_active_days,
        "allow_rare_if_high_expectancy": allow_rare_if_high_expectancy,
        "rare_expectancy_threshold": rare_expectancy_threshold,
        "rare_penalty_relief": rare_penalty_relief,
        "runtime_seconds": duration_sec,
        "dry_run": dry_run,
        "diagnostics": diagnostics,
        "any_holdout_pf_expectancy_positive_raw": any(
            row.metrics_holdout is not None
            and float(row.metrics_holdout["profit_factor"]) > 1.0
            and float(row.metrics_holdout["expectancy"]) > 0.0
            for row in stage_c_sorted
        ),
        "any_holdout_pf_expectancy_positive_accepted": any(
            row.metrics_holdout is not None
            and float(row.metrics_holdout["profit_factor"]) > 1.0
            and float(row.metrics_holdout["expectancy"]) > 0.0
            and not row.rejected
            for row in stage_c_sorted
        ),
        "best": strategies_payload[0] if strategies_payload else None,
        "best_accepted": best_accepted_payload,
    }
    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    _write_stage1_report(
        run_dir=run_dir,
        summary=summary,
        top_three=strategies_payload,
        docs_report_path=docs_report_path,
    )
    docs_base = docs_report_path or (Path("docs") / "stage1_auto_optimization_report.md")
    _write_stage1_real_data_reports(
        run_dir=run_dir,
        summary=summary,
        top_five_rows=top_five,
        docs_md_path=docs_base.parent / "stage1_real_data_report.md",
        docs_json_path=docs_base.parent / "stage1_real_data_report.json",
    )
    _write_stage1_diagnostics_report(
        run_dir=run_dir,
        diagnostics=diagnostics,
        docs_md_path=docs_base.parent / "stage1_diagnostics.md",
    )

    if stage_a_warning:
        logger.warning("Search space too restrictive; insufficient trade sampling.")

    if strategies_payload:
        best = strategies_payload[0]
        best_holdout = best["metrics_holdout"]
        logger.info(
            "Stage-1 best candidate: %s | PF_holdout=%.4f | expectancy_holdout=%.4f | max_dd_combined=%.4f",
            best["candidate_id"],
            float(best_holdout["profit_factor"]),
            float(best_holdout["expectancy"]),
            float(best_holdout["max_drawdown"]),
        )
    if best_accepted_payload is not None:
        best_acc_holdout = best_accepted_payload["metrics_holdout"]
        logger.info(
            "Stage-1 best accepted: %s | pf_adj_holdout=%.4f | exp_lcb_holdout=%.4f | tpm_holdout=%.4f | exposure_ratio=%.4f | score=%.4f",
            best_accepted_payload["candidate_id"],
            float(best_acc_holdout["pf_adj"]),
            float(best_acc_holdout["exp_lcb"]),
            float(best_acc_holdout["trades_per_month"]),
            float(best_acc_holdout["exposure_ratio"]),
            float(best_accepted_payload["score"]),
        )
    logger.info("Saved Stage-1 artifacts to %s", run_dir)
    return run_dir


def _evaluate_temporal_candidate_metrics(
    candidate: Candidate,
    splits: dict[str, dict[str, pd.DataFrame]],
    round_trip_cost_pct: float,
    slippage_pct: float,
    initial_capital: float,
) -> dict[str, dict[str, float | str]]:
    validation_metrics = _evaluate_candidate_metrics(
        candidate=candidate,
        data_by_symbol=splits["validation"],
        round_trip_cost_pct=round_trip_cost_pct,
        slippage_pct=slippage_pct,
        initial_capital=initial_capital,
    )
    holdout_metrics = _evaluate_candidate_metrics(
        candidate=candidate,
        data_by_symbol=splits["holdout"],
        round_trip_cost_pct=round_trip_cost_pct,
        slippage_pct=slippage_pct,
        initial_capital=initial_capital,
    )
    combined_metrics = _evaluate_candidate_metrics(
        candidate=candidate,
        data_by_symbol=splits["combined"],
        round_trip_cost_pct=round_trip_cost_pct,
        slippage_pct=slippage_pct,
        initial_capital=initial_capital,
    )
    return {
        "validation": validation_metrics,
        "holdout": holdout_metrics,
        "combined": combined_metrics,
        "validation_by_symbol": validation_metrics.get("symbol_metrics", {}),
        "holdout_by_symbol": holdout_metrics.get("symbol_metrics", {}),
        "combined_by_symbol": combined_metrics.get("symbol_metrics", {}),
    }


def _build_temporal_splits(
    stage_c_data: dict[str, pd.DataFrame],
    split_mode: str,
) -> dict[str, dict[str, pd.DataFrame]]:
    if split_mode != "60_20_20":
        raise ValueError(f"Unsupported stage1 split_mode: {split_mode}")

    train: dict[str, pd.DataFrame] = {}
    validation: dict[str, pd.DataFrame] = {}
    holdout: dict[str, pd.DataFrame] = {}
    combined: dict[str, pd.DataFrame] = {}

    for symbol, frame in stage_c_data.items():
        train_frame, validation_frame, holdout_frame = _split_symbol_60_20_20(frame)
        train[symbol] = train_frame
        validation[symbol] = validation_frame
        holdout[symbol] = holdout_frame
        combined[symbol] = pd.concat([validation_frame, holdout_frame], axis=0, ignore_index=True)

    return {
        "train": train,
        "validation": validation,
        "holdout": holdout,
        "combined": combined,
    }


def _split_symbol_60_20_20(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = frame.copy().sort_values("timestamp").reset_index(drop=True)
    n = len(data)
    if n < 10:
        raise ValueError("Stage-1 requires at least 10 rows per symbol for 60/20/20 split")

    train_end = int(math.floor(n * 0.60))
    val_end = int(math.floor(n * 0.80))

    train_end = max(1, min(train_end, n - 2))
    val_end = max(train_end + 1, min(val_end, n - 1))

    train = data.iloc[:train_end].reset_index(drop=True)
    validation = data.iloc[train_end:val_end].reset_index(drop=True)
    holdout = data.iloc[val_end:].reset_index(drop=True)

    if validation.empty or holdout.empty:
        split_a = max(1, n - 2)
        split_b = n - 1
        train = data.iloc[:split_a].reset_index(drop=True)
        validation = data.iloc[split_a:split_b].reset_index(drop=True)
        holdout = data.iloc[split_b:].reset_index(drop=True)

    return train, validation, holdout


def _candidate_rejection_reason(
    holdout_metrics: dict[str, float | str],
    trades_per_month_holdout: float,
    min_trades_per_month_floor: float,
    allow_rare_if_high_expectancy: bool,
    rare_expectancy_threshold: float,
    validation_evidence_passed: bool,
) -> str:
    if not validation_evidence_passed:
        return "insufficient_validation_evidence"

    expectancy_holdout = float(holdout_metrics["expectancy"])
    extremely_high_expectancy = (
        allow_rare_if_high_expectancy and expectancy_holdout >= float(rare_expectancy_threshold)
    )
    if float(trades_per_month_holdout) < float(min_trades_per_month_floor) and not extremely_high_expectancy:
        return "degenerate_low_trades_per_month"
    return ""


def _passes_validation_evidence(
    validation_exposure_ratio: float,
    validation_active_days: float,
    min_validation_exposure_ratio: float,
    min_validation_active_days: float,
) -> bool:
    return (
        float(validation_exposure_ratio) >= float(min_validation_exposure_ratio)
        or float(validation_active_days) >= float(min_validation_active_days)
    )


def _compute_trades_per_month(
    trade_count: float,
    duration_days: float,
) -> float:
    days = float(duration_days)
    if days <= 0:
        return 0.0
    return float(trade_count) / (days / 30.0)


def _compute_pf_adjusted(
    profit_factor_holdout: float,
    holdout_trades: float,
) -> float:
    trades = max(0.0, float(holdout_trades))
    pf = _clamp(_finite_or_default(float(profit_factor_holdout), default=1.0), 0.0, 10.0)
    return 1.0 + (pf - 1.0) * (trades / (trades + 50.0))


def _compute_expectancy_lcb(
    mean_holdout: float,
    std_holdout: float,
    holdout_trades: float,
) -> float:
    mean_value = _finite_or_default(float(mean_holdout), default=-1_000.0)
    std_value = max(0.0, _finite_or_default(float(std_holdout), default=0.0))
    n = max(1.0, float(holdout_trades))
    return float(mean_value - (1.0 * std_value / math.sqrt(n)))


def _compute_exposure_penalty(
    exposure_ratio: float,
    threshold: float = 0.02,
) -> float:
    ratio = max(0.0, _finite_or_default(float(exposure_ratio), default=0.0))
    if ratio >= float(threshold):
        return 0.0
    gap = (float(threshold) - ratio) / float(threshold)
    return 5.0 * gap


def _compute_low_signal_penalty(
    trades_per_month_holdout: float,
    target_trades_per_month_holdout: float,
    expectancy_holdout: float,
    allow_rare_if_high_expectancy: bool,
    rare_expectancy_threshold: float,
    rare_penalty_relief: float,
) -> tuple[float, bool]:
    target = max(float(target_trades_per_month_holdout), 1e-9)
    tpm = float(trades_per_month_holdout)
    if tpm >= target:
        return 0.0, False

    penalty = (target - tpm) / target
    relief_applied = False
    if allow_rare_if_high_expectancy and float(expectancy_holdout) >= float(rare_expectancy_threshold):
        penalty *= float(rare_penalty_relief)
        relief_applied = True
    return float(max(0.0, penalty)), relief_applied


def _compute_temporal_score(
    effective_edge: float,
    max_drawdown_holdout: float,
    complexity: float,
    instability: float,
    exposure_penalty: float,
) -> float:
    edge = _finite_or_default(effective_edge, default=-1_000.0)
    dd_holdout = _clamp(_finite_or_default(max_drawdown_holdout, default=1.0), 0.0, 1.0)
    complexity_v = _finite_or_default(complexity, default=1.0)
    instability_v = _finite_or_default(instability, default=1.0)
    exposure_v = max(0.0, _finite_or_default(exposure_penalty, default=0.0))

    return (
        2.0 * edge
        - 1.0 * dd_holdout
        - 0.5 * complexity_v
        - 0.5 * instability_v
        - exposure_v
    )


def _instability_penalty_temporal(
    candidate: Candidate,
    base_score: float,
    splits: dict[str, dict[str, pd.DataFrame]],
    search_space: dict[str, Any],
    round_trip_cost_pct: float,
    slippage_pct: float,
    initial_capital: float,
) -> float:
    perturbed = perturb_candidate(candidate=candidate, search_space=search_space, pct=0.1)
    if not perturbed:
        return 0.0

    variant_scores: list[float] = []
    for variant in perturbed:
        temporal = _evaluate_temporal_candidate_metrics(
            candidate=variant,
            splits=splits,
            round_trip_cost_pct=round_trip_cost_pct,
            slippage_pct=slippage_pct,
            initial_capital=initial_capital,
        )
        score = _compute_temporal_score(
            effective_edge=_compute_expectancy_lcb(
                mean_holdout=float(temporal["holdout"]["expectancy"]),
                std_holdout=float(temporal["holdout"]["expectancy_std"]),
                holdout_trades=float(temporal["holdout"]["trade_count"]),
            )
            * _compute_trades_per_month(
                trade_count=float(temporal["holdout"]["trade_count"]),
                duration_days=float(temporal["holdout"]["duration_days"]),
            ),
            max_drawdown_holdout=float(temporal["holdout"]["max_drawdown"]),
            complexity=complexity_penalty(variant),
            instability=0.0,
            exposure_penalty=_compute_exposure_penalty(float(temporal["holdout"]["exposure_ratio"])),
        )
        variant_scores.append(score)

    mean_variant_score = float(np.mean(variant_scores)) if variant_scores else float(base_score)
    return max(0.0, float(base_score) - mean_variant_score)


def _cagr_from_metrics(metrics: dict[str, float | str]) -> float:
    duration_days = float(metrics.get("duration_days", 0.0))
    if duration_days <= 0:
        return 0.0

    duration_years = duration_days / 365.25
    if duration_years <= 0:
        return 0.0

    return_pct = float(metrics.get("return_pct", 0.0))
    growth = 1.0 + return_pct
    if growth <= 0:
        return -1.0

    return float(growth ** (1.0 / duration_years) - 1.0)


def _evaluate_candidate_metrics(
    candidate: Candidate,
    data_by_symbol: dict[str, pd.DataFrame],
    round_trip_cost_pct: float,
    slippage_pct: float,
    initial_capital: float,
) -> dict[str, float | str]:
    spec = candidate_to_strategy_spec(candidate)

    expectancy_list: list[float] = []
    trade_pnl_values: list[float] = []
    pf_list: list[float] = []
    dd_list: list[float] = []
    trade_count_total = 0.0
    final_equity_list: list[float] = []
    total_bars_in_position = 0.0
    total_bars = 0.0
    symbol_metrics: dict[str, dict[str, float | str]] = {}
    min_ts: pd.Timestamp | None = None
    max_ts: pd.Timestamp | None = None

    for symbol, frame in data_by_symbol.items():
        if frame.empty:
            continue

        total_bars += float(len(frame))

        timestamps = pd.to_datetime(frame["timestamp"], utc=True)
        current_min = timestamps.iloc[0]
        current_max = timestamps.iloc[-1]
        min_ts = current_min if min_ts is None else min(min_ts, current_min)
        max_ts = current_max if max_ts is None else max(max_ts, current_max)

        eval_frame = frame.copy()
        eval_frame["signal"] = generate_signals(eval_frame, spec, gating_mode=candidate.gating_mode)

        result = run_backtest(
            frame=eval_frame,
            strategy_name=spec.name,
            symbol=symbol,
            stop_atr_multiple=float(candidate.params["atr_sl_multiplier"]),
            take_profit_atr_multiple=float(candidate.params["atr_tp_multiplier"]),
            max_hold_bars=int(candidate.params["max_holding_bars"]),
            round_trip_cost_pct=float(round_trip_cost_pct),
            slippage_pct=float(slippage_pct),
            initial_capital=float(initial_capital),
            exit_mode=candidate.exit_mode,
            trailing_atr_k=float(candidate.params["trailing_atr_k"]),
        )

        expectancy_list.append(float(result.metrics["expectancy"]))
        pf_list.append(float(result.metrics["profit_factor"]))
        dd_list.append(float(result.metrics["max_drawdown"]))
        trade_count_total += float(result.metrics["trade_count"])
        symbol_final_equity = (
            float(result.equity_curve["equity"].iloc[-1])
            if not result.equity_curve.empty
            else float(initial_capital)
        )
        symbol_metrics[symbol] = {
            "profit_factor": _clamp(_finite_or_default(float(result.metrics["profit_factor"]), default=10.0), 0.0, 10.0),
            "return_pct": float((symbol_final_equity - float(initial_capital)) / float(initial_capital)),
            "max_drawdown": float(result.metrics["max_drawdown"]),
            "trade_count": float(result.metrics["trade_count"]),
        }
        if not result.trades.empty:
            trade_pnl_values.extend(result.trades["pnl"].astype(float).tolist())
            if "bars_held" in result.trades.columns:
                total_bars_in_position += float(result.trades["bars_held"].astype(float).clip(lower=0).sum())

        if not result.equity_curve.empty:
            final_equity_list.append(float(result.equity_curve["equity"].iloc[-1]))

    if not expectancy_list:
        return {
            "expectancy": -1e9,
            "profit_factor": 0.0,
            "max_drawdown": 1.0,
            "trade_count": 0.0,
            "final_equity": 0.0,
            "return_pct": -1.0,
            "duration_days": 0.0,
            "expectancy_std": 0.0,
            "exposure_ratio": 0.0,
            "total_bars_in_position": 0.0,
            "total_bars": 0.0,
            "active_days": 0.0,
            "symbol_metrics": {},
            "date_range": "n/a",
        }

    final_equity = float(np.mean(final_equity_list)) if final_equity_list else 0.0
    return_pct = (final_equity - float(initial_capital)) / float(initial_capital)
    date_range = f"{min_ts.isoformat()}..{max_ts.isoformat()}" if min_ts is not None and max_ts is not None else "n/a"
    duration_days = (
        float((max_ts - min_ts).total_seconds() / 86400.0)
        if min_ts is not None and max_ts is not None
        else 0.0
    )

    mean_profit_factor = _clamp(_finite_or_default(float(np.mean(pf_list)), default=10.0), 0.0, 10.0)
    expectancy_mean = (
        float(np.mean(trade_pnl_values))
        if trade_pnl_values
        else float(np.mean(expectancy_list))
    )
    expectancy_std = float(np.std(trade_pnl_values, ddof=0)) if trade_pnl_values else 0.0
    exposure_ratio = (float(total_bars_in_position) / float(total_bars)) if total_bars > 0 else 0.0
    active_days = float(total_bars_in_position / 24.0)

    return {
        "expectancy": expectancy_mean,
        "profit_factor": mean_profit_factor,
        "max_drawdown": float(np.mean(dd_list)),
        "trade_count": float(trade_count_total),
        "final_equity": final_equity,
        "return_pct": float(return_pct),
        "duration_days": duration_days,
        "expectancy_std": expectancy_std,
        "exposure_ratio": exposure_ratio,
        "total_bars_in_position": float(total_bars_in_position),
        "total_bars": float(total_bars),
        "active_days": active_days,
        "symbol_metrics": symbol_metrics,
        "date_range": date_range,
    }


def _compute_score(
    metrics: dict[str, float | str],
    candidate: Candidate,
    weights: dict[str, float],
    instability_penalty: float,
) -> float:
    return _raw_score(
        expectancy=float(metrics["expectancy"]),
        profit_factor=float(metrics["profit_factor"]),
        max_drawdown=float(metrics["max_drawdown"]),
        complexity=complexity_penalty(candidate),
        instability=float(instability_penalty),
        weights=weights,
    )


def _raw_score(
    expectancy: float,
    profit_factor: float,
    max_drawdown: float,
    complexity: float,
    instability: float,
    weights: dict[str, float],
) -> float:
    exp = float(expectancy)
    if not math.isfinite(exp):
        exp = -1_000.0

    pf = float(profit_factor)
    if not math.isfinite(pf):
        pf = 10.0
    pf = min(max(pf, 1e-6), 10.0)
    pf_log = math.log(pf)

    dd = float(max_drawdown)
    if not math.isfinite(dd):
        dd = 1.0
    dd = min(max(dd, 0.0), 1.0)

    comp = float(complexity)
    if not math.isfinite(comp):
        comp = 1.0

    instab = float(instability)
    if not math.isfinite(instab):
        instab = 1.0

    return (
        float(weights["expectancy"]) * exp
        + float(weights["log_profit_factor"]) * pf_log
        - float(weights["max_drawdown"]) * dd
        - float(weights["complexity"]) * comp
        - float(weights["instability"]) * instab
    )


def _instability_penalty(
    candidate: Candidate,
    base_metrics: dict[str, float | str],
    data_by_symbol: dict[str, pd.DataFrame],
    search_space: dict[str, Any],
    round_trip_cost_pct: float,
    slippage_pct: float,
    initial_capital: float,
    weights: dict[str, float],
) -> float:
    base_score = _compute_score(metrics=base_metrics, candidate=candidate, weights=weights, instability_penalty=0.0)

    perturbed = perturb_candidate(candidate=candidate, search_space=search_space, pct=0.1)
    if not perturbed:
        return 0.0

    perturb_scores: list[float] = []
    for variant in perturbed:
        metrics = _evaluate_candidate_metrics(
            candidate=variant,
            data_by_symbol=data_by_symbol,
            round_trip_cost_pct=round_trip_cost_pct,
            slippage_pct=slippage_pct,
            initial_capital=initial_capital,
        )
        perturb_scores.append(_compute_score(metrics=metrics, candidate=variant, weights=weights, instability_penalty=0.0))

    mean_perturb = float(np.mean(perturb_scores)) if perturb_scores else base_score
    collapse = max(0.0, base_score - mean_perturb)
    return collapse


def _slice_by_months(frame: pd.DataFrame, months: int) -> pd.DataFrame:
    sliced, _ = slice_last_n_months(frame, window_months=int(months), end_mode="latest")
    return sliced


def _compute_data_hash(data_by_symbol: dict[str, pd.DataFrame]) -> str:
    payload: dict[str, Any] = {}
    for symbol, frame in sorted(data_by_symbol.items(), key=lambda x: x[0]):
        timestamps = pd.to_datetime(frame["timestamp"], utc=True)
        payload[symbol] = {
            "rows": int(len(frame)),
            "start": timestamps.iloc[0].isoformat() if not frame.empty else None,
            "end": timestamps.iloc[-1].isoformat() if not frame.empty else None,
            "close_mean": float(frame["close"].astype(float).mean()) if not frame.empty else None,
            "close_std": float(frame["close"].astype(float).std(ddof=0)) if not frame.empty else None,
        }
    return stable_hash(payload, length=16)


def _load_stage1_data(
    symbols: list[str],
    timeframe: str,
    data_dir: Path,
    dry_run: bool,
    start: str | None,
    seed: int,
) -> dict[str, pd.DataFrame]:
    loaded: dict[str, pd.DataFrame] = {}
    if dry_run:
        for symbol in symbols:
            loaded[symbol] = _generate_synthetic_ohlcv(symbol=symbol, start=start, bars=2400, seed=seed)
        return loaded

    for symbol in symbols:
        loaded[symbol] = load_parquet(symbol=symbol, timeframe=timeframe, data_dir=data_dir)
    return loaded


def _generate_synthetic_ohlcv(symbol: str, start: str | None, bars: int, seed: int) -> pd.DataFrame:
    bars = max(int(bars), 400)
    symbol_seed = int.from_bytes(stable_hash(f"{seed}:{symbol}", length=16).encode("utf-8")[:8], "big", signed=False)
    rng = np.random.default_rng(symbol_seed)

    start_ts = pd.Timestamp(start, tz="UTC") if start else pd.Timestamp("2020-01-01T00:00:00Z")
    timestamps = pd.date_range(start=start_ts, periods=bars, freq="h", tz="UTC")

    base_price = 100.0 + (symbol_seed % 40)
    log_returns = rng.normal(loc=0.0001, scale=0.006, size=bars)
    close = base_price * np.exp(np.cumsum(log_returns))
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    spread = rng.uniform(0.0005, 0.008, size=bars)
    high = np.maximum(open_, close) * (1.0 + spread)
    low = np.minimum(open_, close) * (1.0 - spread)
    volume = rng.uniform(500, 5000, size=bars)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _build_stage_diagnostics(trade_counts: list[float]) -> dict[str, Any]:
    counts = [float(max(0.0, value)) for value in trade_counts]
    total_candidates = len(counts)
    total_trades = float(np.sum(counts)) if counts else 0.0
    avg_trades = (total_trades / float(total_candidates)) if total_candidates > 0 else 0.0
    median_trades = float(np.median(counts)) if counts else 0.0
    min_trades = float(np.min(counts)) if counts else 0.0
    max_trades = float(np.max(counts)) if counts else 0.0
    zero_trade_count = int(sum(1 for value in counts if value <= 0.0))
    percent_zero_trade = (
        (100.0 * float(zero_trade_count) / float(total_candidates)) if total_candidates > 0 else 0.0
    )

    histogram = {
        "0": 0,
        "1-10": 0,
        "11-50": 0,
        "51-200": 0,
        ">200": 0,
    }
    for value in counts:
        if value <= 0.0:
            histogram["0"] += 1
        elif value <= 10.0:
            histogram["1-10"] += 1
        elif value <= 50.0:
            histogram["11-50"] += 1
        elif value <= 200.0:
            histogram["51-200"] += 1
        else:
            histogram[">200"] += 1

    return {
        "total_candidates_evaluated": int(total_candidates),
        "total_trades_evaluated": float(total_trades),
        "avg_trades_per_candidate": float(avg_trades),
        "median_trades_per_candidate": float(median_trades),
        "min_trades": float(min_trades),
        "max_trades": float(max_trades),
        "zero_trade_candidate_count": int(zero_trade_count),
        "percent_zero_trade": float(percent_zero_trade),
        "trade_count_histogram": histogram,
    }


def _write_stage1_diagnostics_report(
    run_dir: Path,
    diagnostics: dict[str, Any],
    docs_md_path: Path,
) -> None:
    stage_a = diagnostics["stages"]["A"]
    warning = stage_a["avg_trades_per_candidate"] < 30.0

    lines: list[str] = []
    lines.append("# Stage-1 Diagnostics")
    lines.append("")
    lines.append(f"- run_id: `{diagnostics['run_id']}`")
    lines.append(f"- seed: `{diagnostics['seed']}`")
    lines.append(f"- split_mode: `{diagnostics['split_mode']}`")
    lines.append("")

    for stage_name in ["A", "B", "C"]:
        stage = diagnostics["stages"][stage_name]
        lines.append(f"## Stage {stage_name}")
        lines.append(f"- total_candidates_evaluated: `{stage['total_candidates_evaluated']}`")
        lines.append(f"- total_trades_evaluated: `{stage['total_trades_evaluated']:.2f}`")
        lines.append(f"- avg_trades_per_candidate: `{stage['avg_trades_per_candidate']:.2f}`")
        lines.append(f"- median_trades_per_candidate: `{stage['median_trades_per_candidate']:.2f}`")
        lines.append(f"- min_trades: `{stage['min_trades']:.2f}`")
        lines.append(f"- max_trades: `{stage['max_trades']:.2f}`")
        lines.append(f"- zero_trade_candidate_count: `{stage['zero_trade_candidate_count']}`")
        lines.append(f"- percent_zero_trade: `{stage['percent_zero_trade']:.2f}`")
        lines.append("- trade_count_histogram:")
        for bucket in ["0", "1-10", "11-50", "51-200", ">200"]:
            lines.append(f"  - {bucket}: `{stage['trade_count_histogram'][bucket]}`")
        lines.append("")

    if warning:
        lines.append("WARNING: Search space too restrictive; insufficient trade sampling.")
        lines.append("")

    md_text = "\n".join(lines).strip() + "\n"
    docs_md_path.parent.mkdir(parents=True, exist_ok=True)
    docs_md_path.write_text(md_text, encoding="utf-8")
    (run_dir / "stage1_diagnostics.md").write_text(md_text, encoding="utf-8")


def _write_stage1_report(
    run_dir: Path,
    summary: dict[str, Any],
    top_three: list[dict[str, Any]],
    docs_report_path: Path | None,
) -> None:
    report_lines: list[str] = []
    report_lines.append("# Stage-1 Auto Optimization Report")
    report_lines.append("")
    report_lines.append(f"- run_id: `{summary['run_id']}`")
    report_lines.append(f"- stage: `{summary['stage_version']}`")
    report_lines.append(f"- runtime_seconds: `{summary['runtime_seconds']:.2f}`")
    report_lines.append(f"- seed: `{summary['seed']}`")
    report_lines.append(f"- config_hash: `{summary['config_hash']}`")
    report_lines.append(f"- data_hash: `{summary['data_hash']}`")
    report_lines.append(f"- cost(round_trip_cost_pct): `{summary['round_trip_cost_pct']}`")
    report_lines.append(f"- candidates A/B/C: `{summary['candidate_count_stage_a']}/{summary['candidate_count_stage_b']}/{summary['candidate_count_stage_c']}`")
    report_lines.append("")

    report_lines.append("## Top 3 Candidates")
    if not top_three:
        report_lines.append("No candidates produced.")
    else:
        for item in top_three:
            metrics = item["metrics_holdout"]
            report_lines.append(f"### Rank {item['rank']} - {item['family']}")
            report_lines.append(f"- Strategy: `{item['strategy_name']}`")
            report_lines.append(f"- Gating: `{item['gating_mode']}`")
            report_lines.append(f"- Exit mode: `{item['exit_mode']}`")
            report_lines.append(f"- Entry rules: {item['rules']['entry']}")
            report_lines.append(f"- Exit rules: {item['rules']['exit']}")
            report_lines.append(f"- Parameters: `{json.dumps(item['parameters'], sort_keys=True)}`")
            report_lines.append(
                "- Holdout metrics: "
                f"trade_count={metrics['trade_count']:.0f}, "
                f"tpm={metrics.get('trades_per_month', 0.0):.4f}, "
                f"pf_adj={metrics.get('pf_adj', 0.0):.4f}, "
                f"PF={metrics['profit_factor']:.4f}, "
                f"expectancy={metrics['expectancy']:.4f}, "
                f"exp_lcb={metrics.get('exp_lcb', 0.0):.4f}, "
                f"effective_edge={metrics.get('effective_edge', 0.0):.4f}, "
                f"exposure_ratio={metrics.get('exposure_ratio', 0.0):.4f}, "
                f"low_signal_penalty={metrics.get('low_signal_penalty', 0.0):.4f}, "
                f"penalty_relief={bool(metrics.get('penalty_relief_applied', False))}, "
                f"max_dd={metrics['max_drawdown']:.4f}, "
                f"return_pct={metrics['return_pct']:.4f}"
            )
            report_lines.append(f"- Holdout range: `{metrics['date_range']}`")
            report_lines.append("")

    # Persist under run artifacts and docs root.
    report_text = "\n".join(report_lines).strip() + "\n"
    (run_dir / "stage1_report.md").write_text(report_text, encoding="utf-8")

    resolved_docs_path = docs_report_path or (Path("docs") / "stage1_auto_optimization_report.md")
    resolved_docs_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_docs_path.write_text(report_text, encoding="utf-8")


def _write_stage1_real_data_reports(
    run_dir: Path,
    summary: dict[str, Any],
    top_five_rows: list[CandidateEval],
    docs_md_path: Path,
    docs_json_path: Path,
) -> None:
    payload_candidates: list[dict[str, Any]] = []
    for idx, row in enumerate(top_five_rows, start=1):
        spec = candidate_to_strategy_spec(row.candidate)
        validation = row.metrics_validation or {}
        holdout = row.metrics_holdout or {}
        combined = row.metrics_combined or {}
        payload_candidates.append(
            {
                "rank": idx,
                "candidate_id": row.candidate.candidate_id,
                "strategy_name": spec.name,
                "strategy_family": row.candidate.family,
                "parameters": row.candidate.params,
                "gating": row.candidate.gating_mode,
                "exit_mode": row.candidate.exit_mode,
                "trade_count_validation": float(validation.get("trade_count", 0.0)),
                "trade_count_holdout": float(holdout.get("trade_count", 0.0)),
                "pf_validation": float(validation.get("profit_factor", 0.0)),
                "pf_holdout": float(holdout.get("profit_factor", 0.0)),
                "pf_adj_holdout": float(row.pf_adj_holdout),
                "expectancy_validation": float(validation.get("expectancy", 0.0)),
                "expectancy_holdout": float(holdout.get("expectancy", 0.0)),
                "exp_lcb_holdout": float(row.exp_lcb_holdout),
                "effective_edge": float(row.effective_edge),
                "max_drawdown_combined": float(combined.get("max_drawdown", 0.0)),
                "max_drawdown_holdout": float(holdout.get("max_drawdown", 0.0)),
                "return_pct_holdout": float(holdout.get("return_pct", 0.0)),
                "cagr_approx_holdout": float(row.cagr_approx_holdout),
                "trades_per_month_holdout": float(row.trades_per_month_holdout),
                "exposure_ratio": float(row.exposure_ratio),
                "exposure_penalty": float(row.exposure_penalty),
                "low_signal_penalty": float(row.low_signal_penalty),
                "penalty_relief_applied": bool(row.penalty_relief_applied),
                "validation_exposure_ratio": float(row.validation_exposure_ratio),
                "validation_active_days": float(row.validation_active_days),
                "validation_evidence_passed": bool(row.validation_evidence_passed),
                "holdout_by_symbol": row.holdout_symbol_metrics or {},
                "validation_range": str(validation.get("date_range", "n/a")),
                "holdout_range": str(holdout.get("date_range", "n/a")),
                "combined_range": str(combined.get("date_range", "n/a")),
                "rejected": bool(row.rejected),
                "rejection_reason": row.rejection_reason,
                "score": float(row.score),
            }
        )

    any_profitable_raw = any(
        item["pf_holdout"] > 1.0 and item["expectancy_holdout"] > 0.0
        for item in payload_candidates
    )
    any_profitable_accepted = any(
        item["pf_holdout"] > 1.0 and item["expectancy_holdout"] > 0.0 and not item["rejected"]
        for item in payload_candidates
    )
    accepted_candidates = [item for item in payload_candidates if not item["rejected"]]
    accepted_top10 = accepted_candidates[:10]
    accepted_tpm_values = [float(item["trades_per_month_holdout"]) for item in accepted_top10]
    accepted_tpm_stats = {
        "count": int(len(accepted_tpm_values)),
        "min": float(np.min(accepted_tpm_values)) if accepted_tpm_values else 0.0,
        "median": float(np.median(accepted_tpm_values)) if accepted_tpm_values else 0.0,
        "max": float(np.max(accepted_tpm_values)) if accepted_tpm_values else 0.0,
    }
    best_accepted = summary.get("best_accepted")
    if best_accepted is None and accepted_candidates:
        best_accepted = accepted_candidates[0]
    best_accepted_flat: dict[str, Any] | None = None
    if best_accepted is not None:
        if "metrics_holdout" in best_accepted and "pf_adj_holdout" not in best_accepted:
            metrics_holdout = best_accepted.get("metrics_holdout", {})
            best_accepted_flat = {
                "candidate_id": best_accepted.get("candidate_id"),
                "strategy_name": best_accepted.get("strategy_name"),
                "score": float(best_accepted.get("score", 0.0)),
                "pf_adj_holdout": float(metrics_holdout.get("pf_adj", 0.0)),
                "exp_lcb_holdout": float(metrics_holdout.get("exp_lcb", 0.0)),
                "trades_per_month_holdout": float(metrics_holdout.get("trades_per_month", 0.0)),
                "exposure_ratio": float(metrics_holdout.get("exposure_ratio", 0.0)),
                "validation_exposure_ratio": float(
                    (best_accepted.get("validation_evidence", {}) or {}).get("exposure_ratio", 0.0)
                ),
                "validation_active_days": float(
                    (best_accepted.get("validation_evidence", {}) or {}).get("active_days", 0.0)
                ),
                "validation_evidence_passed": bool(
                    (best_accepted.get("validation_evidence", {}) or {}).get("passed", False)
                ),
            }
        else:
            best_accepted_flat = best_accepted

    report_payload = {
        "run_id": summary["run_id"],
        "timestamp_utc": utc_now_compact(),
        "split_mode": summary["split_mode"],
        "recent_weight": summary["recent_weight"],
        "min_holdout_trades": summary["min_holdout_trades"],
        "min_validation_exposure_ratio": summary["min_validation_exposure_ratio"],
        "min_validation_active_days": summary["min_validation_active_days"],
        "rejected_due_validation_evidence_count": summary["rejected_due_validation_evidence_count"],
        "round_trip_cost_pct": summary["round_trip_cost_pct"],
        "any_pf_holdout_gt_1_and_expectancy_holdout_gt_0": any_profitable_raw,
        "any_accepted_pf_holdout_gt_1_and_expectancy_holdout_gt_0": any_profitable_accepted,
        "best_accepted_candidate": best_accepted_flat,
        "accepted_top10_tpm_distribution": accepted_tpm_stats,
        "top_5_candidates": payload_candidates,
    }

    docs_json_path.parent.mkdir(parents=True, exist_ok=True)
    docs_json_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    (run_dir / "stage1_real_data_report.json").write_text(json.dumps(report_payload, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Stage-1 Real Data Report")
    lines.append("")
    lines.append(f"- run_id: `{summary['run_id']}`")
    lines.append(f"- split_mode: `{summary['split_mode']}`")
    lines.append(f"- recent_weight: `{summary['recent_weight']}`")
    lines.append(f"- min_holdout_trades: `{summary['min_holdout_trades']}`")
    lines.append(f"- min_validation_exposure_ratio: `{summary['min_validation_exposure_ratio']}`")
    lines.append(f"- min_validation_active_days: `{summary['min_validation_active_days']}`")
    lines.append(f"- rejected_due_validation_evidence_count: `{summary['rejected_due_validation_evidence_count']}`")
    lines.append(f"- target_trades_per_month_holdout: `{summary['target_trades_per_month_holdout']}`")
    lines.append(f"- low_signal_penalty_weight: `{summary['low_signal_penalty_weight']}`")
    lines.append(f"- min_trades_per_month_floor: `{summary['min_trades_per_month_floor']}`")
    lines.append(f"- round_trip_cost_pct: `{summary['round_trip_cost_pct']}`")
    lines.append("")
    lines.append("## Top 5 Candidates")
    lines.append(
        "| rank | strategy | gating | exit | val_trades | hold_trades | tpm_hold | pf_adj | exp_lcb | effective_edge | exposure | PF_val | PF_hold | exp_val | exp_hold | max_dd_hold | return_hold | CAGR_hold | score | rejected |"
    )
    lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for item in payload_candidates:
        lines.append(
            f"| {item['rank']} | {item['strategy_name']} | {item['gating']} | {item['exit_mode']} | "
            f"{item['trade_count_validation']:.0f} | {item['trade_count_holdout']:.0f} | "
            f"{item['trades_per_month_holdout']:.4f} | {item['pf_adj_holdout']:.4f} | "
            f"{item['exp_lcb_holdout']:.4f} | {item['effective_edge']:.4f} | "
            f"{item['exposure_ratio']:.4f} | "
            f"{item['pf_validation']:.4f} | {item['pf_holdout']:.4f} | "
            f"{item['expectancy_validation']:.4f} | {item['expectancy_holdout']:.4f} | "
            f"{item['max_drawdown_holdout']:.4f} | {item['return_pct_holdout']:.4f} | "
            f"{item['cagr_approx_holdout']:.4f} | {item['score']:.4f} | "
            f"{'yes' if item['rejected'] else 'no'} |"
        )
    lines.append("")
    lines.append("### Holdout By Symbol (Top Candidates)")
    for item in payload_candidates:
        by_symbol = item.get("holdout_by_symbol", {})
        btc = by_symbol.get("BTC/USDT", {})
        eth = by_symbol.get("ETH/USDT", {})
        lines.append(
            f"- Rank {item['rank']} {item['strategy_name']}: "
            f"BTC(PF={float(btc.get('profit_factor', 0.0)):.4f}, "
            f"ret={float(btc.get('return_pct', 0.0)):.4f}, "
            f"DD={float(btc.get('max_drawdown', 0.0)):.4f}, "
            f"trades={float(btc.get('trade_count', 0.0)):.0f}) | "
            f"ETH(PF={float(eth.get('profit_factor', 0.0)):.4f}, "
            f"ret={float(eth.get('return_pct', 0.0)):.4f}, "
            f"DD={float(eth.get('max_drawdown', 0.0)):.4f}, "
            f"trades={float(eth.get('trade_count', 0.0)):.0f})"
        )
    lines.append("")
    lines.append("## Accepted Summary")
    if best_accepted_flat is None:
        lines.append("- Best ACCEPTED candidate: none")
    else:
        lines.append(
            "- Best ACCEPTED candidate: "
            f"`{best_accepted_flat['strategy_name']}` "
            f"(pf_adj_holdout={best_accepted_flat['pf_adj_holdout']:.4f}, "
            f"exp_lcb_holdout={best_accepted_flat['exp_lcb_holdout']:.4f}, "
            f"tpm_holdout={best_accepted_flat['trades_per_month_holdout']:.4f}, "
            f"exposure_ratio={best_accepted_flat['exposure_ratio']:.4f}, "
            f"validation_exposure={best_accepted_flat.get('validation_exposure_ratio', 0.0):.4f}, "
            f"validation_active_days={best_accepted_flat.get('validation_active_days', 0.0):.2f}, "
            f"score={best_accepted_flat['score']:.4f})"
        )
    lines.append(
        "- Top 10 accepted candidates tpm distribution: "
        f"min={accepted_tpm_stats['min']:.4f}, "
        f"median={accepted_tpm_stats['median']:.4f}, "
        f"max={accepted_tpm_stats['max']:.4f} "
        f"(count={accepted_tpm_stats['count']})"
    )
    lines.append("")
    lines.append(
        "Does any candidate satisfy PF_holdout > 1 and expectancy_holdout > 0? "
        f"**{'YES' if any_profitable_raw else 'NO'}**"
    )
    lines.append(
        "Accepted candidates only (after low-signal degeneracy filter): "
        f"**{'YES' if any_profitable_accepted else 'NO'}**"
    )
    lines.append("")

    md_text = "\n".join(lines).strip() + "\n"
    docs_md_path.parent.mkdir(parents=True, exist_ok=True)
    docs_md_path.write_text(md_text, encoding="utf-8")
    (run_dir / "stage1_real_data_report.md").write_text(md_text, encoding="utf-8")


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _finite_or_default(value: float, default: float) -> float:
    numeric = float(value)
    return numeric if math.isfinite(numeric) else float(default)
