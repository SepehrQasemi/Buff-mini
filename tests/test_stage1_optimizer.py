"""Stage-1 optimizer tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from buffmini.config import load_config
from buffmini.discovery.funnel import (
    _build_stage_diagnostics,
    _candidate_rejection_reason,
    _compute_expectancy_lcb,
    _compute_exposure_penalty,
    _compute_low_signal_penalty,
    _compute_pf_adjusted,
    _compute_temporal_score,
    _compute_trades_per_month,
    _passes_validation_evidence,
    run_stage1_optimization,
)
from buffmini.discovery.generator import candidate_within_search_space, sample_candidate


def _small_stage1_config(root: Path) -> dict:
    config = load_config(root / "configs" / "default.yaml")
    stage1 = config["evaluation"]["stage1"]
    stage1["candidate_count"] = 40
    stage1["top_k"] = 10
    stage1["top_m"] = 5
    stage1["stage_a_months"] = 3
    stage1["stage_b_months"] = 6
    stage1["holdout_months"] = 3
    stage1["walkforward_splits"] = 2
    stage1["early_stop_patience"] = 20
    stage1["min_stage_a_evals"] = 20
    stage1["split_mode"] = "60_20_20"
    stage1["min_holdout_trades"] = 1
    stage1["recent_weight"] = 2.0
    return config


def test_candidate_generation_stays_inside_search_space() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "default.yaml")
    search_space = config["evaluation"]["stage1"]["search_space"]

    rng = np.random.default_rng(123)
    for idx in range(1, 200):
        candidate = sample_candidate(index=idx, rng=rng, search_space=search_space)
        assert candidate_within_search_space(candidate, search_space)


def test_funnel_reduces_candidate_count_and_writes_artifacts(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = _small_stage1_config(root)

    run_dir = run_stage1_optimization(
        config=config,
        config_path=root / "configs" / "default.yaml",
        dry_run=True,
        runs_dir=tmp_path / "runs",
        run_id="stage1_test_a",
        seed=777,
        docs_report_path=tmp_path / "docs" / "stage1_report.md",
    )

    required = [
        "config.yaml",
        "search_space.json",
        "leaderboard.csv",
        "strategies.json",
        "summary.json",
        "diagnostics.json",
        "stage1_report.md",
        "stage1_real_data_report.md",
        "stage1_real_data_report.json",
        "stage1_diagnostics.md",
    ]
    for name in required:
        assert (run_dir / name).exists(), f"missing artifact {name}"

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["candidate_count_stage_a"] >= summary["candidate_count_stage_b"]
    assert summary["candidate_count_stage_b"] >= summary["candidate_count_stage_c"]
    assert summary["candidate_count_stage_b"] <= config["evaluation"]["stage1"]["top_k"]
    assert summary["candidate_count_stage_c"] <= config["evaluation"]["stage1"]["top_m"]

    diagnostics = json.loads((run_dir / "diagnostics.json").read_text(encoding="utf-8"))
    for stage in ["A", "B", "C"]:
        assert stage in diagnostics["stages"]
        keys = {
            "total_candidates_evaluated",
            "total_trades_evaluated",
            "avg_trades_per_candidate",
            "median_trades_per_candidate",
            "min_trades",
            "max_trades",
            "zero_trade_candidate_count",
            "percent_zero_trade",
            "trade_count_histogram",
        }
        assert keys.issubset(diagnostics["stages"][stage].keys())
    assert diagnostics["stages"]["A"]["total_trades_evaluated"] > 0.0


def test_stage1_reproducibility_same_seed_same_top3(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config = _small_stage1_config(root)

    run_a = run_stage1_optimization(
        config=config,
        config_path=root / "configs" / "default.yaml",
        dry_run=True,
        runs_dir=tmp_path / "runs",
        run_id="stage1_repro_a",
        seed=999,
        docs_report_path=tmp_path / "docs" / "report_a.md",
    )
    run_b = run_stage1_optimization(
        config=config,
        config_path=root / "configs" / "default.yaml",
        dry_run=True,
        runs_dir=tmp_path / "runs",
        run_id="stage1_repro_b",
        seed=999,
        docs_report_path=tmp_path / "docs" / "report_b.md",
    )

    top_a = json.loads((run_a / "strategies.json").read_text(encoding="utf-8"))
    top_b = json.loads((run_b / "strategies.json").read_text(encoding="utf-8"))

    assert len(top_a) == 3
    assert len(top_b) == 3

    slim_a = [
        {
            "rank": item["rank"],
            "candidate_id": item["candidate_id"],
            "family": item["family"],
            "gating_mode": item["gating_mode"],
            "exit_mode": item["exit_mode"],
            "parameters": item["parameters"],
        }
        for item in top_a
    ]
    slim_b = [
        {
            "rank": item["rank"],
            "candidate_id": item["candidate_id"],
            "family": item["family"],
            "gating_mode": item["gating_mode"],
            "exit_mode": item["exit_mode"],
            "parameters": item["parameters"],
        }
        for item in top_b
    ]

    assert slim_a == slim_b


def test_zero_trade_diagnostics_detection() -> None:
    diagnostics = _build_stage_diagnostics([0.0, 0.0, 3.0, 25.0, 500.0])
    assert diagnostics["zero_trade_candidate_count"] == 2
    assert diagnostics["percent_zero_trade"] == 40.0
    histogram = diagnostics["trade_count_histogram"]
    assert histogram["0"] == 2
    assert histogram["1-10"] == 1
    assert histogram["11-50"] == 1
    assert histogram[">200"] == 1


def test_pf_adjusted_downweights_low_trade_spikes() -> None:
    low_trade_pf_adj = _compute_pf_adjusted(profit_factor_holdout=5.0, holdout_trades=1.0)
    high_trade_pf_adj = _compute_pf_adjusted(profit_factor_holdout=5.0, holdout_trades=200.0)
    assert low_trade_pf_adj < high_trade_pf_adj
    assert low_trade_pf_adj > 1.0


def test_expectancy_lcb_is_lower_than_mean_with_variance() -> None:
    lcb = _compute_expectancy_lcb(mean_holdout=10.0, std_holdout=5.0, holdout_trades=25.0)
    assert lcb < 10.0


def test_exposure_penalty_applies_below_threshold() -> None:
    low_exposure_penalty = _compute_exposure_penalty(exposure_ratio=0.005, threshold=0.02)
    high_exposure_penalty = _compute_exposure_penalty(exposure_ratio=0.05, threshold=0.02)
    assert low_exposure_penalty > 0.0
    assert high_exposure_penalty == 0.0


def test_stage13_score_prefers_effective_edge_and_penalizes_exposure() -> None:
    low_exposure = _compute_temporal_score(
        effective_edge=10.0,
        max_drawdown_holdout=0.1,
        complexity=0.5,
        instability=0.2,
        exposure_penalty=3.0,
    )
    high_exposure = _compute_temporal_score(
        effective_edge=10.0,
        max_drawdown_holdout=0.1,
        complexity=0.5,
        instability=0.2,
        exposure_penalty=0.0,
    )
    assert high_exposure > low_exposure


def test_penalty_relief_triggers_when_expectancy_high() -> None:
    penalty_low_exp, relief_low_exp = _compute_low_signal_penalty(
        trades_per_month_holdout=2.0,
        target_trades_per_month_holdout=8.0,
        expectancy_holdout=1.0,
        allow_rare_if_high_expectancy=True,
        rare_expectancy_threshold=3.0,
        rare_penalty_relief=0.5,
    )
    penalty_high_exp, relief_high_exp = _compute_low_signal_penalty(
        trades_per_month_holdout=2.0,
        target_trades_per_month_holdout=8.0,
        expectancy_holdout=5.0,
        allow_rare_if_high_expectancy=True,
        rare_expectancy_threshold=3.0,
        rare_penalty_relief=0.5,
    )
    assert relief_low_exp is False
    assert relief_high_exp is True
    assert penalty_high_exp < penalty_low_exp


def test_degenerate_rejection_triggers_below_tpm_floor() -> None:
    holdout_metrics = {
        "expectancy": 1.0,
    }
    tpm = _compute_trades_per_month(trade_count=1.0, duration_days=60.0)
    reason = _candidate_rejection_reason(
        holdout_metrics=holdout_metrics,
        trades_per_month_holdout=tpm,
        min_trades_per_month_floor=2.0,
        allow_rare_if_high_expectancy=True,
        rare_expectancy_threshold=3.0,
        validation_evidence_passed=True,
    )
    assert reason == "degenerate_low_trades_per_month"


def test_degenerate_rejection_relief_with_extremely_high_expectancy() -> None:
    holdout_metrics = {
        "expectancy": 10.0,
    }
    tpm = _compute_trades_per_month(trade_count=1.0, duration_days=60.0)
    reason = _candidate_rejection_reason(
        holdout_metrics=holdout_metrics,
        trades_per_month_holdout=tpm,
        min_trades_per_month_floor=2.0,
        allow_rare_if_high_expectancy=True,
        rare_expectancy_threshold=3.0,
        validation_evidence_passed=True,
    )
    assert reason == ""


def test_validation_evidence_gate_requires_exposure_or_active_days() -> None:
    assert _passes_validation_evidence(
        validation_exposure_ratio=0.0,
        validation_active_days=0.0,
        min_validation_exposure_ratio=0.01,
        min_validation_active_days=10.0,
    ) is False
    assert _passes_validation_evidence(
        validation_exposure_ratio=0.02,
        validation_active_days=0.0,
        min_validation_exposure_ratio=0.01,
        min_validation_active_days=10.0,
    ) is True
    assert _passes_validation_evidence(
        validation_exposure_ratio=0.0,
        validation_active_days=12.0,
        min_validation_exposure_ratio=0.01,
        min_validation_active_days=10.0,
    ) is True
