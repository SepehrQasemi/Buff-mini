"""Stage-1 optimizer tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from buffmini.config import load_config
from buffmini.discovery.funnel import run_stage1_optimization
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
        "stage1_report.md",
    ]
    for name in required:
        assert (run_dir / name).exists(), f"missing artifact {name}"

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["candidate_count_stage_a"] >= summary["candidate_count_stage_b"]
    assert summary["candidate_count_stage_b"] >= summary["candidate_count_stage_c"]
    assert summary["candidate_count_stage_b"] <= config["evaluation"]["stage1"]["top_k"]
    assert summary["candidate_count_stage_c"] <= config["evaluation"]["stage1"]["top_m"]


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
