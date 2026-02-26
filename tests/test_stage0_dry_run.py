"""Stage-0 dry-run reproducibility and artifact integrity tests."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _run_stage0(root: Path, run_id: str, runs_dir: Path) -> Path:
    run_dir = runs_dir / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)

    cmd = [
        sys.executable,
        "scripts/run_stage0.py",
        "--dry-run",
        "--run-id",
        run_id,
        "--runs-dir",
        str(runs_dir),
    ]
    subprocess.run(cmd, cwd=root, check=True)
    return run_dir


def _hashable_payload(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.pop("run_id", None)
    payload.pop("generated_at_utc", None)
    return payload


def test_stage0_dry_run_artifacts_have_required_files_and_shape() -> None:
    root = Path(__file__).resolve().parents[1]
    runs_dir = root / "runs" / "_pytest_stage0"
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_dir = _run_stage0(root=root, run_id="artifact_check", runs_dir=runs_dir)

    required = ["config.yaml", "summary.json", "leaderboard.csv", "strategies.json"]
    for name in required:
        assert (run_dir / name).exists(), f"Missing required artifact: {name}"

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    for key in ["run_id", "config_hash", "seed", "dry_run", "total_combinations", "combinations_executed"]:
        assert key in summary

    strategies = json.loads((run_dir / "strategies.json").read_text(encoding="utf-8"))
    assert isinstance(strategies, list)
    assert len(strategies) == 3
    for strategy in strategies:
        for key in ["name", "entry_rules", "exit_rules", "parameters"]:
            assert key in strategy

    leaderboard = pd.read_csv(run_dir / "leaderboard.csv")
    for col in ["symbol", "strategy", "trade_count", "expectancy", "win_rate"]:
        assert col in leaderboard.columns


def test_stage0_dry_run_is_reproducible_for_same_config_seed_and_data() -> None:
    root = Path(__file__).resolve().parents[1]
    runs_dir = root / "runs" / "_pytest_stage0"
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_a = _run_stage0(root=root, run_id="repro_a", runs_dir=runs_dir)
    run_b = _run_stage0(root=root, run_id="repro_b", runs_dir=runs_dir)

    leaderboard_a = pd.read_csv(run_a / "leaderboard.csv")
    leaderboard_b = pd.read_csv(run_b / "leaderboard.csv")
    pd.testing.assert_frame_equal(leaderboard_a, leaderboard_b)

    strategies_a = json.loads((run_a / "strategies.json").read_text(encoding="utf-8"))
    strategies_b = json.loads((run_b / "strategies.json").read_text(encoding="utf-8"))
    assert strategies_a == strategies_b

    assert _hashable_payload(run_a / "summary.json") == _hashable_payload(run_b / "summary.json")
