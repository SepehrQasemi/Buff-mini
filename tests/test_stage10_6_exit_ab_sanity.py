"""Stage-10.6 exit A/B sanity checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from buffmini.config import load_config
from buffmini.stage10.evaluate import run_stage10
from buffmini.stage10.exits import trailing_stop_path


def test_stage10_exit_override_dry_run_does_not_crash(tmp_path: Path) -> None:
    config = load_config(Path("configs/default.yaml"))
    runs_dir = tmp_path / "runs"
    docs_dir = tmp_path / "docs"
    for mode in ("fixed_atr", "atr_trailing", "breakeven_1r"):
        summary = run_stage10(
            config=config,
            seed=42,
            dry_run=True,
            cost_mode="v2",
            walkforward_v2_enabled=False,
            exit_mode=mode,
            runs_root=runs_dir,
            docs_dir=docs_dir,
        )
        assert isinstance(summary["run_id"], str) and summary["run_id"]
        assert summary["baseline_vs_stage10"]["stage10"]["trade_count"] >= 0


def test_trailing_stop_monotonicity_still_holds() -> None:
    long_path = trailing_stop_path(
        side="long",
        entry_stop=99.0,
        highs=[100.0, 101.0, 102.5, 103.0],
        lows=[99.2, 99.4, 100.1, 100.8],
        atr_values=[1.0, 1.0, 1.1, 1.0],
        trailing_k=1.5,
    )
    short_path = trailing_stop_path(
        side="short",
        entry_stop=101.0,
        highs=[100.8, 100.6, 100.2, 100.0],
        lows=[100.0, 99.5, 99.1, 98.8],
        atr_values=[1.0, 1.0, 1.1, 1.0],
        trailing_k=1.5,
    )
    assert (np.diff(long_path.to_numpy(dtype=float)) >= -1e-12).all()
    assert (np.diff(short_path.to_numpy(dtype=float)) <= 1e-12).all()
