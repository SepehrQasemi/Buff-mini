"""Stage-10 determinism checks."""

from __future__ import annotations

from pathlib import Path

from buffmini.config import load_config
from buffmini.stage10.evaluate import run_stage10


def test_stage10_dry_run_deterministic_signature(tmp_path: Path) -> None:
    config = load_config(Path("configs/default.yaml"))
    runs_dir = tmp_path / "runs"
    docs_dir = tmp_path / "docs"

    left = run_stage10(
        config=config,
        seed=42,
        dry_run=True,
        cost_mode="v2",
        walkforward_v2_enabled=False,
        runs_root=runs_dir,
        docs_dir=docs_dir,
    )
    right = run_stage10(
        config=config,
        seed=42,
        dry_run=True,
        cost_mode="v2",
        walkforward_v2_enabled=False,
        runs_root=runs_dir,
        docs_dir=docs_dir,
    )

    assert left["determinism"]["status"] == "PASS"
    assert right["determinism"]["status"] == "PASS"
    assert left["determinism"]["signature"] == right["determinism"]["signature"]
