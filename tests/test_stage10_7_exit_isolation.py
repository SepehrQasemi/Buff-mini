"""Stage-10.7 exit isolation tests."""

from __future__ import annotations

from pathlib import Path

from buffmini.config import load_config
from buffmini.stage10.evaluate import run_stage10_exit_ab


def test_stage10_exit_ab_compare_outputs_and_selection(tmp_path: Path) -> None:
    config = load_config(Path("configs/default.yaml"))
    config["evaluation"]["stage10"]["evaluation"]["dry_run_rows"] = 800

    result = run_stage10_exit_ab(
        config=config,
        seed=42,
        dry_run=True,
        cost_mode="v2",
        walkforward_v2_enabled=False,
        runs_root=tmp_path / "runs",
        docs_dir=tmp_path / "docs",
    )
    rows = result["rows"]
    assert len(rows) == 2
    assert {row["exit_mode"] for row in rows} == {"fixed_atr", "atr_trailing"}
    assert result["selected_exit"] in {"fixed_atr", "atr_trailing"}
    run_dir = (tmp_path / "runs") / result["run_id"]
    assert (run_dir / "exit_ab_compare.csv").exists()
    assert (run_dir / "exit_ab_summary.json").exists()
