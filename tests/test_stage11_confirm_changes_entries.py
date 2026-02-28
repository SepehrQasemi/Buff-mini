from __future__ import annotations

from pathlib import Path

from buffmini.config import load_config
from buffmini.stage11.evaluate import apply_stage11_preset, run_stage11


def test_stage11_confirm_changes_entries_vs_baseline(tmp_path: Path) -> None:
    base = load_config(Path("configs/default.yaml"))
    cfg = apply_stage11_preset(base, Path("configs/presets/stage11_confirm.yaml"))
    cfg["evaluation"]["stage10"]["evaluation"]["dry_run_rows"] = 700

    summary = run_stage11(
        config=cfg,
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        runs_root=tmp_path / "runs",
        docs_dir=tmp_path / "docs",
        write_docs=False,
    )
    assert int(summary["entry_delta"]["total_delta_count"]) > 0

