from __future__ import annotations

from pathlib import Path

from buffmini.config import load_config
from buffmini.stage11.evaluate import apply_stage11_preset, run_stage11


def test_stage11_bias_sizing_stats_deterministic(tmp_path: Path) -> None:
    base = load_config(Path("configs/default.yaml"))
    cfg = apply_stage11_preset(base, Path("configs/presets/stage11_bias.yaml"))
    cfg["evaluation"]["stage10"]["evaluation"]["dry_run_rows"] = 650

    left = run_stage11(
        config=cfg,
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        runs_root=tmp_path / "runs_left",
        docs_dir=tmp_path / "docs_left",
        write_docs=False,
    )
    right = run_stage11(
        config=cfg,
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        runs_root=tmp_path / "runs_right",
        docs_dir=tmp_path / "docs_right",
        write_docs=False,
    )
    assert left["sizing_stats"] == right["sizing_stats"]

