from __future__ import annotations

from pathlib import Path

from buffmini.config import load_config
from buffmini.stage11.evaluate import apply_stage11_preset, run_stage11


def test_stage11_bias_and_confirm_effectiveness_enforced(tmp_path: Path) -> None:
    base = load_config(Path("configs/default.yaml"))

    bias_cfg = apply_stage11_preset(base, Path("configs/presets/stage11_bias.yaml"))
    bias_cfg["evaluation"]["stage10"]["evaluation"]["dry_run_rows"] = 700
    bias_summary = run_stage11(
        config=bias_cfg,
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        runs_root=tmp_path / "runs_bias",
        docs_dir=tmp_path / "docs_bias",
        write_docs=False,
    )
    assert float(bias_summary["sizing_stats"]["pct_not_1_0"]) > 0.05
    assert bool(bias_summary["noop_bug_detected"]) is False

    confirm_cfg = apply_stage11_preset(base, Path("configs/presets/stage11_confirm.yaml"))
    confirm_cfg["evaluation"]["stage10"]["evaluation"]["dry_run_rows"] = 700
    confirm_summary = run_stage11(
        config=confirm_cfg,
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        runs_root=tmp_path / "runs_confirm",
        docs_dir=tmp_path / "docs_confirm",
        write_docs=False,
    )
    stats = confirm_summary["confirm_stats"]
    assert int(stats["signals_seen"]) > 0
    assert int(stats["confirmed"]) + int(stats["skipped"]) == int(stats["signals_seen"])
    assert 0.05 < float(stats["confirm_rate"]) < 0.95
    assert bool(confirm_summary["noop_bug_detected"]) is False

