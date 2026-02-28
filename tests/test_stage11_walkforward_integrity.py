from __future__ import annotations

from pathlib import Path

from buffmini.config import load_config
from buffmini.stage11.evaluate import run_stage11


def test_stage11_walkforward_fields_present(tmp_path) -> None:
    config = load_config(Path("configs/default.yaml"))
    config["evaluation"]["stage10"]["evaluation"]["dry_run_rows"] = 600
    config["evaluation"]["stage11"]["enabled"] = True
    config["evaluation"]["stage11"]["hooks"]["confirm"]["enabled"] = False
    config["evaluation"]["stage11"]["hooks"]["exit"]["enabled"] = False

    summary = run_stage11(
        config=config,
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        timeframe="1h",
        cost_mode="v2",
        walkforward_v2_enabled=True,
        runs_root=tmp_path / "runs",
        docs_dir=tmp_path / "docs",
        write_docs=False,
    )
    walkforward = summary["walkforward"]
    assert "baseline_classification" in walkforward
    assert "stage11_classification" in walkforward
    assert walkforward["baseline_classification"] in {"N/A", "STABLE", "UNSTABLE", "INSUFFICIENT_DATA"}
    assert walkforward["stage11_classification"] in {"N/A", "STABLE", "UNSTABLE", "INSUFFICIENT_DATA"}

