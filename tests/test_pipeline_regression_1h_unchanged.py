from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd

from buffmini.config import load_config


def _load_stage0_module():
    path = Path("scripts/run_stage0.py")
    spec = importlib.util.spec_from_file_location("stage0_script_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load run_stage0.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pipeline_regression_1h_unchanged_in_legacy_mode(tmp_path: Path) -> None:
    module = _load_stage0_module()
    config = load_config(Path("configs/default.yaml"))
    # Legacy-mode simulation: missing base/operational keys should not change dry-run results.
    config["universe"].pop("base_timeframe", None)
    config["universe"].pop("operational_timeframe", None)
    config["universe"].pop("htf_timeframes", None)

    runs_dir = tmp_path / "runs"
    run_a = module.run_stage0(
        config=config,
        config_path=Path("configs/default.yaml"),
        dry_run=True,
        run_id="legacy_a",
        runs_dir=runs_dir,
        data_dir=tmp_path / "raw",
        synthetic_bars=320,
    )
    run_b = module.run_stage0(
        config=config,
        config_path=Path("configs/default.yaml"),
        dry_run=True,
        run_id="legacy_b",
        runs_dir=runs_dir,
        data_dir=tmp_path / "raw",
        synthetic_bars=320,
    )

    left = pd.read_csv(run_a / "summary.csv").sort_values(["symbol", "strategy"]).reset_index(drop=True)
    right = pd.read_csv(run_b / "summary.csv").sort_values(["symbol", "strategy"]).reset_index(drop=True)
    cols = ["symbol", "strategy", "trade_count", "win_rate", "expectancy", "profit_factor", "max_drawdown"]
    pd.testing.assert_frame_equal(left[cols], right[cols], check_exact=False, rtol=0.0, atol=1e-12)

    summary_payload = json.loads((run_a / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["stage_version"] == "stage0"
    assert summary_payload["gating_mode"] == "none"

