from __future__ import annotations

import numpy as np
from pathlib import Path

from buffmini.config import load_config
from buffmini.stage11.evaluate import run_disabled_equivalence_snapshot


def test_stage11_disabled_matches_stage10_paths() -> None:
    config = load_config(Path("configs/default.yaml"))
    config["evaluation"]["stage10"]["evaluation"]["dry_run_rows"] = 700
    config["evaluation"]["stage11"]["enabled"] = False

    snapshot = run_disabled_equivalence_snapshot(
        config=config,
        seed=42,
        symbols=["BTC/USDT"],
        timeframe="1h",
        dry_run=True,
    )
    assert "BTC/USDT" in snapshot
    payload = snapshot["BTC/USDT"]
    assert payload["left_trade_times"] == payload["right_trade_times"]
    np.testing.assert_allclose(payload["left_equity"], payload["right_equity"], rtol=0.0, atol=0.0)

