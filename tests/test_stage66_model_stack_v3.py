from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.stage66 import train_model_stack_v3


def test_stage66_trains_registry_v5() -> None:
    rows = 320
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=rows, freq="h", tz="UTC"),
            "f1": np.random.default_rng(42).normal(size=rows),
            "f2": np.random.default_rng(43).normal(size=rows),
            "f3": np.random.default_rng(44).normal(size=rows),
            "tp_before_sl_label": np.where(np.arange(rows) % 3 == 0, 1.0, 0.0),
        }
    )
    registry = train_model_stack_v3(frame, feature_columns=["f1", "f2", "f3"], seed=42)
    assert registry["version"] == "model_registry_v5"
    assert set(registry["base_models"]) >= {"logreg", "hgbt", "rf"}
    assert "summary_hash" in registry

