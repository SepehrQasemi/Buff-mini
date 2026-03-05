from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.stage34.train import TrainConfig, predict_model_proba, train_stage34_models
from buffmini.utils.hashing import stable_hash


def _synthetic_dataset(rows: int = 1200) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=rows, freq="15min", tz="UTC")
    x = np.linspace(0.0, 8.0 * np.pi, rows)
    close = 100.0 + np.sin(x) + 0.05 * np.arange(rows)
    ret = pd.Series(close).pct_change().fillna(0.0)
    label_primary = np.where(ret.shift(-1).fillna(0.0) > 0, 1, -1)
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "open": close - 0.2,
            "high": close + 0.6,
            "low": close - 0.6,
            "close": close,
            "volume": 1000.0 + (np.arange(rows) % 19) * 10.0,
            "ret_1": ret,
            "ret_3": ret.rolling(3, min_periods=1).sum(),
            "atr_14": pd.Series(close).diff().abs().rolling(14, min_periods=2).mean().fillna(0.01),
            "atr_pct": (pd.Series(close).diff().abs().rolling(14, min_periods=2).mean().fillna(0.01) / pd.Series(close)),
            "volume_z_24": (pd.Series(1000.0 + (np.arange(rows) % 19) * 10.0) - 1090.0) / 55.0,
            "ma_dist_20": (pd.Series(close) - pd.Series(close).rolling(20, min_periods=2).mean().bfill()) / pd.Series(close),
            "label_primary": label_primary.astype(int),
            "label_auxiliary": -np.abs(ret.shift(-1).fillna(0.0)),
        }
    )
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def test_stage34_training_deterministic_summary_hash() -> None:
    data = _synthetic_dataset()
    feats = ["ret_1", "ret_3", "atr_14", "atr_pct", "volume_z_24", "ma_dist_20"]
    cfg = TrainConfig(seed=42, models=("logreg", "hgbt", "rf"), calibration="platt")
    models_a, summary_a = train_stage34_models(data, feature_columns=feats, cfg=cfg)
    models_b, summary_b = train_stage34_models(data, feature_columns=feats, cfg=cfg)
    assert stable_hash(summary_a, length=16) == stable_hash(summary_b, length=16)
    assert set(models_a.keys()) == {"logreg", "hgbt", "rf"}
    assert set(models_b.keys()) == {"logreg", "hgbt", "rf"}


def test_stage34_training_probabilities_finite_and_calibrated() -> None:
    data = _synthetic_dataset(rows=900)
    feats = ["ret_1", "ret_3", "atr_14", "atr_pct", "volume_z_24", "ma_dist_20"]
    models, summary = train_stage34_models(
        data,
        feature_columns=feats,
        cfg=TrainConfig(seed=123, models=("logreg", "hgbt"), calibration="platt"),
    )
    assert int(summary["rows_total"]) == int(data.shape[0])
    probe = data.iloc[-120:, :].reset_index(drop=True)
    for name, model in models.items():
        prob = predict_model_proba(model, probe)
        assert prob.shape[0] == probe.shape[0]
        assert np.isfinite(prob).all(), f"non-finite proba for {name}"
        assert ((prob > 0.0) & (prob < 1.0)).all(), f"uncalibrated bounds for {name}"
