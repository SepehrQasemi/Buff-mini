from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.stage34.eval import EvalConfig, evaluate_models_strict
from buffmini.stage34.train import TrainConfig, train_stage34_models


def _build_dataset(rows: int = 2500) -> pd.DataFrame:
    ts = pd.date_range("2023-01-01", periods=rows, freq="1h", tz="UTC")
    x = np.linspace(0, 18 * np.pi, rows)
    close = 100.0 + np.sin(x) * 3.0 + 0.01 * np.arange(rows)
    atr = pd.Series(close).diff().abs().rolling(14, min_periods=2).mean().fillna(0.2)
    ret = pd.Series(close).pct_change().fillna(0.0)
    df = pd.DataFrame(
        {
            "timestamp": ts,
            "open": close - 0.3,
            "high": close + 0.9,
            "low": close - 0.9,
            "close": close,
            "volume": 1000.0 + (np.arange(rows) % 11) * 13.0,
            "ret_1": ret,
            "ret_3": ret.rolling(3, min_periods=1).sum(),
            "atr_14": atr,
            "atr_pct": atr / pd.Series(close),
            "volume_z_24": (pd.Series(1000.0 + (np.arange(rows) % 11) * 13.0) - 1065.0) / 40.0,
            "ma_dist_20": (pd.Series(close) - pd.Series(close).rolling(20, min_periods=2).mean().bfill()) / pd.Series(close),
            "label_primary": np.where(pd.Series(close).shift(-4).fillna(pd.Series(close).iloc[-1]) > pd.Series(close), 1, -1),
            "label_auxiliary": -ret.shift(-1).abs().fillna(0.0),
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        }
    )
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def test_stage34_eval_schema_keys_present() -> None:
    data = _build_dataset(rows=3000)
    feat_cols = ["ret_1", "ret_3", "atr_14", "atr_pct", "volume_z_24", "ma_dist_20"]
    models, _ = train_stage34_models(
        data,
        feature_columns=feat_cols,
        cfg=TrainConfig(seed=42, models=("logreg", "hgbt"), calibration="platt"),
    )
    rows, summary = evaluate_models_strict(
        data,
        models=models,
        cfg=EvalConfig(
            threshold=0.52,
            window_months=(3,),
            step_months=1,
            min_window_trades=3,
            min_window_exposure=0.0,
            min_usable_windows=1,
            mc_min_trades=5,
            seed=42,
        ),
    )
    assert not rows.empty
    expected_cols = {
        "model_name",
        "cost_mode",
        "window_months",
        "wf_executed",
        "mc_triggered",
        "trade_count",
        "exp_lcb",
        "failure_mode",
    }
    assert expected_cols.issubset(set(rows.columns))
    assert {"status", "wf_executed_pct", "mc_trigger_pct", "final_verdict"}.issubset(set(summary.keys()))


def test_stage34_eval_wf_mc_trigger_with_sufficient_trades() -> None:
    data = _build_dataset(rows=3200)
    feat_cols = ["ret_1", "ret_3", "atr_14", "atr_pct", "volume_z_24", "ma_dist_20"]
    models, _ = train_stage34_models(
        data,
        feature_columns=feat_cols,
        cfg=TrainConfig(seed=7, models=("logreg",), calibration="platt"),
    )
    rows, summary = evaluate_models_strict(
        data,
        models=models,
        cfg=EvalConfig(
            threshold=0.51,
            window_months=(3,),
            step_months=1,
            min_window_trades=2,
            min_window_exposure=0.0,
            min_usable_windows=1,
            mc_min_trades=2,
            seed=7,
        ),
    )
    assert not rows.empty
    assert float(summary["wf_executed_pct"]) > 0.0
    assert float(summary["mc_trigger_pct"]) > 0.0
