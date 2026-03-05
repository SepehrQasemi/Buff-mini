from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.stage34.policy import PolicyConfig, replay_policy, select_best_policy
from buffmini.stage34.train import TrainConfig, train_stage34_models
from buffmini.utils.hashing import stable_hash


def _dataset(rows: int = 1800) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=rows, freq="1h", tz="UTC")
    x = np.linspace(0, 12 * np.pi, rows)
    close = 100.0 + np.sin(x) * 2.5 + 0.02 * np.arange(rows)
    atr = pd.Series(close).diff().abs().rolling(14, min_periods=2).mean().fillna(0.2)
    ret = pd.Series(close).pct_change().fillna(0.0)
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "open": close - 0.2,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "volume": 1000 + (np.arange(rows) % 13) * 7.0,
            "ret_1": ret,
            "ret_3": ret.rolling(3, min_periods=1).sum(),
            "atr_14": atr,
            "atr_pct": atr / pd.Series(close),
            "volume_z_24": (pd.Series(1000 + (np.arange(rows) % 13) * 7.0) - 1042.0) / 30.0,
            "ma_dist_20": (pd.Series(close) - pd.Series(close).rolling(20, min_periods=2).mean().bfill()) / pd.Series(close),
            "label_primary": np.where(pd.Series(close).shift(-3).fillna(pd.Series(close).iloc[-1]) > pd.Series(close), 1, -1),
            "label_auxiliary": -ret.shift(-1).abs().fillna(0.0),
            "symbol": "BTC/USDT",
            "timeframe": "1h",
        }
    )
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def test_stage34_policy_replay_has_breakdown_and_deterministic_hash() -> None:
    data = _dataset()
    feats = ["ret_1", "ret_3", "atr_14", "atr_pct", "volume_z_24", "ma_dist_20"]
    models, _ = train_stage34_models(
        data,
        feature_columns=feats,
        cfg=TrainConfig(seed=42, models=("logreg",), calibration="platt"),
    )
    eval_rows = pd.DataFrame(
        [
            {
                "model_name": "logreg",
                "cost_mode": "live",
                "window_months": 3,
                "wf_executed": True,
                "mc_triggered": True,
                "trade_count": 120,
                "exp_lcb": 0.001,
                "positive_windows_ratio": 0.6,
            }
        ]
    )
    policy = select_best_policy(eval_rows, cfg=PolicyConfig(threshold=0.53), seed=42)
    replay_a = replay_policy(data.iloc[-600:, :], model=models["logreg"], policy=policy, mode="live", cfg=PolicyConfig(threshold=0.53))
    replay_b = replay_policy(data.iloc[-600:, :], model=models["logreg"], policy=policy, mode="live", cfg=PolicyConfig(threshold=0.53))
    assert {"accepted_rejected_breakdown", "top_reject_reasons", "trade_count", "status"}.issubset(set(replay_a.keys()))
    assert stable_hash(replay_a, length=16) == stable_hash(replay_b, length=16)
