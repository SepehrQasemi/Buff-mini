from __future__ import annotations

import pandas as pd

from buffmini.stage11.policy import build_stage11_policy_hooks


def test_stage11_bias_hook_is_deterministic_and_bounded() -> None:
    hooks = build_stage11_policy_hooks(
        {
            "bias": {
                "enabled": True,
                "multiplier_min": 0.9,
                "multiplier_max": 1.1,
            },
            "confirm": {"enabled": False},
            "exit": {"enabled": False},
        }
    )
    row = {
        "ema_slope_50": 0.02,
        "atr_pct_rank_252": 0.8,
        "volume_z_120": 0.5,
    }
    left = hooks["bias"](
        timestamp=pd.Timestamp("2026-01-01T00:00:00Z"),
        symbol="BTC/USDT",
        signal_family="MA_SlopePullback",
        signal=1,
        base_row=row,
        activation_multiplier=1.0,
    )
    right = hooks["bias"](
        timestamp=pd.Timestamp("2026-01-01T00:00:00Z"),
        symbol="BTC/USDT",
        signal_family="MA_SlopePullback",
        signal=1,
        base_row=row,
        activation_multiplier=1.0,
    )
    assert left == right
    assert 0.9 <= float(left) <= 1.1

