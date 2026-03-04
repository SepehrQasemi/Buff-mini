from __future__ import annotations

import pandas as pd

from buffmini.stage33.policy_v3 import PolicyV3Config, build_policy_v3


def test_stage33_policy_v3_contract() -> None:
    finalists = pd.DataFrame(
        [
            {"candidate_id": "c1", "context": "TREND", "symbol": "BTC/USDT", "timeframe": "1h", "exp_lcb": 0.03, "usable_windows": 6, "htf": "4h", "ltf": "1h"},
            {"candidate_id": "c2", "context": "TREND", "symbol": "BTC/USDT", "timeframe": "1h", "exp_lcb": 0.02, "usable_windows": 5, "htf": "4h", "ltf": "1h"},
            {"candidate_id": "c3", "context": "RANGE", "symbol": "ETH/USDT", "timeframe": "4h", "exp_lcb": 0.01, "usable_windows": 4, "htf": "1d", "ltf": "4h"},
        ]
    )
    policy = build_policy_v3(
        finalists,
        data_snapshot_id="DATA_FROZEN_v1",
        data_snapshot_hash="abc123",
        config_hash="cfg123",
        cfg=PolicyV3Config(top_k_per_context=2),
    )
    assert str(policy.get("version", "")) == "stage33_policy_v3"
    assert "policy_id" in policy
    assert "contexts" in policy
    trend = policy["contexts"].get("TREND", {})
    assert trend.get("status") == "OK"
    assert len(trend.get("candidates", [])) <= 2

