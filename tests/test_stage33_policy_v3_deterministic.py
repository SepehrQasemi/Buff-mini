from __future__ import annotations

import pandas as pd

from buffmini.stage33.policy_v3 import build_policy_v3
from buffmini.utils.hashing import stable_hash


def test_stage33_policy_v3_deterministic() -> None:
    finalists = pd.DataFrame(
        [
            {"candidate_id": "a", "context": "TREND", "symbol": "BTC/USDT", "timeframe": "1h", "exp_lcb": 0.05, "usable_windows": 10},
            {"candidate_id": "b", "context": "TREND", "symbol": "BTC/USDT", "timeframe": "1h", "exp_lcb": 0.04, "usable_windows": 9},
            {"candidate_id": "c", "context": "RANGE", "symbol": "ETH/USDT", "timeframe": "4h", "exp_lcb": 0.03, "usable_windows": 7},
        ]
    )
    first = build_policy_v3(
        finalists,
        data_snapshot_id="DATA_FROZEN_v1",
        data_snapshot_hash="hash_a",
        config_hash="cfg_a",
    )
    second = build_policy_v3(
        finalists,
        data_snapshot_id="DATA_FROZEN_v1",
        data_snapshot_hash="hash_a",
        config_hash="cfg_a",
    )
    # ignore generated_at when comparing determinism.
    first_norm = dict(first)
    second_norm = dict(second)
    first_norm["generated_at"] = ""
    second_norm["generated_at"] = ""
    assert stable_hash(first_norm, length=16) == stable_hash(second_norm, length=16)

