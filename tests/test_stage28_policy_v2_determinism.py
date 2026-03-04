from __future__ import annotations

import pandas as pd

from buffmini.stage28.policy_v2 import PolicyV2Config, build_policy_v2, compose_policy_signal_v2
from buffmini.utils.hashing import stable_hash


def _fixture_finalists() -> pd.DataFrame:
    rows = []
    for idx in range(8):
        rows.append(
            {
                "candidate_id": f"cand_{idx:02d}",
                "candidate": f"Rulelet{idx}",
                "symbol": "BTC/USDT",
                "timeframe": "1h",
                "family": "price",
                "context": "TREND" if idx % 2 == 0 else "RANGE",
                "context_occurrences": 100,
                "trades_in_context": 40 + idx,
                "exp_lcb": 0.03 + idx * 0.002,
                "expectancy": 0.01 + idx * 0.001,
                "cost_sensitivity": 0.01,
            }
        )
    return pd.DataFrame(rows)


def test_stage28_policy_v2_determinism() -> None:
    cfg = PolicyV2Config(top_k_per_context=2, conflict_mode="net")
    first = build_policy_v2(
        _fixture_finalists(),
        data_snapshot_id="DATA_FROZEN_v1",
        data_snapshot_hash="hash123",
        config_hash="cfg123",
        cfg=cfg,
    )
    second = build_policy_v2(
        _fixture_finalists(),
        data_snapshot_id="DATA_FROZEN_v1",
        data_snapshot_hash="hash123",
        config_hash="cfg123",
        cfg=cfg,
    )
    assert stable_hash(first, length=16) == stable_hash(second, length=16)


def test_stage28_policy_v2_signal_compose_determinism() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=24, freq="1h", tz="UTC"),
            "ctx_state": ["TREND"] * 12 + ["RANGE"] * 12,
        }
    )
    finalists = _fixture_finalists()
    policy = build_policy_v2(
        finalists,
        data_snapshot_id="DATA_FROZEN_v1",
        data_snapshot_hash="hash123",
        config_hash="cfg123",
        cfg=PolicyV2Config(top_k_per_context=2, conflict_mode="net"),
    )
    sig_map = {
        f"cand_{idx:02d}": pd.Series(([1, -1, 0, 1] * 6)[:24], index=frame.index, dtype=int)
        for idx in range(8)
    }
    first_sig, first_trace = compose_policy_signal_v2(frame=frame, policy=policy, candidate_signals=sig_map)
    second_sig, second_trace = compose_policy_signal_v2(frame=frame, policy=policy, candidate_signals=sig_map)
    assert stable_hash(first_sig.tolist(), length=16) == stable_hash(second_sig.tolist(), length=16)
    assert stable_hash(first_trace.to_dict(orient="records"), length=16) == stable_hash(
        second_trace.to_dict(orient="records"), length=16
    )

