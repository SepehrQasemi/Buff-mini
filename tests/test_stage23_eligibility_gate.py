from __future__ import annotations

import pandas as pd

from buffmini.stage23.eligibility import evaluate_eligibility


def _frame() -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=6, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "score_trend": [0.9, 0.8, 0.2, 0.1, 0.6, 0.7],
            "score_range": [0.1, 0.2, 0.8, 0.9, 0.4, 0.3],
            "score_chop": [0.1, 0.2, 0.8, 0.7, 0.2, 0.1],
            "atr_pct_rank_252": [0.55, 0.60, 0.50, 0.45, 0.55, 0.60],
            "regime_label_stage10": ["TREND", "TREND", "CHOP", "RANGE", "TREND", "TREND"],
        }
    )


def _stage23_cfg() -> dict:
    return {
        "eligibility": {
            "min_score_default": 0.35,
            "per_regime_thresholds": {"CHOP": 0.50},
        }
    }


def test_eligibility_score_bounds_and_reasons() -> None:
    frame = _frame()
    raw = pd.Series([1, 1, 1, -1, 1, 0], index=frame.index, dtype=int)
    result = evaluate_eligibility(frame=frame, raw_side=raw, family="price", policy_snapshot=_stage23_cfg(), symbol="BTC/USDT")
    score = pd.to_numeric(result["score"], errors="coerce")
    assert ((score >= 0.0) & (score <= 1.0)).all()

    eligible = pd.to_numeric(result["eligible"], errors="coerce").fillna(False).astype(bool)
    reasons = pd.Series(result["reasons"], index=frame.index).astype(str)
    blocked = (raw != 0) & (~eligible)
    if bool(blocked.any()):
        assert (reasons.loc[blocked] != "").all()


def test_eligibility_deterministic() -> None:
    frame = _frame()
    raw = pd.Series([1, 1, -1, -1, 1, 1], index=frame.index, dtype=int)
    left = evaluate_eligibility(frame=frame, raw_side=raw, family="price", policy_snapshot=_stage23_cfg(), symbol="BTC/USDT")
    right = evaluate_eligibility(frame=frame, raw_side=raw, family="price", policy_snapshot=_stage23_cfg(), symbol="BTC/USDT")
    assert left["eligible"].tolist() == right["eligible"].tolist()
    assert pd.Series(left["score"]).equals(pd.Series(right["score"]))
    assert pd.Series(left["reasons"]).equals(pd.Series(right["reasons"]))


def test_known_scenario_passes_eligibility() -> None:
    frame = _frame()
    raw = pd.Series([1, 1, 0, 0, 1, 1], index=frame.index, dtype=int)
    result = evaluate_eligibility(frame=frame, raw_side=raw, family="price", policy_snapshot=_stage23_cfg(), symbol="BTC/USDT")
    eligible = pd.to_numeric(result["eligible"], errors="coerce").fillna(False).astype(bool)
    assert int(eligible.sum()) >= 2

