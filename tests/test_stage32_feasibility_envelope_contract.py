from __future__ import annotations

import pandas as pd

from buffmini.stage32.feasibility import candidate_feasibility_envelope


def test_stage32_feasibility_envelope_contract() -> None:
    signals = pd.DataFrame(
        {
            "candidate_id": ["a", "a", "b", "b"],
            "symbol": ["BTC/USDT", "BTC/USDT", "ETH/USDT", "ETH/USDT"],
            "timeframe": ["1h", "1h", "4h", "4h"],
            "context": ["TREND", "RANGE", "TREND", "RANGE"],
            "stop_dist_pct": [0.01, 0.02, 0.03, 0.015],
        }
    )
    out = candidate_feasibility_envelope(
        signals=signals,
        equity_tiers=[100.0, 1000.0],
        min_notional=10.0,
        cost_rt_pct=0.1,
        max_notional_pct=1.0,
        risk_cap=0.2,
    )
    required = {
        "candidate_id",
        "symbol",
        "timeframe",
        "context",
        "equity",
        "signals",
        "feasible_pct",
        "min_required_risk_p50",
        "min_required_risk_p90",
        "recommended_risk_floor",
    }
    assert not out.empty
    assert required.issubset(set(out.columns))

