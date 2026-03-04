from __future__ import annotations

import pandas as pd

from buffmini.stage28.feasibility_envelope import compute_feasibility_envelope


def test_stage28_feasibility_envelope_monotonic_with_equity() -> None:
    signals = pd.DataFrame(
        {
            "symbol": ["BTC/USDT"] * 4,
            "timeframe": ["1h"] * 4,
            "context": ["TREND"] * 4,
            "stop_dist_pct": [0.01, 0.015, 0.02, 0.03],
        }
    )
    out = compute_feasibility_envelope(
        signals=signals,
        equity_tiers=[100.0, 1000.0],
        min_notional=10.0,
        cost_rt_pct=0.001,
        max_notional_pct=1.0,
        risk_cap=0.2,
    )
    assert not out.empty
    low = out.loc[out["equity"] == 100.0].iloc[0]
    high = out.loc[out["equity"] == 1000.0].iloc[0]
    assert float(high["min_required_risk_p50"]) < float(low["min_required_risk_p50"])
    assert float(high["feasible_pct"]) >= float(low["feasible_pct"])

