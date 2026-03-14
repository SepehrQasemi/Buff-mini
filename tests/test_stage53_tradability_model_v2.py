from __future__ import annotations

import pandas as pd

from buffmini.stage53 import fit_tradability_model_v2, predict_tradability_model_v2, route_tradability_v2


def _dataset(rows: int = 48) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    payload = []
    for idx, timestamp in enumerate(timestamps):
        rr = 1.2 + (idx % 5) * 0.18
        cost_edge = -0.0002 + (idx % 4) * 0.0008
        payload.append(
            {
                "timestamp": timestamp,
                "beam_score": 0.35 + (idx % 6) * 0.08,
                "cost_edge_proxy": cost_edge,
                "rr_first_target": rr,
                "family_code": float(idx % 3),
                "timeframe_code": float(idx % 5),
                "tp_before_sl_label": float(1.0 if rr >= 1.5 and cost_edge > 0 else 0.0),
                "expected_net_after_cost_label": float(cost_edge + (rr - 1.5) * 0.0015),
                "mae_pct_label": float(-0.004 + (idx % 3) * 0.0004),
                "mfe_pct_label": float(0.006 + (idx % 5) * 0.0008),
                "expected_hold_bars_label": float(6 + (idx % 5) * 2),
            }
        )
    return pd.DataFrame(payload)


def test_stage53_fit_predict_and_route() -> None:
    dataset = _dataset()
    feature_columns = ["beam_score", "cost_edge_proxy", "rr_first_target", "family_code", "timeframe_code"]
    model = fit_tradability_model_v2(dataset, feature_columns=feature_columns, seed=42, probability_bins=5)
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": f"c{idx}",
                "beam_score": 0.45 + idx * 0.05,
                "cost_edge_proxy": 0.0005 + idx * 0.0002,
                "rr_first_target": 1.5 + idx * 0.1,
                "family_code": float(idx % 3),
                "timeframe_code": float(idx % 5),
                "rr_model": {"first_target_rr": 1.5 + idx * 0.1},
                "exp_lcb_proxy": 0.001 + idx * 0.001,
            }
            for idx in range(6)
        ]
    )
    predictions = predict_tradability_model_v2(model, candidates)
    routed = route_tradability_v2(candidates, predictions=predictions)
    assert {"tp_before_sl_prob", "expected_net_after_cost", "replay_priority"}.issubset(predictions.columns)
    assert int(routed["counts"]["stage_b"]) <= int(routed["counts"]["stage_a"]) <= int(routed["counts"]["input"])
