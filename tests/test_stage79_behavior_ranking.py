from __future__ import annotations

import numpy as np
import pandas as pd

from buffmini.research.behavior import build_behavioral_fingerprints
from buffmini.stage48.tradability_learning import score_candidates_with_ranker


def _bars() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01T00:00:00Z", periods=64, freq="1h", tz="UTC")
    base = 100.0 + np.cumsum(np.sin(np.arange(64) / 5.0) * 0.3)
    frame = pd.DataFrame(
        {
            "timestamp": idx,
            "open": base - 0.05,
            "high": base + 0.20,
            "low": base - 0.20,
            "close": base,
            "volume": 1000.0 + (np.arange(64) % 5) * 10.0,
        }
    )
    frame["sig_dup"] = 0
    frame.loc[frame.index % 8 == 1, "sig_dup"] = 1
    frame["sig_alt"] = 0
    frame.loc[frame.index % 11 == 2, "sig_alt"] = -1
    return frame


def test_stage79_ranking_adds_behavior_risk_and_classes() -> None:
    frame = _bars()
    candidates = pd.DataFrame(
        [
            {
                "candidate_id": "dup_a",
                "beam_score": 0.81,
                "exp_lcb_proxy": 0.010,
                "cost_edge_proxy": 0.010,
                "time_stop_bars": 10,
                "transfer_risk_prior": 0.20,
                "rr_model": {"first_target_rr": 1.7},
                "signal_spec": {"type": "frame_columns", "long_col": "sig_dup"},
            },
            {
                "candidate_id": "dup_b",
                "beam_score": 0.79,
                "exp_lcb_proxy": 0.009,
                "cost_edge_proxy": 0.010,
                "time_stop_bars": 10,
                "transfer_risk_prior": 0.20,
                "rr_model": {"first_target_rr": 1.7},
                "signal_spec": {"type": "frame_columns", "long_col": "sig_dup"},
            },
            {
                "candidate_id": "diverse_c",
                "beam_score": 0.77,
                "exp_lcb_proxy": 0.008,
                "cost_edge_proxy": 0.011,
                "time_stop_bars": 14,
                "transfer_risk_prior": 0.15,
                "rr_model": {"first_target_rr": 1.8},
                "signal_spec": {"type": "frame_columns", "short_col": "sig_alt"},
            },
        ]
    )
    labels = pd.DataFrame(
        {
            "tradable": [1, 1, 1, 0, 1],
            "net_return_after_cost": [0.004, 0.003, 0.002, -0.001, 0.003],
            "rr_adequacy": [1, 1, 1, 1, 1],
        }
    )
    ranked = score_candidates_with_ranker(candidates, labels, market_frame=frame)
    required = {
        "behavioral_fingerprint",
        "candidate_class",
        "entry_overlap_score",
        "clustering_risk",
        "aggregate_risk",
        "regime_activation_map",
    }
    assert required.issubset(ranked.columns)
    by_id = ranked.set_index("candidate_id")
    assert float(by_id.loc["dup_a", "overlap_duplication_risk"]) > float(by_id.loc["diverse_c", "overlap_duplication_risk"])
    assert float(by_id.loc["dup_b", "rank_score"]) < float(by_id.loc["diverse_c", "rank_score"])
    assert str(by_id.loc["diverse_c", "candidate_class"]) == "promising_but_unproven"


def test_stage79_behavior_fingerprints_bound_work_but_preserve_family_diversity() -> None:
    frame = _bars()
    rows = []
    for idx in range(80):
        rows.append(
            {
                "candidate_id": f"cont_{idx}",
                "family": "structure_pullback_continuation",
                "beam_score": 0.8 - (idx * 0.001),
                "exp_lcb_proxy": 0.008,
                "cost_edge_proxy": 0.006,
                "time_stop_bars": 12,
                "rr_model": {"first_target_rr": 1.6},
                "signal_spec": {"type": "frame_columns", "long_col": "sig_dup"},
            }
        )
    for idx in range(80):
        rows.append(
            {
                "candidate_id": f"rev_{idx}",
                "family": "failed_breakout_reversal",
                "beam_score": 0.78 - (idx * 0.001),
                "exp_lcb_proxy": 0.007,
                "cost_edge_proxy": 0.005,
                "time_stop_bars": 14,
                "rr_model": {"first_target_rr": 1.5},
                "signal_spec": {"type": "frame_columns", "short_col": "sig_alt"},
            }
        )
    profiled = build_behavioral_fingerprints(pd.DataFrame(rows), frame, max_candidates=24, max_bars=48)
    assert len(profiled) <= 24
    assert set(profiled["candidate_id"].str.split("_").str[0]) == {"cont", "rev"}
