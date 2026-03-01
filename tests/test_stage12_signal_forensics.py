from __future__ import annotations

import pandas as pd

from buffmini.stage12.forensics import extract_trade_context_rows, summarize_signal_forensics


def _sample_frame() -> pd.DataFrame:
    ts = pd.date_range("2026-01-01", periods=12, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100 + i for i in range(12)],
            "high": [101 + i for i in range(12)],
            "low": [99 + i for i in range(12)],
            "close": [100 + i for i in range(12)],
            "atr_14": [1.0] * 12,
            "atr_pct_rank_252": [0.6] * 12,
            "trend_strength_stage10": [0.015] * 12,
            "regime_label_stage10": ["TREND"] * 12,
            "score_trend": [0.8] * 12,
            "score_range": [0.2] * 12,
            "score_vol_expansion": [0.4] * 12,
            "score_chop": [0.1] * 12,
        }
    )
    return frame


def test_extract_trade_context_rows_non_empty() -> None:
    frame = _sample_frame()
    trades = pd.DataFrame(
        [
            {
                "entry_time": frame["timestamp"].iloc[1],
                "exit_time": frame["timestamp"].iloc[4],
                "entry_price": 101.0,
                "side": "long",
                "pnl": 1.0,
            },
            {
                "entry_time": frame["timestamp"].iloc[5],
                "exit_time": frame["timestamp"].iloc[7],
                "entry_price": 105.0,
                "side": "long",
                "pnl": -1.0,
            },
        ]
    )
    rows = extract_trade_context_rows(
        combo_key="abc",
        symbol="BTC/USDT",
        timeframe="1h",
        strategy="Trend Pullback",
        strategy_key="Trend_Pullback",
        strategy_source="stage06",
        exit_type="fixed_atr",
        cost_level="realistic",
        frame=frame,
        trades=trades,
        context_cfg={},
    )
    assert len(rows) == 2
    assert all("context_score" in row for row in rows)
    assert all(0.0 <= float(row["context_score"]) <= 1.0 for row in rows)


def test_signal_forensics_detects_separation() -> None:
    rows = []
    for idx in range(60):
        rows.append(
            {
                "strategy": "A",
                "is_winner": True,
                "context_score": 0.8,
                "regime_alignment": 0.8,
                "volatility_percentile": 0.6,
                "ATR_ratio": 0.01,
                "holding_duration": 5,
                "MFE": 0.03,
                "MAE": 0.01,
            }
        )
    for idx in range(60):
        rows.append(
            {
                "strategy": "A",
                "is_winner": False,
                "context_score": 0.3,
                "regime_alignment": 0.3,
                "volatility_percentile": 0.5,
                "ATR_ratio": 0.01,
                "holding_duration": 5,
                "MFE": 0.01,
                "MAE": 0.03,
            }
        )
    fp, summary, by_strategy = summarize_signal_forensics(
        trade_context=pd.DataFrame(rows),
        context_cfg={"separation_effect_size_threshold": 0.10, "min_samples": 30},
    )
    assert not fp.empty
    assert not by_strategy.empty
    assert summary["context_separation_detected"] is True
    assert summary["final_stage12_2_verdict"] == "CONTEXT_DEPENDENT_EDGE"

