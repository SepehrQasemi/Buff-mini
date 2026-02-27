"""Stage-2 portfolio tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from buffmini.data.features import calculate_features
from buffmini.discovery.funnel import _generate_synthetic_ohlcv
from buffmini.portfolio.builder import (
    CandidateWindowResult,
    EvaluationWindow,
    Stage1CandidateRecord,
    build_equal_weights,
    build_portfolio_return_series,
    build_volatility_weights,
    evaluate_candidate_windows,
    resolve_evaluation_window,
    slice_time_window,
)
from buffmini.portfolio.correlation import (
    build_correlation_matrix,
    effective_number_of_strategies,
)


def _candidate_record() -> Stage1CandidateRecord:
    return Stage1CandidateRecord(
        candidate_id="cand_test",
        result_tier="Tier A",
        strategy_name="Trend Pullback",
        family="TrendPullback",
        gating_mode="none",
        exit_mode="fixed_atr",
        params={
            "channel_period": 55,
            "ema_fast": 20,
            "ema_slow": 200,
            "rsi_long_entry": 35,
            "rsi_short_entry": 65,
            "bollinger_period": 20,
            "bollinger_std": 2.0,
            "atr_sl_multiplier": 1.5,
            "atr_tp_multiplier": 3.0,
            "trailing_atr_k": 1.5,
            "max_holding_bars": 24,
            "regime_gate_long": False,
            "regime_gate_short": False,
        },
        holdout_months_used=3,
        effective_edge=10.0,
        exact_holdout_range=None,
    )


def _candidate_window_result(candidate_id: str, returns: pd.Series) -> CandidateWindowResult:
    equity = 10_000.0 * (1.0 + returns).cumprod()
    exposure = pd.Series(0.5, index=returns.index, dtype=float)
    window = EvaluationWindow(
        holdout_start=returns.index[0],
        holdout_end=returns.index[-2],
        forward_end=returns.index[-1],
        window_mode="shifted_for_forward_window",
    )
    record = _candidate_record()
    record = Stage1CandidateRecord(
        candidate_id=candidate_id,
        result_tier=record.result_tier,
        strategy_name=record.strategy_name,
        family=record.family,
        gating_mode=record.gating_mode,
        exit_mode=record.exit_mode,
        params=record.params,
        holdout_months_used=record.holdout_months_used,
        effective_edge=record.effective_edge,
        exact_holdout_range=record.exact_holdout_range,
    )
    trades = pd.DataFrame(
        {
            "candidate_id": [candidate_id, candidate_id],
            "candidate_pnl": [100.0, -50.0],
            "entry_time": [returns.index[0], returns.index[2]],
            "exit_time": [returns.index[1], returns.index[3]],
        }
    )
    return CandidateWindowResult(
        record=record,
        window=window,
        holdout_returns=returns,
        holdout_equity=equity,
        holdout_exposure=exposure,
        holdout_trades=trades,
        forward_returns=returns.iloc[:2],
        forward_equity=equity.iloc[:2],
        forward_exposure=exposure.iloc[:2],
        forward_trades=trades.iloc[:1].copy(),
    )


def _feature_data() -> dict[str, pd.DataFrame]:
    btc = calculate_features(_generate_synthetic_ohlcv("BTC/USDT", "2024-01-01T00:00:00Z", bars=2400, seed=7))
    eth = calculate_features(_generate_synthetic_ohlcv("ETH/USDT", "2024-01-01T00:00:00Z", bars=2400, seed=11))
    return {"BTC/USDT": btc, "ETH/USDT": eth}


def test_portfolio_weights_sum_to_one() -> None:
    index = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
    returns_a = pd.Series([0.0, 0.01, -0.005, 0.002], index=index, dtype=float)
    returns_b = pd.Series([0.0, 0.005, 0.002, -0.001], index=index, dtype=float)
    candidate_results = {
        "a": _candidate_window_result("a", returns_a),
        "b": _candidate_window_result("b", returns_b),
    }

    equal_weights = build_equal_weights(["a", "b"])
    vol_weights = build_volatility_weights(candidate_results)

    assert abs(sum(equal_weights.values()) - 1.0) < 1e-12
    assert abs(sum(vol_weights.values()) - 1.0) < 1e-12


def test_correlation_matrix_symmetric() -> None:
    index = pd.date_range("2025-01-01", periods=5, freq="h", tz="UTC")
    matrix = build_correlation_matrix(
        {
            "a": pd.Series([0.0, 1.0, 0.0, -1.0, 0.5], index=index, dtype=float),
            "b": pd.Series([0.0, 0.5, 0.1, -0.4, 0.2], index=index, dtype=float),
        }
    )
    assert matrix.equals(matrix.T)


def test_effective_number_of_strategies_valid() -> None:
    effective_n = effective_number_of_strategies({"a": 0.5, "b": 0.5})
    assert effective_n == 2.0


def test_equal_weight_portfolio_matches_manual_combination() -> None:
    index = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
    series_by_candidate = {
        "a": pd.Series([0.0, 0.01, 0.02, -0.01], index=index, dtype=float),
        "b": pd.Series([0.0, 0.03, -0.01, 0.02], index=index, dtype=float),
    }
    weights = build_equal_weights(["a", "b"])
    combined = build_portfolio_return_series(series_by_candidate, weights)
    manual = (series_by_candidate["a"] + series_by_candidate["b"]) / 2.0
    pd.testing.assert_series_equal(combined, manual, check_names=False)


def test_forward_window_does_not_overlap_holdout() -> None:
    feature_data = _feature_data()
    window = resolve_evaluation_window(record=_candidate_record(), feature_data=feature_data, forward_days=30)
    btc = feature_data["BTC/USDT"]
    holdout = slice_time_window(btc, window.holdout_start, window.holdout_end, True, True)
    forward = slice_time_window(btc, window.holdout_end, window.forward_end, False, True)

    holdout_ts = pd.to_datetime(holdout["timestamp"], utc=True)
    forward_ts = pd.to_datetime(forward["timestamp"], utc=True)

    assert holdout_ts.max() < forward_ts.min()


def test_stage2_candidate_evaluation_has_no_future_leakage() -> None:
    feature_data = _feature_data()
    record = _candidate_record()
    baseline = evaluate_candidate_windows(
        record=record,
        feature_data=feature_data,
        initial_capital=10_000.0,
        round_trip_cost_pct=0.1,
        slippage_pct=0.0005,
        forward_days=30,
    )

    shocked_feature_data: dict[str, pd.DataFrame] = {}
    for symbol, frame in feature_data.items():
        shocked = frame.copy()
        holdout_end = baseline.window.holdout_end
        timestamps = pd.to_datetime(shocked["timestamp"], utc=True)
        future_mask = timestamps > holdout_end
        shocked.loc[future_mask, "close"] = shocked.loc[future_mask, "close"] * 3.0
        shocked.loc[future_mask, "high"] = shocked.loc[future_mask, "high"] * 3.0
        shocked.loc[future_mask, "low"] = shocked.loc[future_mask, "low"] * 0.5
        shocked_feature_data[symbol] = calculate_features(
            shocked[["timestamp", "open", "high", "low", "close", "volume"]]
        )

    shocked = evaluate_candidate_windows(
        record=record,
        feature_data=shocked_feature_data,
        initial_capital=10_000.0,
        round_trip_cost_pct=0.1,
        slippage_pct=0.0005,
        forward_days=30,
    )

    pd.testing.assert_series_equal(baseline.holdout_returns, shocked.holdout_returns)
