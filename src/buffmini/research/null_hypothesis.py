"""Stage-101 null-hypothesis attack against surfaced candidates."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from buffmini.research.campaign import evaluate_scope_campaign, select_campaign_families
from buffmini.research.modes import build_mode_context
from buffmini.utils.hashing import stable_hash
from buffmini.validation import load_candidate_market_frame, run_candidate_replay, run_custom_signal_replay
from buffmini.validation.candidate_runtime import build_candidate_signal_series


def run_null_hypothesis_attack(
    config: dict[str, Any],
    *,
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    candidate_limit: int = 3,
) -> dict[str, Any]:
    families = select_campaign_families(limit=6)
    scope = evaluate_scope_campaign(
        config=config,
        symbol=symbol,
        timeframe=timeframe,
        families=families,
        candidate_limit=int(max(1, candidate_limit)),
        requested_mode="evaluation",
        auto_pin_resolved_end=True,
        relax_continuity=False,
        evaluate_transfer=False,
        ranking_profile="stage99_quality_acceleration",
        data_source_override="canonical_eval",
    )
    if bool(scope.get("blocked", False)):
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "candidate_count_reviewed": 0,
            "candidate_rows": [],
            "comparison_rows": [],
            "control_win_counts": {},
            "candidate_beats_all_controls_count": 0,
            "candidate_beats_majority_controls_count": 0,
            "blocked": True,
            "blocked_reason": str(scope.get("blocked_reason", "")),
            "summary_hash": stable_hash({"blocked_reason": scope.get("blocked_reason", "")}, length=16),
        }

    ranked_lookup = {
        str(row.get("candidate_id", "")): dict(row)
        for row in getattr(scope.get("ranked_frame"), "to_dict", lambda **_: [])(orient="records")
    }
    chosen = _select_candidates(scope=scope, limit=int(max(1, candidate_limit)))
    effective_cfg, _ = build_mode_context(config, requested_mode="evaluation", auto_pin_resolved_end=True)
    frame, market_meta = load_candidate_market_frame(effective_cfg, symbol=symbol, timeframe=timeframe)

    candidate_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    control_win_counts: Counter[str] = Counter()
    for chosen_row in chosen:
        candidate = dict(ranked_lookup.get(str(chosen_row.get("candidate_id", "")), {}))
        if not candidate:
            continue
        base = run_candidate_replay(candidate=candidate, config=effective_cfg, symbol=symbol, frame=frame, market_meta=market_meta)
        base_metrics = dict(base.get("metrics", {}))
        base_signal = build_candidate_signal_series(frame.copy(), candidate)
        control_results = []
        for control_name, signal in build_control_signals(frame=frame, base_signal=base_signal, seed_key=str(candidate.get("candidate_id", ""))):
            replay = run_custom_signal_replay(
                candidate=candidate,
                config=effective_cfg,
                symbol=symbol,
                signal=signal,
                frame=frame,
                market_meta=market_meta,
                variant_label=control_name,
            )
            control_metrics = dict(replay.get("metrics", {}))
            control_row = {
                "candidate_id": str(candidate.get("candidate_id", "")),
                "family": str(candidate.get("family", "")),
                "control": control_name,
                "candidate_exp_lcb": float(base_metrics.get("exp_lcb", -1.0)),
                "control_exp_lcb": float(control_metrics.get("exp_lcb", -1.0)),
                "candidate_expectancy": float(base_metrics.get("expectancy", 0.0)),
                "control_expectancy": float(control_metrics.get("expectancy", 0.0)),
                "candidate_profit_factor": float(base_metrics.get("profit_factor", 0.0)),
                "control_profit_factor": float(control_metrics.get("profit_factor", 0.0)),
                "candidate_trade_count": int(base_metrics.get("trade_count", 0)),
                "control_trade_count": int(control_metrics.get("trade_count", 0)),
                "candidate_beats_control": candidate_beats_control(base_metrics, control_metrics),
            }
            control_results.append(control_row)
            comparison_rows.append(control_row)
            if not bool(control_row["candidate_beats_control"]):
                control_win_counts[str(control_name)] += 1
        candidate_rows.append(
            {
                "candidate_id": str(candidate.get("candidate_id", "")),
                "family": str(candidate.get("family", "")),
                "expected_regime": str(candidate.get("expected_regime", "")),
                "base_exp_lcb": float(base_metrics.get("exp_lcb", -1.0)),
                "base_expectancy": float(base_metrics.get("expectancy", 0.0)),
                "base_profit_factor": float(base_metrics.get("profit_factor", 0.0)),
                "controls_beaten": int(sum(1 for row in control_results if bool(row.get("candidate_beats_control", False)))),
                "controls_total": int(len(control_results)),
                "beats_all_controls": bool(control_results and all(bool(row.get("candidate_beats_control", False)) for row in control_results)),
                "beats_majority_controls": bool(control_results and sum(bool(row.get("candidate_beats_control", False)) for row in control_results) >= (len(control_results) // 2 + 1)),
            }
        )

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "candidate_count_reviewed": int(len(candidate_rows)),
        "candidate_rows": candidate_rows,
        "comparison_rows": comparison_rows,
        "control_win_counts": dict(control_win_counts),
        "candidate_beats_all_controls_count": int(sum(1 for row in candidate_rows if bool(row.get("beats_all_controls", False)))),
        "candidate_beats_majority_controls_count": int(sum(1 for row in candidate_rows if bool(row.get("beats_majority_controls", False)))),
        "blocked": False,
        "blocked_reason": "",
        "summary_hash": stable_hash(
            {
                "candidate_rows": candidate_rows,
                "control_win_counts": dict(control_win_counts),
            },
            length=16,
        ),
    }


def build_control_signals(
    *,
    frame: pd.DataFrame,
    base_signal: pd.Series,
    seed_key: str,
) -> list[tuple[str, pd.Series]]:
    return [
        ("inverted_signal", (-base_signal).astype(int)),
        ("randomized_signal", randomized_signal(base_signal, seed_key=seed_key)),
        ("delayed_fake_signal", base_signal.shift(3).fillna(0).astype(int)),
        ("momentum_baseline", momentum_baseline_signal(frame)),
        ("mean_reversion_baseline", mean_reversion_baseline_signal(frame)),
    ]


def randomized_signal(base_signal: pd.Series, *, seed_key: str) -> pd.Series:
    signal = pd.Series(base_signal).fillna(0).astype(int)
    active_positions = np.flatnonzero(signal.to_numpy() != 0)
    values = signal.to_numpy(copy=True)
    if len(active_positions) == 0:
        return signal.astype(int)
    rng = np.random.default_rng(abs(hash(str(seed_key))) % (2**32))
    replacement_positions = np.sort(rng.choice(len(signal), size=len(active_positions), replace=False))
    replacement_values = values[active_positions].copy()
    rng.shuffle(replacement_values)
    values[:] = 0
    values[replacement_positions] = replacement_values
    return pd.Series(values, index=signal.index, dtype=int)


def momentum_baseline_signal(frame: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(frame.get("close"), errors="coerce").astype(float)
    sma20 = close.rolling(20, min_periods=10).mean()
    sma50 = close.rolling(50, min_periods=20).mean()
    long_cond = (close > sma50) & (sma20 > sma50)
    short_cond = (close < sma50) & (sma20 < sma50)
    return pd.Series(np.where(long_cond, 1, np.where(short_cond, -1, 0)), index=frame.index, dtype=int).shift(1).fillna(0).astype(int)


def mean_reversion_baseline_signal(frame: pd.DataFrame) -> pd.Series:
    close = pd.to_numeric(frame.get("close"), errors="coerce").astype(float)
    mean20 = close.rolling(20, min_periods=10).mean()
    std20 = close.rolling(20, min_periods=10).std(ddof=0).replace(0.0, np.nan)
    zscore = ((close - mean20) / std20).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    long_cond = zscore <= -1.25
    short_cond = zscore >= 1.25
    return pd.Series(np.where(long_cond, 1, np.where(short_cond, -1, 0)), index=frame.index, dtype=int).shift(1).fillna(0).astype(int)


def candidate_beats_control(candidate_metrics: dict[str, Any], control_metrics: dict[str, Any]) -> bool:
    return bool(
        float(candidate_metrics.get("exp_lcb", -1.0)) > float(control_metrics.get("exp_lcb", -1.0))
        and float(candidate_metrics.get("expectancy", 0.0)) >= float(control_metrics.get("expectancy", 0.0))
        and float(candidate_metrics.get("profit_factor", 0.0)) >= float(control_metrics.get("profit_factor", 0.0))
    )


def _select_candidates(*, scope: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    evaluations = list(scope.get("evaluations", []))
    promising = [row for row in evaluations if str(row.get("final_class", "")) == "promising_but_unproven"]
    chosen = promising or evaluations
    return chosen[: max(1, int(limit))]
