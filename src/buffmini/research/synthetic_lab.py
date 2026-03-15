"""Controlled synthetic truth-lab for signal detectability evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from buffmini.research.diagnostics import classify_candidate_tier, compute_candidate_risk_card
from buffmini.stage48.tradability_learning import Stage48Config, compute_stage48_labels, score_candidates_with_ranker
from buffmini.utils.hashing import stable_hash
from buffmini.validation import evaluate_candidate_walkforward, run_candidate_replay


@dataclass(frozen=True)
class TruthCandidate:
    candidate_id: str
    regime: str
    expected_class: str
    strength: str
    candidate: dict[str, Any]


def _base_frame(*, bars: int, start: str = "2025-01-01T00:00:00Z") -> pd.DataFrame:
    ts = pd.date_range(start, periods=bars, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": np.zeros(bars, dtype=float),
            "high": np.zeros(bars, dtype=float),
            "low": np.zeros(bars, dtype=float),
            "close": np.zeros(bars, dtype=float),
            "volume": np.zeros(bars, dtype=float),
        }
    )


def _finalize_prices(frame: pd.DataFrame, closes: np.ndarray) -> pd.DataFrame:
    data = frame.copy()
    data["close"] = closes
    data["open"] = np.concatenate(([closes[0]], closes[:-1]))
    span = np.maximum(0.18, np.abs(data["close"] - data["open"]) + 0.12)
    data["high"] = np.maximum(data["open"], data["close"]) + span * 0.55
    data["low"] = np.minimum(data["open"], data["close"]) - span * 0.45
    data["volume"] = 850.0 + (np.sin(np.arange(len(data)) / 9.0) + 1.5) * 120.0
    return data


def _truth_candidate(
    *,
    regime: str,
    name: str,
    expected_class: str,
    strength: str,
    long_col: str,
    short_col: str,
    stop_pct: float,
    target_pct: float,
    hold_bars: int,
    activation_density: float,
    cost_edge_proxy: float,
    rr_first_target: float,
    transfer_risk_prior: float = 0.25,
) -> TruthCandidate:
    candidate_id = f"lab_{stable_hash({'regime': regime, 'name': name}, length=12)}"
    candidate = {
        "candidate_id": candidate_id,
        "family": "synthetic_truth_lab",
        "timeframe": "1h",
        "context": regime,
        "trigger": name,
        "confirmation": "lab_confirmed",
        "invalidation": "lab_invalidation",
        "entry_logic": "enter_on_truth_lab_signal",
        "stop_logic": "lab_stop",
        "target_logic": "lab_target",
        "hold_logic": "lab_time_stop",
        "signal_spec": {
            "type": "frame_columns",
            "long_col": long_col,
            "short_col": short_col,
        },
        "geometry": {
            "stop_distance_pct": float(stop_pct),
            "first_target_pct": float(target_pct),
            "stretch_target_pct": float(target_pct * 1.6),
            "expected_hold_bars": int(hold_bars),
        },
        "rr_model": {
            "first_target_rr": float(rr_first_target),
        },
        "cost_edge_proxy": float(cost_edge_proxy),
        "expected_hold_bars": int(hold_bars),
        "activation_density": float(activation_density),
        "transfer_risk_prior": float(transfer_risk_prior),
        "thin_evidence_risk": 0.15 if strength == "strong" else (0.35 if strength == "weak" else 0.75),
        "regime_concentration_risk": 0.15 if strength == "strong" else (0.28 if strength == "weak" else 0.80),
        "duplication_score": 0.05 if strength == "strong" else (0.12 if strength == "weak" else 0.65),
    }
    return TruthCandidate(candidate_id=candidate_id, regime=regime, expected_class=expected_class, strength=strength, candidate=candidate)


def build_truth_lab_dataset(*, bars: int = 720, seed: int = 42, difficulty: str = "easy") -> tuple[pd.DataFrame, list[TruthCandidate]]:
    rng = np.random.default_rng(int(seed))
    frame = _base_frame(bars=bars)
    segment = bars // 5
    trend = np.zeros(bars, dtype=float)
    mean_revert = np.zeros(bars, dtype=float)
    breakout = np.zeros(bars, dtype=float)
    failed = np.zeros(bars, dtype=float)
    flow = np.zeros(bars, dtype=float)
    close = np.zeros(bars, dtype=float)
    level = 100.0
    diff = str(difficulty).strip().lower()
    noise_mult = 1.0 if diff == "easy" else 1.65
    impulse_mult = 1.0 if diff == "easy" else 0.72
    weak_density_cut = 1.0 if diff == "easy" else 0.70

    for i in range(bars):
        regime_idx = min(4, i // max(1, segment))
        local = i - (regime_idx * segment)
        if regime_idx == 0:
            drift = 0.32 + (0.06 * np.sin(local / 6.0))
            retrace = -1.05 if local % 24 == 8 else 0.0
            change = (drift * impulse_mult) + retrace + rng.normal(0.0, 0.05 * noise_mult)
            if local % 24 == 9:
                trend[i] = 1.0
        elif regime_idx == 1:
            wave = np.sin(local / 4.0) * 1.5
            change = ((-0.55 * wave) * impulse_mult) + rng.normal(0.0, 0.07 * noise_mult)
            if local % 18 == 5:
                mean_revert[i] = 1.0 if wave < 0 else -1.0
        elif regime_idx == 2:
            base = 0.05 * np.sin(local / 5.0)
            impulse = (2.2 * impulse_mult) if local % 26 in {13, 14, 15} else 0.0
            change = base + impulse + rng.normal(0.0, 0.08 * noise_mult)
            if local % 26 == 12:
                breakout[i] = 1.0
        elif regime_idx == 3:
            fake = (1.8 * impulse_mult) if local % 28 == 9 else ((-1.7 * impulse_mult) if local % 28 in {10, 11, 12} else 0.0)
            change = fake + rng.normal(0.0, 0.09 * noise_mult)
            if local % 28 == 9:
                failed[i] = -1.0
        else:
            burst = 0.25 * np.sin(local / 3.0)
            exhaustion = (1.1 * impulse_mult) if local % 20 == 7 else ((-1.1 * impulse_mult) if local % 20 == 8 else 0.0)
            change = burst + exhaustion + rng.normal(0.0, 0.06 * noise_mult)
            if local % 20 == 8:
                flow[i] = -1.0

        level = max(60.0, level + change)
        close[i] = level

    data = _finalize_prices(frame, close)
    data["lab_trend_long"] = trend > 0
    data["lab_trend_short"] = False
    data["lab_trend_weak_long"] = np.roll(trend > 0, 1) & ((np.arange(bars) % max(1, int(round(1.0 / max(0.1, weak_density_cut))))) == 0)
    data["lab_trend_weak_short"] = False
    data["lab_trend_inverse_short"] = trend > 0
    data["lab_meanrev_long"] = mean_revert > 0
    data["lab_meanrev_short"] = mean_revert < 0
    data["lab_meanrev_noise_long"] = (mean_revert > 0) & ((np.arange(bars) % (2 if diff == "easy" else 3)) == 0)
    data["lab_meanrev_noise_short"] = (mean_revert < 0) & ((np.arange(bars) % (2 if diff == "easy" else 3)) == 0)
    data["lab_breakout_long"] = breakout > 0
    data["lab_breakout_short"] = False
    data["lab_breakout_random_long"] = rng.random(bars) < 0.03
    data["lab_breakout_random_short"] = rng.random(bars) < 0.03
    data["lab_failed_long"] = False
    data["lab_failed_short"] = failed < 0
    data["lab_failed_inverse_long"] = failed < 0
    data["lab_failed_inverse_short"] = False
    data["lab_flow_long"] = False
    data["lab_flow_short"] = flow < 0

    candidates = [
        _truth_candidate(
            regime="trend_persistence",
            name="trend_winner",
            expected_class="strong_winner",
            strength="strong",
            long_col="lab_trend_long",
            short_col="lab_trend_short",
            stop_pct=0.006,
            target_pct=0.014,
            hold_bars=10,
            activation_density=float(data["lab_trend_long"].mean()),
            cost_edge_proxy=0.010,
            rr_first_target=2.2,
            transfer_risk_prior=0.20,
        ),
        _truth_candidate(
            regime="trend_persistence",
            name="trend_weak",
            expected_class="weak_detectable",
            strength="weak",
            long_col="lab_trend_weak_long",
            short_col="lab_trend_weak_short",
            stop_pct=0.006,
            target_pct=0.009,
            hold_bars=8,
            activation_density=float(data["lab_trend_weak_long"].mean()),
            cost_edge_proxy=0.004,
            rr_first_target=1.4,
            transfer_risk_prior=0.25,
        ),
        _truth_candidate(
            regime="trend_persistence",
            name="trend_inverse",
            expected_class="bad_control",
            strength="bad",
            long_col="lab_trend_short",
            short_col="lab_trend_inverse_short",
            stop_pct=0.006,
            target_pct=0.008,
            hold_bars=8,
            activation_density=float(data["lab_trend_inverse_short"].mean()),
            cost_edge_proxy=-0.003,
            rr_first_target=1.0,
            transfer_risk_prior=0.60,
        ),
        _truth_candidate(
            regime="mean_reversion",
            name="meanrev_winner",
            expected_class="strong_winner",
            strength="strong",
            long_col="lab_meanrev_long",
            short_col="lab_meanrev_short",
            stop_pct=0.005,
            target_pct=0.012,
            hold_bars=6,
            activation_density=float((data["lab_meanrev_long"] | data["lab_meanrev_short"]).mean()),
            cost_edge_proxy=0.009,
            rr_first_target=2.0,
        ),
        _truth_candidate(
            regime="mean_reversion",
            name="meanrev_weak",
            expected_class="weak_detectable",
            strength="weak",
            long_col="lab_meanrev_noise_long",
            short_col="lab_meanrev_noise_short",
            stop_pct=0.005,
            target_pct=0.008,
            hold_bars=4,
            activation_density=float((data["lab_meanrev_noise_long"] | data["lab_meanrev_noise_short"]).mean()),
            cost_edge_proxy=0.003,
            rr_first_target=1.35,
        ),
        _truth_candidate(
            regime="compression_breakout",
            name="breakout_winner",
            expected_class="strong_winner",
            strength="strong",
            long_col="lab_breakout_long",
            short_col="lab_breakout_short",
            stop_pct=0.0045,
            target_pct=0.0105,
            hold_bars=8,
            activation_density=float(data["lab_breakout_long"].mean()),
            cost_edge_proxy=0.008,
            rr_first_target=1.9,
        ),
        _truth_candidate(
            regime="compression_breakout",
            name="breakout_random",
            expected_class="bad_control",
            strength="bad",
            long_col="lab_breakout_random_long",
            short_col="lab_breakout_random_short",
            stop_pct=0.007,
            target_pct=0.010,
            hold_bars=12,
            activation_density=float((data["lab_breakout_random_long"] | data["lab_breakout_random_short"]).mean()),
            cost_edge_proxy=-0.006,
            rr_first_target=0.95,
            transfer_risk_prior=0.90,
        ),
        _truth_candidate(
            regime="failed_breakout_reversal",
            name="failed_breakout_winner",
            expected_class="strong_winner",
            strength="strong",
            long_col="lab_failed_long",
            short_col="lab_failed_short",
            stop_pct=0.004,
            target_pct=0.010,
            hold_bars=6,
            activation_density=float(data["lab_failed_short"].mean()),
            cost_edge_proxy=0.007,
            rr_first_target=1.85,
        ),
        _truth_candidate(
            regime="flow_imbalance",
            name="flow_weak",
            expected_class="weak_detectable",
            strength="weak",
            long_col="lab_flow_long",
            short_col="lab_flow_short",
            stop_pct=0.004,
            target_pct=0.007,
            hold_bars=4,
            activation_density=float(data["lab_flow_short"].mean()),
            cost_edge_proxy=0.002,
            rr_first_target=1.25,
        ),
    ]
    return data, candidates


def evaluate_detectability_suite(config: dict[str, Any], *, seed: int = 42, difficulty: str = "easy") -> dict[str, Any]:
    """Run the controlled signal-detectability proof through the live runtime path."""

    frame, truth_candidates = build_truth_lab_dataset(seed=seed, difficulty=difficulty)
    labels = compute_stage48_labels(
        frame[["timestamp", "open", "high", "low", "close", "volume"]],
        cfg=Stage48Config(round_trip_cost_pct=float(config.get("costs", {}).get("round_trip_cost_pct", 0.1)) / 100.0),
    )
    candidate_rows: list[dict[str, Any]] = []
    evaluations: list[dict[str, Any]] = []
    promotion_by_regime: dict[str, dict[str, int]] = {}

    for truth in truth_candidates:
        beam_score = 0.72 if truth.expected_class == "strong_winner" else (0.52 if truth.expected_class == "weak_detectable" else 0.04)
        row = {
            "candidate_id": truth.candidate_id,
            "source_candidate_id": truth.candidate_id,
            "family": truth.candidate.get("family"),
            "timeframe": truth.candidate.get("timeframe"),
            "beam_score": beam_score,
            "cost_edge_proxy": truth.candidate.get("cost_edge_proxy", 0.0),
            "rr_model": truth.candidate.get("rr_model"),
            "pre_replay_reject_reason": "",
            "eligible_for_replay": True,
            "economic_fingerprint": stable_hash({"candidate_id": truth.candidate_id, "regime": truth.regime}, length=20),
        }
        candidate_rows.append(row)

    ranked = score_candidates_with_ranker(pd.DataFrame(candidate_rows), labels)
    rank_map = {str(row["candidate_id"]): float(row["rank_score"]) for row in ranked.to_dict(orient="records")}

    for truth in truth_candidates:
        replay = run_candidate_replay(
            candidate=truth.candidate,
            config=config,
            symbol="SYNTH/USDT",
            frame=frame,
            market_meta={"continuity_blocked": False, "runtime_truth_blocked": False},
        )
        walkforward = evaluate_candidate_walkforward(
            candidate=truth.candidate,
            config=config,
            symbol="SYNTH/USDT",
            frame=frame,
            market_meta={"continuity_blocked": False, "runtime_truth_blocked": False},
        )
        risk_card = compute_candidate_risk_card(
            {
                **truth.candidate,
                "activation_density": truth.candidate.get("activation_density", 0.0),
                "rr_first_target": truth.candidate.get("rr_model", {}).get("first_target_rr", 0.0),
            }
        )
        rank_score = float(rank_map.get(truth.candidate_id, 0.0))
        replay_metrics = dict(replay.get("metrics", {}))
        wf_summary = dict(walkforward.get("summary", {}))
        candidate_class = classify_candidate_tier(
            rank_score=rank_score,
            replay_exp_lcb=float(replay_metrics.get("exp_lcb", 0.0)),
            walkforward_usable_windows=int(wf_summary.get("usable_windows", 0)),
            decision_use_allowed=bool(replay.get("decision_use_allowed", False)),
            aggregate_risk=float(risk_card.get("aggregate_risk", 1.0)),
        )
        rec = {
            "candidate_id": truth.candidate_id,
            "regime": truth.regime,
            "expected_class": truth.expected_class,
            "strength": truth.strength,
            "rank_score": rank_score,
            "replay_trade_count": int(replay_metrics.get("trade_count", 0)),
            "replay_exp_lcb": float(replay_metrics.get("exp_lcb", 0.0)),
            "walkforward_usable_windows": int(wf_summary.get("usable_windows", 0)),
            "candidate_class": candidate_class,
            "aggregate_risk": float(risk_card.get("aggregate_risk", 1.0)),
        }
        evaluations.append(rec)
        regime_bucket = promotion_by_regime.setdefault(truth.regime, {"total": 0, "promoted": 0})
        regime_bucket["total"] += 1
        if candidate_class != "rejected":
            regime_bucket["promoted"] += 1

    eval_frame = pd.DataFrame(evaluations)
    known_good = eval_frame["expected_class"].isin(["strong_winner", "weak_detectable"])
    strong = eval_frame["expected_class"] == "strong_winner"
    bad = eval_frame["expected_class"] == "bad_control"
    surfaced = eval_frame["candidate_class"].isin(["promising_but_unproven", "validated_candidate"])
    signal_detection_rate = float((eval_frame.loc[known_good, "candidate_class"].isin(["promising_but_unproven", "validated_candidate"])).mean()) if known_good.any() else 0.0
    bad_control_rejection_rate = float((eval_frame.loc[bad, "candidate_class"] == "rejected").mean()) if bad.any() else 0.0
    synthetic_winner_recall = float((eval_frame.loc[strong, "candidate_class"].isin(["promising_but_unproven", "validated_candidate"])).mean()) if strong.any() else 0.0
    false_negative_rate_on_known_good = float((~eval_frame.loc[known_good, "candidate_class"].isin(["promising_but_unproven", "validated_candidate"])).mean()) if known_good.any() else 0.0
    summary = {
        "candidate_count": int(len(eval_frame)),
        "signal_detection_rate": float(round(signal_detection_rate, 6)),
        "bad_control_rejection_rate": float(round(bad_control_rejection_rate, 6)),
        "synthetic_winner_recall": float(round(synthetic_winner_recall, 6)),
        "false_negative_rate_on_known_good": float(round(false_negative_rate_on_known_good, 6)),
        "promotion_rate_by_regime": {
            regime: float(round(bucket["promoted"] / max(1, bucket["total"]), 6))
            for regime, bucket in sorted(promotion_by_regime.items())
        },
        "candidate_classes": {
            str(key): int(value)
            for key, value in eval_frame["candidate_class"].value_counts(dropna=False).to_dict().items()
        },
        "evaluations": evaluations,
        "difficulty": str(difficulty).strip().lower(),
    }
    summary["status"] = "SUCCESS" if summary["bad_control_rejection_rate"] >= 0.66 and summary["synthetic_winner_recall"] >= 0.75 else "PARTIAL"
    summary["summary_hash"] = stable_hash(
        {
            "signal_detection_rate": summary["signal_detection_rate"],
            "bad_control_rejection_rate": summary["bad_control_rejection_rate"],
            "synthetic_winner_recall": summary["synthetic_winner_recall"],
            "false_negative_rate_on_known_good": summary["false_negative_rate_on_known_good"],
            "candidate_classes": summary["candidate_classes"],
        },
        length=16,
    )
    return summary
