"""Stage-1 candidate generation and search-space helpers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from buffmini.baselines.stage0 import (
    bollinger_mean_reversion,
    donchian_breakout,
    range_breakout_with_ema_trend_filter,
    rsi_mean_reversion,
    trend_pullback,
)
from buffmini.types import StrategySpec
from buffmini.utils.hashing import stable_hash


FAMILY_BUILDERS = {
    "DonchianBreakout": donchian_breakout,
    "RSIMeanReversion": rsi_mean_reversion,
    "TrendPullback": trend_pullback,
    "BollingerMeanReversion": bollinger_mean_reversion,
    "RangeBreakoutTrendFilter": range_breakout_with_ema_trend_filter,
}

ENTRY_CONDITION_COUNT = {
    "DonchianBreakout": 2,
    "RSIMeanReversion": 2,
    "TrendPullback": 3,
    "BollingerMeanReversion": 2,
    "RangeBreakoutTrendFilter": 2,
}


@dataclass(frozen=True)
class Candidate:
    """Single sampled strategy candidate for Stage-1."""

    candidate_id: str
    family: str
    gating_mode: str
    exit_mode: str
    params: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "family": self.family,
            "gating_mode": self.gating_mode,
            "exit_mode": self.exit_mode,
            "params": deepcopy(self.params),
        }


def sample_candidate(index: int, rng: np.random.Generator, search_space: dict[str, Any]) -> Candidate:
    """Sample one candidate from configured search space."""

    family = str(rng.choice(search_space["families"]))
    gating_mode = str(rng.choice(search_space["gating_modes"]))
    exit_mode = str(rng.choice(search_space["exit_modes"]))

    ema_pair = rng.choice(search_space["ema_pairs"])  # shape (2,)
    ema_fast = int(ema_pair[0])
    ema_slow = int(ema_pair[1])

    rsi_long_entry = int(
        rng.integers(
            int(search_space["rsi_long_entry_min"]),
            int(search_space["rsi_long_entry_max"]) + 1,
        )
    )
    rsi_short_entry = int(
        rng.integers(
            int(search_space["rsi_short_entry_min"]),
            int(search_space["rsi_short_entry_max"]) + 1,
        )
    )

    params: dict[str, Any] = {
        "channel_period": int(rng.choice(search_space["donchian_periods"])),
        "ema_fast": ema_fast,
        "ema_slow": ema_slow,
        "rsi_long_entry": rsi_long_entry,
        "rsi_short_entry": rsi_short_entry,
        "bollinger_period": int(search_space["bollinger_period"]),
        "bollinger_std": float(rng.choice(search_space["bollinger_stds"])),
        "atr_sl_multiplier": float(
            rng.uniform(float(search_space["atr_sl_min"]), float(search_space["atr_sl_max"]))
        ),
        "atr_tp_multiplier": float(
            rng.uniform(float(search_space["atr_tp_min"]), float(search_space["atr_tp_max"]))
        ),
        "trailing_atr_k": float(
            rng.uniform(float(search_space["trailing_atr_k_min"]), float(search_space["trailing_atr_k_max"]))
        ),
        "max_holding_bars": int(rng.choice(search_space["max_holding_bars"])),
        "regime_gate_long": bool(rng.integers(0, 2)),
        "regime_gate_short": bool(rng.integers(0, 2)),
    }

    candidate_id = f"cand_{index:06d}_{stable_hash([family, gating_mode, exit_mode, params], length=8)}"

    candidate = Candidate(
        candidate_id=candidate_id,
        family=family,
        gating_mode=gating_mode,
        exit_mode=exit_mode,
        params=params,
    )
    if not candidate_within_search_space(candidate, search_space):
        raise ValueError("Sampled candidate outside configured search space")
    return candidate


def candidate_within_search_space(candidate: Candidate, search_space: dict[str, Any]) -> bool:
    """Check candidate values against configured search space bounds."""

    if candidate.family not in search_space["families"]:
        return False
    if candidate.gating_mode not in search_space["gating_modes"]:
        return False
    if candidate.exit_mode not in search_space["exit_modes"]:
        return False

    p = candidate.params
    if int(p["channel_period"]) not in {int(x) for x in search_space["donchian_periods"]}:
        return False
    if [int(p["ema_fast"]), int(p["ema_slow"])] not in [[int(a), int(b)] for a, b in search_space["ema_pairs"]]:
        return False

    if not int(search_space["rsi_long_entry_min"]) <= int(p["rsi_long_entry"]) <= int(search_space["rsi_long_entry_max"]):
        return False
    if not int(search_space["rsi_short_entry_min"]) <= int(p["rsi_short_entry"]) <= int(search_space["rsi_short_entry_max"]):
        return False

    if int(p["bollinger_period"]) != int(search_space["bollinger_period"]):
        return False
    if float(p["bollinger_std"]) not in {float(x) for x in search_space["bollinger_stds"]}:
        return False

    if not float(search_space["atr_sl_min"]) <= float(p["atr_sl_multiplier"]) <= float(search_space["atr_sl_max"]):
        return False
    if not float(search_space["atr_tp_min"]) <= float(p["atr_tp_multiplier"]) <= float(search_space["atr_tp_max"]):
        return False
    if not float(search_space["trailing_atr_k_min"]) <= float(p["trailing_atr_k"]) <= float(search_space["trailing_atr_k_max"]):
        return False
    if int(p["max_holding_bars"]) not in {int(x) for x in search_space["max_holding_bars"]}:
        return False

    return True


def candidate_to_strategy_spec(candidate: Candidate) -> StrategySpec:
    """Convert candidate to executable strategy specification."""

    base = FAMILY_BUILDERS[candidate.family]()
    params = dict(base.parameters)

    params["regime_gate"] = {
        "long": bool(candidate.params["regime_gate_long"]),
        "short": bool(candidate.params["regime_gate_short"]),
    }

    if candidate.family == "DonchianBreakout":
        params["channel_period"] = int(candidate.params["channel_period"])
    elif candidate.family == "RSIMeanReversion":
        params["rsi_long_entry"] = int(candidate.params["rsi_long_entry"])
        params["rsi_short_entry"] = int(candidate.params["rsi_short_entry"])
    elif candidate.family == "TrendPullback":
        params["ema_fast"] = int(candidate.params["ema_fast"])
        params["ema_slow"] = int(candidate.params["ema_slow"])
        params["rsi_long_entry"] = int(candidate.params["rsi_long_entry"])
        params["rsi_short_entry"] = int(candidate.params["rsi_short_entry"])
    elif candidate.family == "BollingerMeanReversion":
        params["bollinger_period"] = int(candidate.params["bollinger_period"])
        params["bollinger_std"] = float(candidate.params["bollinger_std"])
        params["rsi_long_entry"] = int(candidate.params["rsi_long_entry"])
        params["rsi_short_entry"] = int(candidate.params["rsi_short_entry"])
    elif candidate.family == "RangeBreakoutTrendFilter":
        params["channel_period"] = int(candidate.params["channel_period"])
        params["ema_fast"] = int(candidate.params["ema_fast"])
        params["ema_slow"] = int(candidate.params["ema_slow"])

    return StrategySpec(
        name=base.name,
        entry_rules=base.entry_rules,
        exit_rules=base.exit_rules,
        parameters=params,
    )


def complexity_penalty(candidate: Candidate) -> float:
    """Compute complexity penalty from entry condition count and tuned params."""

    entry_count = ENTRY_CONDITION_COUNT.get(candidate.family, 2)
    tuned_params = 0

    if candidate.family == "DonchianBreakout":
        tuned_params += 1
    elif candidate.family == "RSIMeanReversion":
        tuned_params += 2
    elif candidate.family == "TrendPullback":
        tuned_params += 4
    elif candidate.family == "BollingerMeanReversion":
        tuned_params += 3
    elif candidate.family == "RangeBreakoutTrendFilter":
        tuned_params += 3

    tuned_params += 4  # exit parameters
    tuned_params += 2  # regime-gate directional flags
    tuned_params += 1  # gating mode selector

    return float(entry_count + tuned_params) / 20.0


def perturb_candidate(candidate: Candidate, search_space: dict[str, Any], pct: float = 0.1) -> list[Candidate]:
    """Create +/- perturbations for instability checks."""

    def _scaled(v: float, sign: int) -> float:
        return v * (1.0 + sign * pct)

    variants: list[Candidate] = []
    for sign in (-1, 1):
        params = deepcopy(candidate.params)

        params["atr_sl_multiplier"] = _clamp(
            _scaled(float(params["atr_sl_multiplier"]), sign),
            float(search_space["atr_sl_min"]),
            float(search_space["atr_sl_max"]),
        )
        params["atr_tp_multiplier"] = _clamp(
            _scaled(float(params["atr_tp_multiplier"]), sign),
            float(search_space["atr_tp_min"]),
            float(search_space["atr_tp_max"]),
        )
        params["trailing_atr_k"] = _clamp(
            _scaled(float(params["trailing_atr_k"]), sign),
            float(search_space["trailing_atr_k_min"]),
            float(search_space["trailing_atr_k_max"]),
        )

        bars_scaled = int(round(_scaled(int(params["max_holding_bars"]), sign)))
        params["max_holding_bars"] = _nearest_int_option(bars_scaled, search_space["max_holding_bars"])

        if candidate.family in {"DonchianBreakout", "RangeBreakoutTrendFilter"}:
            ch_scaled = int(round(_scaled(int(params["channel_period"]), sign)))
            params["channel_period"] = _nearest_int_option(ch_scaled, search_space["donchian_periods"])

        if candidate.family in {"RSIMeanReversion", "TrendPullback", "BollingerMeanReversion"}:
            params["rsi_long_entry"] = int(
                _clamp(
                    int(params["rsi_long_entry"]) + (2 * sign),
                    int(search_space["rsi_long_entry_min"]),
                    int(search_space["rsi_long_entry_max"]),
                )
            )
            params["rsi_short_entry"] = int(
                _clamp(
                    int(params["rsi_short_entry"]) - (2 * sign),
                    int(search_space["rsi_short_entry_min"]),
                    int(search_space["rsi_short_entry_max"]),
                )
            )

        if candidate.family == "BollingerMeanReversion":
            bb_scaled = _scaled(float(params["bollinger_std"]), sign)
            params["bollinger_std"] = _nearest_float_option(bb_scaled, search_space["bollinger_stds"])

        variant = Candidate(
            candidate_id=f"{candidate.candidate_id}_p{sign:+d}",
            family=candidate.family,
            gating_mode=candidate.gating_mode,
            exit_mode=candidate.exit_mode,
            params=params,
        )
        variants.append(variant)

    return variants


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _nearest_int_option(value: int, options: list[int]) -> int:
    choices = [int(x) for x in options]
    return min(choices, key=lambda x: abs(x - int(value)))


def _nearest_float_option(value: float, options: list[float]) -> float:
    choices = [float(x) for x in options]
    return min(choices, key=lambda x: abs(x - float(value)))
