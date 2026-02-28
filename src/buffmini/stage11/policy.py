"""Stage-11 default hook policy implementations (config-driven, bias-first)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from buffmini.stage10.signals import signal_family_type
from buffmini.stage11.hooks import build_noop_hooks


DEFAULT_POLICY_CFG: dict[str, Any] = {
    "bias": {
        "enabled": True,
        "multiplier_min": 0.9,
        "multiplier_max": 1.1,
        "trend_boost": 1.10,
        "range_boost": 1.05,
        "vol_cut": 0.95,
        "trend_slope_scale": 0.01,
    },
    "confirm": {
        "enabled": False,
        "threshold": 0.55,
        "max_delay_bars": 3,
    },
    "exit": {
        "enabled": False,
        "tighten_trailing_scale": 0.9,
    },
}


def build_stage11_policy_hooks(policy_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build callable hook set from policy config."""

    cfg = _merge(DEFAULT_POLICY_CFG, policy_cfg or {})
    hooks = build_noop_hooks()

    if bool(cfg["bias"]["enabled"]):
        hooks["bias"] = _build_bias_hook(cfg["bias"])
    if bool(cfg["confirm"]["enabled"]):
        hooks["confirm"] = _build_confirm_hook(cfg["confirm"])
    if bool(cfg["exit"]["enabled"]):
        hooks["exit"] = _build_exit_hook(cfg["exit"])
    return hooks


def _build_bias_hook(cfg: dict[str, Any]):
    m_min = float(cfg.get("multiplier_min", 0.9))
    m_max = float(cfg.get("multiplier_max", 1.1))
    trend_boost = float(cfg.get("trend_boost", 1.10))
    range_boost = float(cfg.get("range_boost", 1.05))
    vol_cut = float(cfg.get("vol_cut", 0.95))
    trend_slope_scale = max(1e-6, float(cfg.get("trend_slope_scale", 0.01)))

    class _BiasHook:
        def __init__(self) -> None:
            self._multipliers: list[float] = []

        def __call__(
            self,
            *,
            timestamp: pd.Timestamp,
            symbol: str,
            signal_family: str,
            signal: int,
            base_row: dict[str, Any],
            activation_multiplier: float,
        ) -> float:
            _ = (timestamp, symbol, signal, activation_multiplier)
            family_type = signal_family_type(signal_family)
            trend_score = _score_trend(base_row, trend_slope_scale)
            range_score = _score_range(base_row, trend_score)
            vol_score = _score_vol_expansion(base_row)

            raw = 1.0
            if family_type == "trend":
                raw *= 1.0 + (trend_boost - 1.0) * trend_score
            elif family_type == "mean_reversion":
                raw *= 1.0 + (range_boost - 1.0) * range_score
            raw *= 1.0 - (1.0 - vol_cut) * vol_score

            confidence = max(trend_score, range_score, vol_score)
            smooth = 1.0 + (raw - 1.0) * confidence
            value = float(np.clip(smooth, m_min, m_max))
            self._multipliers.append(value)
            return value

        def get_stats(self) -> dict[str, Any]:
            if not self._multipliers:
                return {
                    "values": [],
                    "mean_multiplier": 1.0,
                    "p05": 1.0,
                    "p50": 1.0,
                    "p95": 1.0,
                    "pct_not_1_0": 0.0,
                }
            arr = np.asarray(self._multipliers, dtype=float)
            return {
                "values": [float(item) for item in arr.tolist()],
                "mean_multiplier": float(arr.mean()),
                "p05": float(np.quantile(arr, 0.05)),
                "p50": float(np.quantile(arr, 0.50)),
                "p95": float(np.quantile(arr, 0.95)),
                "pct_not_1_0": float(np.mean(np.abs(arr - 1.0) > 1e-12)),
            }

    return _BiasHook()


def _build_confirm_hook(cfg: dict[str, Any]):
    threshold = float(cfg.get("threshold", 0.55))
    max_delay_bars = max(0, int(cfg.get("max_delay_bars", 3)))

    class _ConfirmDelayHook:
        def __init__(self) -> None:
            self._pending: list[dict[str, int]] = []
            self._signals_seen = 0
            self._confirmed = 0
            self._skipped = 0
            self._delays: list[int] = []
            self._entry_deltas: list[int] = []

        def start_sequence(self) -> None:
            self._pending = []

        def __call__(
            self,
            *,
            timestamp: pd.Timestamp,
            symbol: str,
            signal_family: str,
            signal: int,
            base_row: dict[str, Any],
        ) -> int:
            _ = (timestamp, symbol, signal_family)
            emitted = 0
            score = _score_confirm(base_row)

            if int(signal) != 0:
                self._signals_seen += 1
                self._pending.append({"signal": int(np.sign(signal)), "delay": 0, "remaining": max_delay_bars})

            next_pending: list[dict[str, int]] = []
            emitted_once = False
            for item in self._pending:
                slope = _to_float(base_row.get("stage11_confirm_ema_slope_50", base_row.get("ema_slope_50", 0.0)))
                direction_ok = (int(item["signal"]) > 0 and slope >= 0.0) or (int(item["signal"]) < 0 and slope <= 0.0)
                if not emitted_once and score >= threshold and direction_ok:
                    emitted = int(item["signal"])
                    emitted_once = True
                    self._confirmed += 1
                    self._delays.append(int(item["delay"]))
                    self._entry_deltas.append(int(item["delay"]))
                    continue
                item["delay"] = int(item["delay"]) + 1
                item["remaining"] = int(item["remaining"]) - 1
                if int(item["remaining"]) < 0:
                    self._skipped += 1
                else:
                    next_pending.append(item)
            self._pending = next_pending
            return int(emitted)

        def finalize_sequence(self) -> None:
            if self._pending:
                self._skipped += len(self._pending)
            self._pending = []

        def get_stats(self) -> dict[str, Any]:
            seen = int(self._signals_seen)
            confirmed = int(self._confirmed)
            skipped = int(self._skipped)
            delays = np.asarray(self._delays, dtype=float) if self._delays else np.asarray([], dtype=float)
            confirm_rate = float(confirmed / seen) if seen > 0 else 0.0
            return {
                "signals_seen": seen,
                "confirmed": confirmed,
                "skipped": skipped,
                "confirm_rate": confirm_rate,
                "median_delay_bars": float(np.median(delays)) if delays.size else 0.0,
                "entry_delay_bars": [int(item) for item in self._entry_deltas],
            }

    return _ConfirmDelayHook()


def _build_exit_hook(cfg: dict[str, Any]):
    tighten = float(cfg.get("tighten_trailing_scale", 0.9))

    def _hook(
        *,
        timestamp: pd.Timestamp,
        symbol: str,
        signal_family: str,
        exit_mode: str,
        trailing_atr_k: float,
        partial_size: float,
        base_row: dict[str, Any],
    ) -> dict[str, Any]:
        _ = (timestamp, symbol, signal_family)
        adverse = _score_vol_expansion(base_row) > 0.8
        if adverse and str(exit_mode) == "atr_trailing":
            return {
                "exit_mode": str(exit_mode),
                "trailing_atr_k": float(max(0.1, trailing_atr_k * tighten)),
                "partial_size": float(partial_size),
            }
        return {
            "exit_mode": str(exit_mode),
            "trailing_atr_k": float(trailing_atr_k),
            "partial_size": float(partial_size),
        }

    return _hook


def _score_trend(row: dict[str, Any], scale: float) -> float:
    slope = _to_float(
        row.get(
            "stage11_context_ema_slope_50",
            row.get("ema_slope_50", row.get("trend_strength_stage10", 0.0)),
        )
    )
    return _clip01(_sigmoid(abs(slope) / scale))


def _score_vol_expansion(row: dict[str, Any]) -> float:
    atr_rank = _to_float(row.get("stage11_context_atr_pct_rank_252", row.get("atr_pct_rank_252", 0.5)))
    return _clip01(atr_rank)


def _score_range(row: dict[str, Any], trend_score: float) -> float:
    atr_rank = _to_float(row.get("stage11_context_atr_pct_rank_252", row.get("atr_pct_rank_252", 0.5)))
    mid_vol = 1.0 - abs(atr_rank - 0.5) / 0.25
    return _clip01((1.0 - trend_score) * max(0.0, mid_vol))


def _score_confirm(row: dict[str, Any]) -> float:
    volume_z = _to_float(row.get("stage11_confirm_volume_z_120", row.get("volume_z_120", 0.0)))
    slope = abs(_to_float(row.get("stage11_confirm_ema_slope_50", row.get("ema_slope_50", 0.0))))
    volume_score = _clip01((volume_z + 1.0) / 2.0)
    slope_score = _clip01(slope / 0.02)
    return _clip01(0.5 * volume_score + 0.5 * slope_score)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(dict(out[key]), value)
        else:
            out[key] = value
    return out


def _to_float(value: Any) -> float:
    try:
        numeric = float(value)
    except Exception:
        numeric = 0.0
    if not np.isfinite(numeric):
        return 0.0
    return float(numeric)


def _sigmoid(value: float) -> float:
    clipped = float(np.clip(value, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-clipped)))


def _clip01(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))
