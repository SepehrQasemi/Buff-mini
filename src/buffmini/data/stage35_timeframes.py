"""Stage-35 timeframe rebuild helpers with 1m-first policy and integrity checks."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from buffmini.data.loader import standardize_ohlcv_frame, validate_ohlcv_frame
from buffmini.data.resample import resample_monthly_ohlcv, resample_ohlcv


def _tf_minutes(tf: str) -> int:
    text = str(tf).strip()
    if text == "1M":
        return 43_200
    match = re.fullmatch(r"(\d+)([mhdw])", text, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unsupported timeframe: {tf}")
    value = int(match.group(1))
    unit = match.group(2).lower()
    mult = {"m": 1, "h": 60, "d": 1440, "w": 10080}[unit]
    return int(value * mult)


def _is_divisible(base_tf: str, target_tf: str) -> bool:
    if str(target_tf) == "1M":
        return str(base_tf) in {"1d", "1h", "1m"}
    base = _tf_minutes(base_tf)
    target = _tf_minutes(target_tf)
    return target >= base and target % base == 0


def choose_resample_base(available_timeframes: list[str], target_timeframe: str) -> str:
    """Choose 1m as truth source when available and divisible; fallback to smallest divisor."""

    available = [str(v) for v in available_timeframes]
    target = str(target_timeframe)
    if target in available:
        return target
    if "1m" in available and _is_divisible("1m", target):
        return "1m"
    divisors = [tf for tf in available if _is_divisible(tf, target)]
    if not divisors:
        raise ValueError(f"No divisible base for {target} from available {sorted(available)}")
    return sorted(divisors, key=_tf_minutes)[0]


def build_timeframe_from_base(
    base_frame: pd.DataFrame,
    *,
    base_timeframe: str,
    target_timeframe: str,
    drop_incomplete_last: bool = True,
) -> pd.DataFrame:
    base = standardize_ohlcv_frame(base_frame)
    validate_ohlcv_frame(base)
    if str(target_timeframe) == str(base_timeframe):
        return base.copy()
    if str(target_timeframe) == "1M":
        out = resample_monthly_ohlcv(base, partial_last_bucket=not bool(drop_incomplete_last))
    else:
        out = resample_ohlcv(
            base,
            target_timeframe=str(target_timeframe),
            base_timeframe=str(base_timeframe),
            partial_last_bucket=not bool(drop_incomplete_last),
        )
    validate_ohlcv_frame(out)
    return out


def integrity_checks(
    base_frame: pd.DataFrame,
    rebuilt_frame: pd.DataFrame,
    *,
    base_timeframe: str,
    target_timeframe: str,
    volume_tol: float = 1e-9,
) -> dict[str, Any]:
    base = standardize_ohlcv_frame(base_frame)
    out = standardize_ohlcv_frame(rebuilt_frame)
    validate_ohlcv_frame(base)
    validate_ohlcv_frame(out)
    if out.empty:
        return {
            "target_timeframe": str(target_timeframe),
            "rows": 0,
            "high_low_consistent": True,
            "volume_consistent": True,
            "max_volume_abs_diff": 0.0,
        }

    high_low_ok = bool(((out["high"] >= out[["open", "close"]].max(axis=1)) & (out["low"] <= out[["open", "close"]].min(axis=1))).all())

    if str(target_timeframe) == "1M":
        expected = resample_monthly_ohlcv(base, partial_last_bucket=False)
    else:
        expected = resample_ohlcv(base, target_timeframe=target_timeframe, base_timeframe=base_timeframe, partial_last_bucket=False)
    merged = out.merge(expected[["timestamp", "volume"]], on="timestamp", how="left", suffixes=("_out", "_exp"))
    merged["volume_exp"] = pd.to_numeric(merged["volume_exp"], errors="coerce").fillna(0.0)
    merged["volume_out"] = pd.to_numeric(merged["volume_out"], errors="coerce").fillna(0.0)
    diffs = (merged["volume_out"] - merged["volume_exp"]).abs()
    max_diff = float(diffs.max()) if not diffs.empty else 0.0
    volume_ok = bool(max_diff <= float(volume_tol))

    return {
        "target_timeframe": str(target_timeframe),
        "rows": int(out.shape[0]),
        "high_low_consistent": bool(high_low_ok),
        "volume_consistent": bool(volume_ok),
        "max_volume_abs_diff": float(max_diff),
    }

