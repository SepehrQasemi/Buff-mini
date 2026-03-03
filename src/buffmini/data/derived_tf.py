"""Stage-26.9.3 divisible-base derived timeframe engine."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from buffmini.constants import CANONICAL_DATA_DIR, DERIVED_DATA_DIR
from buffmini.data.loader import standardize_ohlcv_frame, validate_ohlcv_frame
from buffmini.data.resample import resample_monthly_ohlcv, resample_ohlcv
from buffmini.utils.hashing import stable_hash


CANONICAL_TIMEFRAMES: tuple[str, ...] = (
    "1m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "12h",
    "1d",
    "1w",
    "1M",
)


@dataclass(frozen=True)
class TimeframeLoadResult:
    frame: pd.DataFrame
    source_timeframe: str
    target_timeframe: str
    cache_path: Path
    meta_path: Path
    cache_hit: bool
    data_hash: str


def canonical_tf_path(
    *,
    symbol: str,
    timeframe: str,
    exchange: str = "binance",
    canonical_dir: Path = CANONICAL_DATA_DIR,
) -> Path:
    safe_symbol = str(symbol).replace("/", "-").replace(":", "-")
    return Path(canonical_dir) / str(exchange).strip().lower() / safe_symbol / f"{timeframe}.parquet"


def canonical_meta_path(
    *,
    symbol: str,
    timeframe: str,
    exchange: str = "binance",
    canonical_dir: Path = CANONICAL_DATA_DIR,
) -> Path:
    return canonical_tf_path(symbol=symbol, timeframe=timeframe, exchange=exchange, canonical_dir=canonical_dir).with_suffix(".meta.json")


def derived_tf_path(
    *,
    symbol: str,
    timeframe: str,
    exchange: str = "binance",
    derived_dir: Path = DERIVED_DATA_DIR,
) -> Path:
    safe_symbol = str(symbol).replace("/", "-").replace(":", "-")
    return Path(derived_dir) / str(exchange).strip().lower() / safe_symbol / f"{timeframe}.parquet"


def derived_meta_path(
    *,
    symbol: str,
    timeframe: str,
    exchange: str = "binance",
    derived_dir: Path = DERIVED_DATA_DIR,
) -> Path:
    return derived_tf_path(symbol=symbol, timeframe=timeframe, exchange=exchange, derived_dir=derived_dir).with_suffix(".meta.json")


def canonical_timeframes_available(
    *,
    symbol: str,
    exchange: str = "binance",
    canonical_dir: Path = CANONICAL_DATA_DIR,
) -> list[str]:
    symbol_dir = Path(canonical_dir) / str(exchange).strip().lower() / str(symbol).replace("/", "-").replace(":", "-")
    values: list[str] = []
    if symbol_dir.exists():
        for file in sorted(symbol_dir.glob("*.parquet")):
            values.append(file.stem)
    for tf in CANONICAL_TIMEFRAMES:
        if tf not in values:
            path = canonical_tf_path(symbol=symbol, timeframe=tf, exchange=exchange, canonical_dir=canonical_dir)
            if path.exists():
                values.append(tf)
    return sorted(values, key=_sort_key)


def resolve_divisible_base(target_tf: str, canonical_tfs_available: list[str]) -> str:
    """Resolve canonical source timeframe using strict divisibility (never nearest rounding)."""

    target = str(target_tf).strip()
    available = [str(tf).strip() for tf in canonical_tfs_available]
    if target in available:
        return target
    if target == "1M":
        raise ValueError("No divisible canonical base for monthly target without canonical 1M")

    target_minutes = _timeframe_minutes(target)
    candidates = [tf for tf in available if _is_divisor(tf, target_minutes)]
    if not candidates:
        raise ValueError(f"No divisible canonical base available for target {target}; available={sorted(available)}")

    # Deterministic resolver aligned with Stage-26.9 examples.
    if target_minutes < 60:
        for pref in ("5m", "15m", "1m"):
            if pref in candidates:
                if target == "45m" and "15m" in candidates:
                    return "15m"
                if target == "30m" and "5m" in candidates:
                    return "5m"
                if pref != "15m" or target_minutes % 15 == 0:
                    return pref
    elif target_minutes < 1440:
        for pref in ("1h", "4h", "2h", "30m", "15m", "5m", "1m"):
            if pref in candidates:
                return pref
    else:
        for pref in ("1d", "12h", "6h", "4h", "2h", "1h", "30m", "15m", "5m", "1m"):
            if pref in candidates:
                return pref

    # Fallback: highest divisible duration for runtime efficiency.
    return sorted(candidates, key=lambda item: _timeframe_minutes(item), reverse=True)[0]


def get_timeframe(
    *,
    symbol: str,
    timeframe: str,
    exchange: str = "binance",
    canonical_dir: Path = CANONICAL_DATA_DIR,
    derived_dir: Path = DERIVED_DATA_DIR,
    drop_incomplete_last: bool = True,
    generator_version: str = "stage26_9",
) -> TimeframeLoadResult:
    """Load canonical timeframe or build/cache derived timeframe from divisible canonical base."""

    tf = str(timeframe).strip()
    canonical_path = canonical_tf_path(symbol=symbol, timeframe=tf, exchange=exchange, canonical_dir=canonical_dir)
    if canonical_path.exists():
        frame = standardize_ohlcv_frame(pd.read_parquet(canonical_path))
        validate_ohlcv_frame(frame)
        return TimeframeLoadResult(
            frame=frame,
            source_timeframe=tf,
            target_timeframe=tf,
            cache_path=canonical_path,
            meta_path=canonical_meta_path(symbol=symbol, timeframe=tf, exchange=exchange, canonical_dir=canonical_dir),
            cache_hit=True,
            data_hash=_frame_hash(frame),
        )

    available = canonical_timeframes_available(symbol=symbol, exchange=exchange, canonical_dir=canonical_dir)
    base_tf = resolve_divisible_base(tf, available)
    source_path = canonical_tf_path(symbol=symbol, timeframe=base_tf, exchange=exchange, canonical_dir=canonical_dir)
    if not source_path.exists():
        raise FileNotFoundError(f"Canonical source missing: {source_path}")

    source_frame = standardize_ohlcv_frame(pd.read_parquet(source_path))
    validate_ohlcv_frame(source_frame)
    source_hash = _frame_hash(source_frame)

    cache_path = derived_tf_path(symbol=symbol, timeframe=tf, exchange=exchange, derived_dir=derived_dir)
    meta_path = derived_meta_path(symbol=symbol, timeframe=tf, exchange=exchange, derived_dir=derived_dir)
    cache_key = stable_hash(
        {
            "symbol": str(symbol),
            "exchange": str(exchange),
            "target_tf": tf,
            "source_tf": base_tf,
            "source_hash": source_hash,
            "drop_incomplete_last": bool(drop_incomplete_last),
            "generator_version": str(generator_version),
        },
        length=24,
    )

    if cache_path.exists() and meta_path.exists():
        try:
            meta = dict(json.loads(meta_path.read_text(encoding="utf-8")))
        except Exception:
            meta = {}
        if str(meta.get("cache_key", "")) == str(cache_key):
            frame = standardize_ohlcv_frame(pd.read_parquet(cache_path))
            validate_ohlcv_frame(frame)
            return TimeframeLoadResult(
                frame=frame,
                source_timeframe=base_tf,
                target_timeframe=tf,
                cache_path=cache_path,
                meta_path=meta_path,
                cache_hit=True,
                data_hash=_frame_hash(frame),
            )

    frame = _resample_with_support(source_frame, source_tf=base_tf, target_tf=tf, drop_incomplete_last=drop_incomplete_last)
    validate_ohlcv_frame(frame)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False, compression="zstd")
    payload = {
        "timeframe": tf,
        "source_timeframe": base_tf,
        "symbol": str(symbol),
        "exchange": str(exchange),
        "source_hash": source_hash,
        "target_hash": _frame_hash(frame),
        "cache_key": cache_key,
        "drop_incomplete_last": bool(drop_incomplete_last),
        "generator_version": str(generator_version),
        "rows": int(frame.shape[0]),
        "start_ts": _frame_start(frame),
        "end_ts": _frame_end(frame),
    }
    meta_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return TimeframeLoadResult(
        frame=frame,
        source_timeframe=base_tf,
        target_timeframe=tf,
        cache_path=cache_path,
        meta_path=meta_path,
        cache_hit=False,
        data_hash=_frame_hash(frame),
    )


def _resample_with_support(source_frame: pd.DataFrame, *, source_tf: str, target_tf: str, drop_incomplete_last: bool) -> pd.DataFrame:
    if target_tf == "1M":
        return resample_monthly_ohlcv(source_frame, partial_last_bucket=not bool(drop_incomplete_last))
    return resample_ohlcv(
        source_frame,
        target_timeframe=target_tf,
        base_timeframe=source_tf,
        partial_last_bucket=not bool(drop_incomplete_last),
    )


def _sort_key(tf: str) -> tuple[int, int]:
    if tf == "1M":
        return (2, 0)
    try:
        return (0, _timeframe_minutes(tf))
    except Exception:
        return (1, 0)


def _is_divisor(candidate_tf: str, target_minutes: int) -> bool:
    try:
        candidate_minutes = _timeframe_minutes(candidate_tf)
    except Exception:
        return False
    return candidate_minutes > 0 and target_minutes % candidate_minutes == 0


def _timeframe_minutes(tf: str) -> int:
    text = str(tf).strip()
    match = re.fullmatch(r"(\d+)([mhdw])", text, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Unsupported timeframe: {tf}")
    value = int(match.group(1))
    unit = match.group(2).lower()
    mult = {"m": 1, "h": 60, "d": 1440, "w": 10080}[unit]
    return int(value * mult)


def _frame_hash(frame: pd.DataFrame) -> str:
    cols = [c for c in ("timestamp", "open", "high", "low", "close", "volume") if c in frame.columns]
    if not cols:
        return stable_hash({"rows": int(frame.shape[0])}, length=24)
    return stable_hash(frame.loc[:, cols].to_dict(orient="list"), length=24)


def _frame_start(frame: pd.DataFrame) -> str | None:
    ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce").dropna()
    return ts.iloc[0].isoformat() if not ts.empty else None


def _frame_end(frame: pd.DataFrame) -> str | None:
    ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce").dropna()
    return ts.iloc[-1].isoformat() if not ts.empty else None
