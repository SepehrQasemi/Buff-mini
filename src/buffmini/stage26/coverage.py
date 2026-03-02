"""Stage-26 data coverage utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.constants import RAW_DATA_DIR


@dataclass(frozen=True)
class CoverageResult:
    symbol: str
    timeframe: str
    path: str
    exists: bool
    start_ts: str | None
    end_ts: str | None
    coverage_days: float
    coverage_years: float
    expected_bars: int
    observed_bars: int
    duplicate_timestamps: int
    non_monotonic: bool
    missing_bars_estimate: int
    gap_days_estimate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "path": self.path,
            "exists": bool(self.exists),
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "coverage_days": float(self.coverage_days),
            "coverage_years": float(self.coverage_years),
            "expected_bars": int(self.expected_bars),
            "observed_bars": int(self.observed_bars),
            "duplicate_timestamps": int(self.duplicate_timestamps),
            "non_monotonic": bool(self.non_monotonic),
            "missing_bars_estimate": int(self.missing_bars_estimate),
            "gap_days_estimate": float(self.gap_days_estimate),
        }


def symbol_timeframe_path(symbol: str, timeframe: str, *, data_dir: Path = RAW_DATA_DIR) -> Path:
    stem = str(symbol).replace("/", "-").replace(":", "-")
    return Path(data_dir) / f"{stem}_{timeframe}.parquet"


def timeframe_seconds(timeframe: str) -> int:
    tf = str(timeframe).strip().lower()
    mapping = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "2h": 7200,
        "4h": 14400,
        "1d": 86400,
    }
    if tf not in mapping:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return int(mapping[tf])


def audit_symbol_coverage(
    *,
    symbol: str,
    timeframe: str = "1m",
    data_dir: Path = RAW_DATA_DIR,
    end_mode: str = "latest",
) -> CoverageResult:
    """Audit deterministic coverage stats from cached parquet."""

    _ = str(end_mode).strip().lower()  # reserved for future explicit end anchoring.
    path = symbol_timeframe_path(symbol, timeframe, data_dir=data_dir)
    if not path.exists():
        return CoverageResult(
            symbol=str(symbol),
            timeframe=str(timeframe),
            path=path.as_posix(),
            exists=False,
            start_ts=None,
            end_ts=None,
            coverage_days=0.0,
            coverage_years=0.0,
            expected_bars=0,
            observed_bars=0,
            duplicate_timestamps=0,
            non_monotonic=False,
            missing_bars_estimate=0,
            gap_days_estimate=0.0,
        )

    frame = pd.read_parquet(path)
    if frame.empty:
        return CoverageResult(
            symbol=str(symbol),
            timeframe=str(timeframe),
            path=path.as_posix(),
            exists=True,
            start_ts=None,
            end_ts=None,
            coverage_days=0.0,
            coverage_years=0.0,
            expected_bars=0,
            observed_bars=0,
            duplicate_timestamps=0,
            non_monotonic=False,
            missing_bars_estimate=0,
            gap_days_estimate=0.0,
        )

    ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce").dropna()
    if ts.empty:
        return CoverageResult(
            symbol=str(symbol),
            timeframe=str(timeframe),
            path=path.as_posix(),
            exists=True,
            start_ts=None,
            end_ts=None,
            coverage_days=0.0,
            coverage_years=0.0,
            expected_bars=0,
            observed_bars=0,
            duplicate_timestamps=0,
            non_monotonic=False,
            missing_bars_estimate=0,
            gap_days_estimate=0.0,
        )

    ts_sorted = ts.sort_values().reset_index(drop=True)
    step = int(timeframe_seconds(timeframe))
    start = pd.Timestamp(ts_sorted.iloc[0])
    end = pd.Timestamp(ts_sorted.iloc[-1])
    span_seconds = max(0.0, float((end - start).total_seconds()))
    coverage_days = float(span_seconds / 86400.0)
    coverage_years = float(coverage_days / 365.25)
    expected_bars = int(np.floor(span_seconds / max(1, step)) + 1)
    observed_bars = int(ts_sorted.shape[0])
    duplicate_timestamps = int(ts_sorted.duplicated().sum())
    non_monotonic = bool((ts.diff().dropna() < pd.Timedelta(0)).any())
    missing_est = int(max(0, expected_bars - observed_bars))
    gap_days_est = float((missing_est * step) / 86400.0)

    return CoverageResult(
        symbol=str(symbol),
        timeframe=str(timeframe),
        path=path.as_posix(),
        exists=True,
        start_ts=start.isoformat(),
        end_ts=end.isoformat(),
        coverage_days=float(coverage_days),
        coverage_years=float(coverage_years),
        expected_bars=int(expected_bars),
        observed_bars=int(observed_bars),
        duplicate_timestamps=int(duplicate_timestamps),
        non_monotonic=bool(non_monotonic),
        missing_bars_estimate=int(missing_est),
        gap_days_estimate=float(gap_days_est),
    )

