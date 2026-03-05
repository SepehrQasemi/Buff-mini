"""Shared normalization and quality helpers for CoinAPI endpoint adapters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


def as_utc_timestamp(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts)


def normalize_series_frame(
    rows: Iterable[dict[str, Any]],
    *,
    symbol: str,
    endpoint: str,
    value_columns: list[str],
    source: str = "coinapi",
    ingest_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    ingest = pd.Timestamp.utcnow() if ingest_ts is None else pd.Timestamp(ingest_ts)
    ingest = pd.to_datetime(ingest, utc=True)
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        columns = ["ts", "symbol", *value_columns, "source", "ingest_ts"]
        return pd.DataFrame(columns=columns)

    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["ts"]).copy()
    for col in value_columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=value_columns).copy()
    frame["symbol"] = str(symbol)
    frame["source"] = str(source)
    frame["ingest_ts"] = ingest
    frame = frame.sort_values("ts").drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    return frame[["ts", "symbol", *value_columns, "source", "ingest_ts"]]


def quality_summary(
    frame: pd.DataFrame,
    *,
    symbol: str,
    endpoint: str,
    gap_threshold: pd.Timedelta,
    expected_start: pd.Timestamp | None = None,
    expected_end: pd.Timestamp | None = None,
) -> dict[str, Any]:
    work = frame.copy()
    if work.empty:
        return {
            "symbol": str(symbol),
            "endpoint": str(endpoint),
            "sample_count": 0,
            "start_ts": None,
            "end_ts": None,
            "duplicate_count": 0,
            "missing_ratio": 1.0,
            "gaps_count": 0,
            "max_gap_minutes": 0.0,
        }
    work["ts"] = pd.to_datetime(work["ts"], utc=True, errors="coerce")
    work = work.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
    duplicate_count = int(work["ts"].duplicated(keep=False).sum())
    work = work.drop_duplicates(subset=["ts"], keep="last").reset_index(drop=True)
    diffs = work["ts"].diff().dropna()
    gap_mask = diffs > gap_threshold
    gaps = diffs[gap_mask]
    max_gap_minutes = float((gaps.max() / pd.Timedelta(minutes=1)) if not gaps.empty else 0.0)

    observed_start = pd.Timestamp(work["ts"].iloc[0])
    observed_end = pd.Timestamp(work["ts"].iloc[-1])
    expected_lo = observed_start if expected_start is None else pd.Timestamp(expected_start).tz_convert("UTC")
    expected_hi = observed_end if expected_end is None else pd.Timestamp(expected_end).tz_convert("UTC")
    if expected_hi < expected_lo:
        expected_hi = expected_lo
    expected_span = max(1, int(((expected_hi - expected_lo) / gap_threshold) + 1))
    sample_count = int(work.shape[0])
    missing_ratio = float(max(0.0, 1.0 - (sample_count / float(expected_span))))

    return {
        "symbol": str(symbol),
        "endpoint": str(endpoint),
        "sample_count": sample_count,
        "start_ts": observed_start.isoformat(),
        "end_ts": observed_end.isoformat(),
        "duplicate_count": int(duplicate_count),
        "missing_ratio": float(missing_ratio),
        "gaps_count": int(gaps.shape[0]),
        "max_gap_minutes": float(max_gap_minutes),
    }


def write_canonical(
    frame: pd.DataFrame,
    *,
    symbol: str,
    endpoint: str,
    data_root: Path = Path("data") / "coinapi" / "canonical",
) -> Path:
    safe_symbol = str(symbol).replace("/", "_").replace(":", "_")
    out_dir = Path(data_root) / safe_symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{endpoint}.parquet"
    frame.to_parquet(out_path, index=False, compression="zstd")
    return out_path


def write_coverage_summary(
    payload: dict[str, Any],
    *,
    symbol: str,
    endpoint: str,
    data_root: Path = Path("data") / "coinapi" / "meta",
) -> Path:
    safe_symbol = str(symbol).replace("/", "_").replace(":", "_")
    out_dir = Path(data_root) / "coverage" / safe_symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{endpoint}_coverage.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    return out_path

