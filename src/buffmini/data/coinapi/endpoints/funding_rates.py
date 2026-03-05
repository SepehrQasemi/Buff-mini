"""Funding rates endpoint adapter for CoinAPI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ._common import as_utc_timestamp, normalize_series_frame, quality_summary, write_canonical, write_coverage_summary

FUNDING_RATES_ENDPOINT_NAME = "funding_rates"
FUNDING_RATES_PATH = "/v1/exchangerate/futures/funding_rate/history"


def normalize_funding_rates(
    payload: list[dict[str, Any]],
    *,
    symbol: str,
    source: str = "coinapi",
    ingest_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for item in payload:
        ts = (
            as_utc_timestamp(item.get("time_period_start"))
            or as_utc_timestamp(item.get("time_exchange"))
            or as_utc_timestamp(item.get("time_close"))
            or as_utc_timestamp(item.get("time_open"))
            or as_utc_timestamp(item.get("time"))
        )
        if ts is None:
            continue
        value = item.get("rate")
        if value is None:
            value = item.get("funding_rate")
        if value is None:
            value = item.get("fundingRate")
        if value is None:
            continue
        rows.append({"ts": ts, "funding_rate": value})
    return normalize_series_frame(
        rows,
        symbol=symbol,
        endpoint=FUNDING_RATES_ENDPOINT_NAME,
        value_columns=["funding_rate"],
        source=source,
        ingest_ts=ingest_ts,
    )


def funding_coverage_summary(
    frame: pd.DataFrame,
    *,
    symbol: str,
    expected_start: pd.Timestamp | None = None,
    expected_end: pd.Timestamp | None = None,
) -> dict[str, Any]:
    return quality_summary(
        frame,
        symbol=symbol,
        endpoint=FUNDING_RATES_ENDPOINT_NAME,
        gap_threshold=pd.Timedelta(hours=6),
        expected_start=expected_start,
        expected_end=expected_end,
    )


def write_funding_canonical(
    frame: pd.DataFrame,
    *,
    symbol: str,
    expected_start: pd.Timestamp | None = None,
    expected_end: pd.Timestamp | None = None,
    canonical_root: Path = Path("data") / "coinapi" / "canonical",
    meta_root: Path = Path("data") / "coinapi" / "meta",
) -> tuple[Path, Path]:
    parquet_path = write_canonical(frame, symbol=symbol, endpoint=FUNDING_RATES_ENDPOINT_NAME, data_root=canonical_root)
    coverage = funding_coverage_summary(frame, symbol=symbol, expected_start=expected_start, expected_end=expected_end)
    coverage_path = write_coverage_summary(coverage, symbol=symbol, endpoint=FUNDING_RATES_ENDPOINT_NAME, data_root=meta_root)
    return parquet_path, coverage_path
