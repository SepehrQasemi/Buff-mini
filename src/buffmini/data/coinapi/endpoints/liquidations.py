"""Liquidations endpoint adapter for CoinAPI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ._common import as_utc_timestamp, normalize_series_frame, quality_summary, write_canonical, write_coverage_summary

LIQUIDATIONS_ENDPOINT_NAME = "liquidations"
LIQUIDATIONS_PATH = "/v1/exchangerate/futures/liquidations/history"


def normalize_liquidations(
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
        side = str(item.get("side", "")).upper()
        qty = item.get("quantity") or item.get("qty") or item.get("amount") or 0.0
        value = item.get("value") or item.get("notional") or 0.0
        if side in {"BUY", "B"}:
            buy = qty
            sell = 0.0
        elif side in {"SELL", "S"}:
            buy = 0.0
            sell = qty
        else:
            buy = item.get("liq_buy", 0.0)
            sell = item.get("liq_sell", 0.0)
        rows.append(
            {
                "ts": ts,
                "liq_buy": buy,
                "liq_sell": sell,
                "liq_notional": value,
            }
        )
    return normalize_series_frame(
        rows,
        symbol=symbol,
        endpoint=LIQUIDATIONS_ENDPOINT_NAME,
        value_columns=["liq_buy", "liq_sell", "liq_notional"],
        source=source,
        ingest_ts=ingest_ts,
    )


def liquidations_coverage_summary(
    frame: pd.DataFrame,
    *,
    symbol: str,
    expected_start: pd.Timestamp | None = None,
    expected_end: pd.Timestamp | None = None,
) -> dict[str, Any]:
    return quality_summary(
        frame,
        symbol=symbol,
        endpoint=LIQUIDATIONS_ENDPOINT_NAME,
        gap_threshold=pd.Timedelta(hours=6),
        expected_start=expected_start,
        expected_end=expected_end,
    )


def write_liquidations_canonical(
    frame: pd.DataFrame,
    *,
    symbol: str,
    expected_start: pd.Timestamp | None = None,
    expected_end: pd.Timestamp | None = None,
    canonical_root: Path = Path("data") / "coinapi" / "canonical",
    meta_root: Path = Path("data") / "coinapi" / "meta",
) -> tuple[Path, Path]:
    parquet_path = write_canonical(frame, symbol=symbol, endpoint=LIQUIDATIONS_ENDPOINT_NAME, data_root=canonical_root)
    coverage = liquidations_coverage_summary(frame, symbol=symbol, expected_start=expected_start, expected_end=expected_end)
    coverage_path = write_coverage_summary(coverage, symbol=symbol, endpoint=LIQUIDATIONS_ENDPOINT_NAME, data_root=meta_root)
    return parquet_path, coverage_path

