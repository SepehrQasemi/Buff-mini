"""Funding and open-interest data loaders for futures extras."""

from __future__ import annotations

import time
from typing import Any

import ccxt
import numpy as np
import pandas as pd


def create_binance_futures_exchange() -> ccxt.Exchange:
    """Create Binance USD-M futures exchange client with rate limiting."""

    return ccxt.binanceusdm({"enableRateLimit": True})


def futures_symbol(symbol: str) -> str:
    """Normalize spot-like symbol to futures perp symbol."""

    raw = str(symbol)
    if ":" in raw:
        return raw
    if raw.endswith("/USDT"):
        return f"{raw}:USDT"
    return raw


def binance_oi_symbol(symbol: str) -> str:
    """Convert symbol to Binance OI endpoint symbol format, e.g. BTCUSDT."""

    raw = str(symbol)
    if ":" in raw:
        raw = raw.split(":", 1)[0]
    return raw.replace("/", "").replace("-", "")


def fetch_funding_history(
    exchange: ccxt.Exchange,
    symbol: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch funding history and return normalized rows with columns [ts, funding_rate]."""

    since = int(start_ms)
    end_ms = int(end_ms)
    rows: list[dict[str, Any]] = []
    perp = futures_symbol(symbol)

    interval_ms = 8 * 3600 * 1000  # funding cadence on Binance perp.
    chunk_span_ms = max(interval_ms, int(limit) * interval_ms)

    while since <= end_ms:
        chunk_end_ms = min(end_ms, since + chunk_span_ms - 1)
        batch = exchange.fetch_funding_rate_history(
            symbol=perp,
            since=since,
            limit=int(limit),
            params={"endTime": chunk_end_ms},
        )
        if not batch:
            break

        for item in batch:
            ts_ms = int(item.get("timestamp", 0) or 0)
            if ts_ms <= 0:
                continue
            if ts_ms < start_ms or ts_ms > end_ms:
                continue
            rate = item.get("fundingRate")
            if rate is None:
                info = item.get("info") if isinstance(item.get("info"), dict) else {}
                rate = info.get("fundingRate")
            rows.append(
                {
                    "ts": pd.to_datetime(ts_ms, unit="ms", utc=True),
                    "funding_rate": float(rate),
                }
            )

        last_ts = int(batch[-1].get("timestamp", 0) or 0)
        next_since = last_ts + 1
        if next_since <= since:
            break
        since = next_since
        if len(batch) < int(limit) and chunk_end_ms >= end_ms:
            break

    frame = pd.DataFrame(rows, columns=["ts", "funding_rate"]) if rows else pd.DataFrame(columns=["ts", "funding_rate"])
    return _clean_series_frame(frame=frame, value_col="funding_rate")


def align_funding_to_ohlcv(
    ohlcv: pd.DataFrame,
    funding: pd.DataFrame,
    timeframe: str = "1h",
) -> pd.DataFrame:
    """Align funding to candle-open timestamps using latest event ts <= candle close."""

    return _align_latest_event_to_candles(
        ohlcv=ohlcv,
        events=funding,
        value_col="funding_rate",
        timeframe=timeframe,
    )


def funding_quality_report(funding: pd.DataFrame) -> dict[str, Any]:
    """Return basic data-quality checks for funding event data."""

    return _series_quality_report(frame=funding, value_col="funding_rate", expected_gap_hours=8)


def fetch_open_interest_history(
    exchange: ccxt.Exchange,
    symbol: str,
    start_ms: int,
    end_ms: int,
    timeframe: str = "1h",
    limit: int = 500,
) -> pd.DataFrame:
    """Fetch open-interest history and return normalized rows with columns [ts, open_interest]."""

    frame, _ = fetch_open_interest_history_backfill(
        exchange=exchange,
        symbol=symbol,
        start_ms=start_ms,
        end_ms=end_ms,
        timeframe=timeframe,
        limit=limit,
    )
    return frame


def fetch_open_interest_history_backfill(
    exchange: ccxt.Exchange,
    symbol: str,
    start_ms: int,
    end_ms: int,
    timeframe: str = "1h",
    limit: int = 500,
    max_retries: int = 3,
    retry_backoff_sec: float = 0.8,
    sleep_between_chunks_sec: float = 0.0,
    chunk_fetcher: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Backfill open-interest history backwards in deterministic chunks.

    The loop requests fixed-size windows going backwards from ``end_ms`` until
    it either reaches ``start_ms`` or the exchange rejects older timestamps.
    """

    since = int(start_ms)
    end_ms = int(end_ms)
    rows: list[dict[str, Any]] = []
    perp = futures_symbol(symbol)

    interval_ms = int(exchange.parse_timeframe(str(timeframe))) * 1000
    chunk_span_ms = max(interval_ms, int(limit) * interval_ms)
    cursor_end_ms = int(end_ms)
    requests_count = 0
    stop_reason = "reached_start"
    warnings: list[str] = []

    fetcher = chunk_fetcher or _fetch_open_interest_chunk

    while cursor_end_ms >= since:
        chunk_start_ms = max(since, cursor_end_ms - chunk_span_ms + 1)
        try:
            batch = fetcher(
                exchange=exchange,
                perp_symbol=perp,
                timeframe=str(timeframe),
                start_ms=int(chunk_start_ms),
                end_ms=int(cursor_end_ms),
                limit=int(limit),
                max_retries=int(max_retries),
                retry_backoff_sec=float(retry_backoff_sec),
            )
        except ccxt.BadRequest:
            stop_reason = "api_rejected_older_timestamps"
            warnings.append("Binance OI API rejected older timestamps before requested start.")
            break
        requests_count += 1
        if not batch:
            stop_reason = "no_more_data"
            break

        rows.extend(_normalize_open_interest_batch(batch, start_ms=start_ms, end_ms=end_ms))
        batch_sorted = sorted(batch, key=lambda item: int(item.get("timestamp", 0) or 0))
        earliest_ts = int(batch_sorted[0].get("timestamp", 0) or 0)
        if earliest_ts <= since:
            stop_reason = "reached_start"
            break

        next_cursor_end = earliest_ts - 1
        if next_cursor_end >= cursor_end_ms:
            stop_reason = "stuck_cursor"
            break
        cursor_end_ms = next_cursor_end

        if len(batch) < int(limit):
            stop_reason = "insufficient_chunk_rows"
            break
        if float(sleep_between_chunks_sec) > 0:
            time.sleep(float(sleep_between_chunks_sec))

    frame = (
        pd.DataFrame(rows, columns=["ts", "open_interest"])
        if rows
        else pd.DataFrame(columns=["ts", "open_interest"])
    )
    cleaned = _clean_series_frame(frame=frame, value_col="open_interest")
    info = {
        "requests_count": int(requests_count),
        "stop_reason": str(stop_reason),
        "warnings": warnings,
    }
    return cleaned, info


def _normalize_open_interest_batch(batch: list[dict[str, Any]], start_ms: int, end_ms: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in batch:
        ts_ms = int(item.get("timestamp", 0) or 0)
        if ts_ms <= 0:
            continue
        if ts_ms < start_ms or ts_ms > end_ms:
            continue
        value = item.get("openInterestAmount")
        if value is None:
            value = item.get("openInterestValue")
        if value is None:
            value = item.get("openInterest")
        if value is None:
            value = item.get("sumOpenInterest")
        if value is None:
            info = item.get("info") if isinstance(item.get("info"), dict) else {}
            value = info.get("sumOpenInterest") or info.get("openInterest")
        if value is None:
            continue
        rows.append(
            {
                "ts": pd.to_datetime(ts_ms, unit="ms", utc=True),
                "open_interest": float(value),
            }
        )
    return rows


def _fetch_open_interest_chunk(
    exchange: ccxt.Exchange,
    perp_symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    limit: int,
    max_retries: int,
    retry_backoff_sec: float,
) -> list[dict[str, Any]]:
    params = {
        "symbol": binance_oi_symbol(perp_symbol),
        "period": str(timeframe),
        "limit": int(limit),
        "endTime": int(end_ms),
    }
    # Keep fixed-size chunk requests deterministic.
    if int(start_ms) > 0:
        params["startTime"] = int(start_ms)

    for attempt in range(int(max_retries)):
        try:
            return exchange.fapiDataGetOpenInterestHist(params)
        except ccxt.BadRequest:
            if "startTime" in params:
                # Retention-boundary fallback: query by endTime only.
                fallback_params = dict(params)
                fallback_params.pop("startTime", None)
                return exchange.fapiDataGetOpenInterestHist(fallback_params)
            raise
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout):
            if attempt + 1 >= int(max_retries):
                raise
            time.sleep(float(retry_backoff_sec) * float(2**attempt))
    return []


def align_open_interest_to_ohlcv(
    ohlcv: pd.DataFrame,
    open_interest: pd.DataFrame,
    timeframe: str = "1h",
) -> pd.DataFrame:
    """Align open interest to candle-open timestamps using latest event ts <= candle close."""

    return _align_latest_event_to_candles(
        ohlcv=ohlcv,
        events=open_interest,
        value_col="open_interest",
        timeframe=timeframe,
    )


def open_interest_quality_report(open_interest: pd.DataFrame) -> dict[str, Any]:
    """Return basic data-quality checks for open-interest event data."""

    return _series_quality_report(frame=open_interest, value_col="open_interest", expected_gap_hours=1)


def open_interest_coverage_report(
    open_interest: pd.DataFrame,
    expected_start_ts: pd.Timestamp,
    expected_end_ts: pd.Timestamp,
    timeframe: str = "1h",
) -> dict[str, Any]:
    """Compute coverage metrics against expected OHLCV window."""

    cleaned = _clean_series_frame(open_interest, value_col="open_interest")
    expected_start = pd.Timestamp(expected_start_ts)
    expected_end = pd.Timestamp(expected_end_ts)
    if expected_start.tzinfo is None:
        expected_start = expected_start.tz_localize("UTC")
    else:
        expected_start = expected_start.tz_convert("UTC")
    if expected_end.tzinfo is None:
        expected_end = expected_end.tz_localize("UTC")
    else:
        expected_end = expected_end.tz_convert("UTC")

    if expected_end < expected_start:
        expected_rows = 0
    else:
        expected_rows = int(((expected_end - expected_start).total_seconds() // 3600) + 1)

    ts = pd.to_datetime(cleaned["ts"], utc=True) if not cleaned.empty else pd.Series(dtype="datetime64[ns, UTC]")
    diffs = ts.diff().dropna() if not ts.empty else pd.Series(dtype="timedelta64[ns]")
    gap_mask = diffs > pd.Timedelta(hours=1.5)
    largest_gap_hours = float((diffs.max() / pd.Timedelta(hours=1))) if not diffs.empty else 0.0

    row_count = int(len(cleaned))
    coverage_ratio = float(row_count / expected_rows) if expected_rows > 0 else 0.0
    warnings: list[str] = []
    if coverage_ratio < 0.9:
        warnings.append("Coverage ratio below 0.90; Binance OI history retention likely limited.")

    return {
        "start_ts": ts.iloc[0].isoformat() if not ts.empty else None,
        "end_ts": ts.iloc[-1].isoformat() if not ts.empty else None,
        "row_count": row_count,
        "total_expected_rows": int(expected_rows),
        "coverage_ratio": float(coverage_ratio),
        "gap_count": int(gap_mask.sum()) if not diffs.empty else 0,
        "largest_gap_hours": float(largest_gap_hours),
        "warnings": warnings,
        "timeframe": str(timeframe),
    }


def _align_latest_event_to_candles(
    ohlcv: pd.DataFrame,
    events: pd.DataFrame,
    value_col: str,
    timeframe: str,
) -> pd.DataFrame:
    if "timestamp" not in ohlcv.columns:
        raise ValueError("ohlcv must contain timestamp")

    candles = ohlcv[["timestamp"]].copy()
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)
    delta = pd.Timedelta(hours=1) if str(timeframe) == "1h" else pd.Timedelta(0)
    candles["candle_close_ts"] = candles["timestamp"] + delta

    series = _clean_series_frame(events.copy(), value_col=value_col)
    if series.empty:
        aligned = candles[["timestamp"]].copy()
        aligned[value_col] = np.nan
        return aligned

    merged = pd.merge_asof(
        candles.sort_values("candle_close_ts"),
        series[["ts", value_col]].sort_values("ts"),
        left_on="candle_close_ts",
        right_on="ts",
        direction="backward",
        allow_exact_matches=True,
    )
    aligned = merged[["timestamp", value_col]].copy()
    return aligned.reset_index(drop=True)


def _clean_series_frame(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["ts", value_col])

    clean = frame.copy()
    clean["ts"] = pd.to_datetime(clean["ts"], utc=True, errors="coerce")
    clean[value_col] = pd.to_numeric(clean[value_col], errors="coerce")
    clean = clean.dropna(subset=["ts", value_col])
    clean = clean.drop_duplicates(subset=["ts"], keep="last").sort_values("ts").reset_index(drop=True)
    return clean[["ts", value_col]]


def _series_quality_report(frame: pd.DataFrame, value_col: str, expected_gap_hours: int) -> dict[str, Any]:
    cleaned = _clean_series_frame(frame, value_col=value_col)
    if cleaned.empty:
        return {
            "rows": 0,
            "start_ts": None,
            "end_ts": None,
            "monotonic_ts": True,
            "duplicates": 0,
            "finite": True,
            "gaps_count": 0,
            "expected_gap_hours": int(expected_gap_hours),
        }

    ts = pd.to_datetime(cleaned["ts"], utc=True)
    diffs = ts.diff().dropna()
    expected_gap = pd.Timedelta(hours=int(expected_gap_hours))
    gaps_count = int((diffs > (expected_gap * 1.5)).sum()) if not diffs.empty else 0

    raw = frame.copy()
    raw["ts"] = pd.to_datetime(raw["ts"], utc=True, errors="coerce")
    duplicates = int(raw["ts"].duplicated().sum()) if "ts" in raw.columns else 0

    return {
        "rows": int(len(cleaned)),
        "start_ts": ts.iloc[0].isoformat(),
        "end_ts": ts.iloc[-1].isoformat(),
        "monotonic_ts": bool(ts.is_monotonic_increasing),
        "duplicates": duplicates,
        "finite": bool(np.isfinite(pd.to_numeric(cleaned[value_col], errors="coerce")).all()),
        "gaps_count": gaps_count,
        "expected_gap_hours": int(expected_gap_hours),
    }
