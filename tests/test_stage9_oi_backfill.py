"""Stage-9.2.1 OI backfill pagination and coverage tests."""

from __future__ import annotations

from typing import Any

import pandas as pd

from buffmini.data.futures_extras import (
    fetch_open_interest_history_backfill,
    open_interest_coverage_report,
)
from buffmini.utils.hashing import stable_hash


class _DummyExchange:
    def parse_timeframe(self, timeframe: str) -> int:
        if timeframe != "1h":
            raise ValueError("unexpected timeframe")
        return 3600


def _chunk_fetcher(
    *,
    exchange: Any,
    perp_symbol: str,
    timeframe: str,
    start_ms: int,
    end_ms: int,
    limit: int,
    max_retries: int,
    retry_backoff_sec: float,
) -> list[dict[str, Any]]:
    del exchange, perp_symbol, timeframe, max_retries, retry_backoff_sec
    points = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    in_window = [point for point in points if start_ms <= point * 3600000 <= end_ms]
    in_window = in_window[-limit:]
    if not in_window:
        return []
    # include deterministic duplicate at boundary for dedup test.
    raw = [{"timestamp": point * 3600000, "openInterest": float(1000 + point)} for point in in_window]
    raw.append({"timestamp": in_window[-1] * 3600000, "openInterest": float(2000 + in_window[-1])})
    return raw


def test_oi_backfill_pagination_and_dedup_deterministic() -> None:
    exchange = _DummyExchange()
    frame_a, info_a = fetch_open_interest_history_backfill(
        exchange=exchange,
        symbol="BTC/USDT",
        start_ms=0,
        end_ms=10 * 3600000,
        timeframe="1h",
        limit=3,
        chunk_fetcher=_chunk_fetcher,
    )
    frame_b, info_b = fetch_open_interest_history_backfill(
        exchange=exchange,
        symbol="BTC/USDT",
        start_ms=0,
        end_ms=10 * 3600000,
        timeframe="1h",
        limit=3,
        chunk_fetcher=_chunk_fetcher,
    )

    assert frame_a.equals(frame_b)
    assert info_a == info_b
    assert frame_a["ts"].is_monotonic_increasing
    assert frame_a["ts"].duplicated().sum() == 0
    assert len(frame_a) == 10
    assert frame_a["ts"].iloc[0] == pd.Timestamp("1970-01-01T01:00:00Z")
    assert stable_hash(frame_a.to_dict(orient="records"), length=16) == stable_hash(
        frame_b.to_dict(orient="records"),
        length=16,
    )


def test_oi_coverage_gap_detection() -> None:
    frame = pd.DataFrame(
        {
            "ts": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T01:00:00Z",
                    "2026-01-01T04:00:00Z",
                    "2026-01-01T05:00:00Z",
                ],
                utc=True,
            ),
            "open_interest": [1.0, 1.1, 1.2, 1.3],
        }
    )
    coverage = open_interest_coverage_report(
        open_interest=frame,
        expected_start_ts=pd.Timestamp("2026-01-01T00:00:00Z"),
        expected_end_ts=pd.Timestamp("2026-01-01T05:00:00Z"),
        timeframe="1h",
    )
    assert coverage["row_count"] == 4
    assert coverage["total_expected_rows"] == 6
    assert coverage["gap_count"] == 1
    assert coverage["largest_gap_hours"] == 3.0
    assert 0.0 <= coverage["coverage_ratio"] <= 1.0
