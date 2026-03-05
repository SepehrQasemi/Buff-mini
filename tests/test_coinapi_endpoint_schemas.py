from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from buffmini.data.coinapi.endpoints.funding_rates import funding_coverage_summary, normalize_funding_rates
from buffmini.data.coinapi.endpoints.liquidations import liquidations_coverage_summary, normalize_liquidations
from buffmini.data.coinapi.endpoints.open_interest import normalize_open_interest, open_interest_coverage_summary


def _load_fixture(name: str) -> list[dict]:
    path = Path(__file__).resolve().parent / "data" / "coinapi" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_funding_schema_normalization_and_quality() -> None:
    payload = _load_fixture("funding_fixture.json")
    frame = normalize_funding_rates(payload, symbol="BTC/USDT")
    assert list(frame.columns) == ["ts", "symbol", "funding_rate", "source", "ingest_ts"]
    assert frame["ts"].is_monotonic_increasing
    assert int(frame["ts"].duplicated().sum()) == 0
    assert frame.shape[0] == 2
    cov = funding_coverage_summary(
        frame,
        symbol="BTC/USDT",
        expected_start=pd.Timestamp("2026-01-01T00:00:00Z"),
        expected_end=pd.Timestamp("2026-01-02T00:00:00Z"),
    )
    assert cov["sample_count"] == 2
    assert cov["gaps_count"] >= 1


def test_open_interest_schema_normalization_and_coverage() -> None:
    payload = _load_fixture("open_interest_fixture.json")
    frame = normalize_open_interest(payload, symbol="ETH/USDT")
    assert list(frame.columns) == ["ts", "symbol", "open_interest", "source", "ingest_ts"]
    assert frame["ts"].is_monotonic_increasing
    cov = open_interest_coverage_summary(
        frame,
        symbol="ETH/USDT",
        expected_start=pd.Timestamp("2026-01-01T00:00:00Z"),
        expected_end=pd.Timestamp("2026-01-05T00:00:00Z"),
    )
    assert cov["sample_count"] == 3
    assert cov["missing_ratio"] > 0
    assert cov["gaps_count"] == 1


def test_liquidations_schema_normalization_and_coverage() -> None:
    payload = _load_fixture("liquidations_fixture.json")
    frame = normalize_liquidations(payload, symbol="BTC/USDT")
    assert list(frame.columns) == ["ts", "symbol", "liq_buy", "liq_sell", "liq_notional", "source", "ingest_ts"]
    assert float(frame["liq_buy"].sum()) == 2.5
    assert float(frame["liq_sell"].sum()) == 3.0
    cov = liquidations_coverage_summary(
        frame,
        symbol="BTC/USDT",
        expected_start=pd.Timestamp("2026-01-01T00:00:00Z"),
        expected_end=pd.Timestamp("2026-01-01T06:00:00Z"),
    )
    assert cov["sample_count"] == 2
    assert cov["missing_ratio"] >= 0

