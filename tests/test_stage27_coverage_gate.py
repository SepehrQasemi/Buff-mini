from __future__ import annotations

from pathlib import Path

import pandas as pd

from buffmini.stage27.coverage_gate import evaluate_coverage_gate


def _write_symbol_1m(path: Path, start: str, end: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")],
            "open": [1.0, 1.0],
            "high": [1.0, 1.0],
            "low": [1.0, 1.0],
            "close": [1.0, 1.0],
            "volume": [1.0, 1.0],
        }
    )
    frame.to_parquet(path, index=False)


def _cfg(required: float = 4.0, minimum: float = 1.0, fail: bool = True) -> dict:
    return {
        "data": {
            "coverage": {
                "required_years": float(required),
                "min_years_to_run": float(minimum),
                "fail_if_below_min": bool(fail),
            }
        }
    }


def test_stage27_coverage_gate_fails_without_allow_or_fallback(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_symbol_1m(
        data_dir / "binance" / "BTC-USDT" / "1m.parquet",
        "2022-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
    )
    _write_symbol_1m(
        data_dir / "binance" / "ETH-USDT" / "1m.parquet",
        "2025-12-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
    )

    decision = evaluate_coverage_gate(
        config=_cfg(),
        symbols=["BTC/USDT", "ETH/USDT"],
        timeframe="1m",
        data_dir=data_dir,
        allow_insufficient_data=False,
        auto_btc_fallback=False,
    )
    assert decision.can_run is False
    assert decision.status == "INSUFFICIENT_DATA"
    assert "ETH/USDT" in decision.insufficient_symbols


def test_stage27_coverage_gate_auto_fallback_to_btc(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    _write_symbol_1m(
        data_dir / "binance" / "BTC-USDT" / "1m.parquet",
        "2022-01-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
    )
    _write_symbol_1m(
        data_dir / "binance" / "ETH-USDT" / "1m.parquet",
        "2025-12-01T00:00:00Z",
        "2026-01-01T00:00:00Z",
    )
    decision = evaluate_coverage_gate(
        config=_cfg(),
        symbols=["BTC/USDT", "ETH/USDT"],
        timeframe="1m",
        data_dir=data_dir,
        allow_insufficient_data=False,
        auto_btc_fallback=True,
    )
    assert decision.can_run is True
    assert decision.used_symbols == ["BTC/USDT"]
    assert decision.disabled_symbols == ["ETH/USDT"]
    assert "auto_fallback_disabled:ETH/USDT" in decision.notes

