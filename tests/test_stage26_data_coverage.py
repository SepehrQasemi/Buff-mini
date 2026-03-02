from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from buffmini.stage26.coverage import audit_symbol_coverage


def _write_ohlcv(path: Path, *, start: str, periods: int, freq: str) -> None:
    ts = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    frame = pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100.0 + float(i) * 0.01 for i in range(periods)],
            "high": [100.2 + float(i) * 0.01 for i in range(periods)],
            "low": [99.8 + float(i) * 0.01 for i in range(periods)],
            "close": [100.1 + float(i) * 0.01 for i in range(periods)],
            "volume": [10.0 for _ in range(periods)],
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def test_stage26_coverage_function_metrics(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    _write_ohlcv(
        data_dir / "BTC-USDT_1h.parquet",
        start="2024-01-01T00:00:00Z",
        periods=24 * 400,
        freq="1h",
    )
    result = audit_symbol_coverage(
        symbol="BTC/USDT",
        timeframe="1h",
        data_dir=data_dir,
    )
    assert result.exists is True
    assert result.coverage_years > 1.0
    assert result.non_monotonic is False
    assert result.duplicate_timestamps == 0
    assert result.expected_bars >= result.observed_bars


def test_stage26_coverage_script_exit_codes(tmp_path: Path) -> None:
    data_dir = tmp_path / "raw"
    docs_dir = tmp_path / "docs"
    _write_ohlcv(
        data_dir / "BTC-USDT_1h.parquet",
        start="2024-01-01T00:00:00Z",
        periods=24 * 500,
        freq="1h",
    )
    _write_ohlcv(
        data_dir / "ETH-USDT_1h.parquet",
        start="2024-01-01T00:00:00Z",
        periods=24 * 500,
        freq="1h",
    )

    ok = subprocess.run(
        [
            sys.executable,
            "scripts/audit_data_coverage.py",
            "--symbols",
            "BTC/USDT,ETH/USDT",
            "--base-timeframe",
            "1h",
            "--required-years",
            "1",
            "--data-dir",
            str(data_dir),
            "--docs-dir",
            str(docs_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0
    assert (docs_dir / "stage26_data_coverage_4y.json").exists()
    assert (docs_dir / "stage26_data_coverage_4y.md").exists()

    fail = subprocess.run(
        [
            sys.executable,
            "scripts/audit_data_coverage.py",
            "--symbols",
            "BTC/USDT,ETH/USDT",
            "--base-timeframe",
            "1h",
            "--required-years",
            "4",
            "--data-dir",
            str(data_dir),
            "--docs-dir",
            str(docs_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert fail.returncode == 2
