"""Trade map data contract tests."""

from __future__ import annotations

import json

import pandas as pd

from buffmini.data.storage import save_parquet
from buffmini.ui.components.trade_map import plot_trade_map


def test_trade_map_reads_bundle_and_returns_markers(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "r1"
    bundle_dir = run_dir / "ui_bundle"
    data_dir = tmp_path / "data" / "raw"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=24, freq="H", tz="UTC"),
            "open": [100 + i for i in range(24)],
            "high": [101 + i for i in range(24)],
            "low": [99 + i for i in range(24)],
            "close": [100 + i for i in range(24)],
            "volume": [1.0] * 24,
        }
    )
    save_parquet(bars, symbol="BTC/USDT", timeframe="1h", data_dir=data_dir)

    (bundle_dir / "summary_ui.json").write_text(
        json.dumps(
            {
                "run_id": "r1",
                "timeframe": "1h",
                "execution_mode": "net",
                "run_window_start_ts": "2026-01-01T00:00:00+00:00",
                "run_window_end_ts": "2026-01-01T23:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "timestamp": ["2026-01-01T02:00:00+00:00", "2026-01-01T05:00:00+00:00"],
            "symbol": ["BTC/USDT", "BTC/USDT"],
            "direction": [1, -1],
            "action": ["entry", "entry"],
            "notional_fraction_of_equity": [0.2, 0.2],
        }
    ).to_csv(bundle_dir / "trades.csv", index=False)

    fig, warnings, markers = plot_trade_map(
        run_dir=run_dir,
        symbol="BTC/USDT",
        direction_filter="both",
        data_dir=data_dir,
    )

    assert fig is not None
    assert markers is not None
    assert not markers.empty
    assert isinstance(warnings, list)
