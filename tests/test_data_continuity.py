from __future__ import annotations

import pandas as pd

from buffmini.data.continuity import continuity_report


def test_continuity_detects_gap() -> None:
    idx = pd.to_datetime(
        [
            "2025-01-01T00:00:00Z",
            "2025-01-01T01:00:00Z",
            "2025-01-01T03:00:00Z",
        ],
        utc=True,
    )
    frame = pd.DataFrame({"timestamp": idx, "close": [1.0, 1.1, 1.2]})
    report = continuity_report(frame, timeframe="1h", max_gap_bars=0)
    assert report["passes_strict"] is False
    assert report["gap_count"] >= 1
