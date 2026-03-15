from __future__ import annotations

import pandas as pd

from buffmini.data.continuity import continuity_report
from buffmini.research.canonical_repair import build_contiguous_evaluation_suffix


def test_stage96_build_contiguous_suffix_after_last_gap() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01T00:00:00Z",
                    "2024-01-01T01:00:00Z",
                    "2024-01-01T04:00:00Z",
                    "2024-01-01T05:00:00Z",
                ],
                utc=True,
            ),
            "open": [1.0, 1.1, 1.2, 1.3],
            "high": [1.0, 1.1, 1.2, 1.3],
            "low": [1.0, 1.1, 1.2, 1.3],
            "close": [1.0, 1.1, 1.2, 1.3],
            "volume": [10.0, 10.0, 10.0, 10.0],
        }
    )
    repaired, meta = build_contiguous_evaluation_suffix(frame, timeframe="1h")
    report = continuity_report(repaired, timeframe="1h", max_gap_bars=0)
    assert meta["trim_start_ts"] == "2024-01-01T04:00:00+00:00"
    assert len(repaired) == 2
    assert report["gap_count"] == 0
    assert report["passes_strict"] is True
