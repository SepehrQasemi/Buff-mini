"""Paper trading playback artifact contract tests."""

from __future__ import annotations

import json

import pandas as pd

from buffmini.ui.components.paper_playback import load_playback_artifacts, playback_snapshot


def test_paper_playback_contract_parses_minimal_bundle(tmp_path) -> None:
    run_dir = tmp_path / "runs" / "r1"
    bundle = run_dir / "ui_bundle"
    bundle.mkdir(parents=True, exist_ok=True)

    (bundle / "summary_ui.json").write_text(
        json.dumps({"run_id": "r1", "timeframe": "1h", "execution_mode": "net"}),
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "timestamp": ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
            "symbol": ["BTC/USDT", "ALL"],
            "action": ["open", "hold"],
            "exposure": [0.5, 0.4],
            "reason": ["signal", ""],
            "equity": [10000, 10010],
        }
    ).to_csv(bundle / "playback_state.csv", index=False)

    summary, playback, events, warnings = load_playback_artifacts(run_dir)
    assert summary["run_id"] == "r1"
    assert not playback.empty
    assert isinstance(events, pd.DataFrame)
    assert isinstance(warnings, list)

    snap = playback_snapshot(playback, 1)
    assert snap["bar_index"] == 1
    assert snap["timestamp"] is not None
    assert snap["current_exposure"] >= 0.0
