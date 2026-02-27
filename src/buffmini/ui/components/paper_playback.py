"""Paper trading playback helpers (artifact-driven)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_PLAYBACK_COLUMNS = {"timestamp", "symbol", "action", "exposure", "reason", "equity"}


def load_playback_artifacts(run_dir: Path) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, list[str]]:
    """Load summary/playback/events from ui_bundle."""

    warnings: list[str] = []
    bundle = Path(run_dir) / "ui_bundle"

    summary_path = bundle / "summary_ui.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        try:
            parsed = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                summary = parsed
        except Exception as exc:
            warnings.append(f"Failed parsing summary_ui.json: {exc}")
    else:
        warnings.append("summary_ui.json missing")

    playback_path = bundle / "playback_state.csv"
    if playback_path.exists():
        try:
            playback = pd.read_csv(playback_path)
        except Exception as exc:
            warnings.append(f"Failed reading playback_state.csv: {exc}")
            playback = pd.DataFrame()
    else:
        warnings.append("playback_state.csv missing")
        playback = pd.DataFrame()

    events_path = bundle / "events.csv"
    if events_path.exists():
        try:
            events = pd.read_csv(events_path)
        except Exception as exc:
            warnings.append(f"Failed reading events.csv: {exc}")
            events = pd.DataFrame()
    else:
        events = pd.DataFrame()

    if not playback.empty:
        missing = REQUIRED_PLAYBACK_COLUMNS - set(playback.columns)
        if missing:
            warnings.append(f"playback_state.csv missing columns: {sorted(missing)}")
        playback = playback.copy()
        playback["timestamp"] = pd.to_datetime(playback["timestamp"], utc=True, errors="coerce")
        playback = playback.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    if not events.empty and "ts" in events.columns:
        events = events.copy()
        events["ts"] = pd.to_datetime(events["ts"], utc=True, errors="coerce")

    return summary, playback, events, warnings


def playback_snapshot(playback: pd.DataFrame, bar_index: int) -> dict[str, Any]:
    """Return deterministic snapshot at playback index."""

    if playback.empty:
        return {
            "bar_index": 0,
            "timestamp": None,
            "rows": pd.DataFrame(),
            "last_action": None,
            "current_exposure": 0.0,
            "equity": None,
        }

    idx = max(0, min(int(bar_index), len(playback) - 1))
    current_ts = pd.to_datetime(playback.iloc[idx]["timestamp"], utc=True)
    rows = playback[playback["timestamp"] == current_ts].copy().reset_index(drop=True)

    last_action = None
    if not rows.empty:
        last_action = {
            "symbol": str(rows.iloc[-1].get("symbol", "")),
            "action": str(rows.iloc[-1].get("action", "")),
            "reason": str(rows.iloc[-1].get("reason", "")),
        }

    current_exposure = float(pd.to_numeric(rows.get("exposure", 0.0), errors="coerce").fillna(0.0).max()) if not rows.empty else 0.0
    equity_value = None
    if not rows.empty and "equity" in rows.columns:
        try:
            equity_value = float(rows.iloc[-1]["equity"])
        except Exception:
            equity_value = None

    return {
        "bar_index": idx,
        "timestamp": current_ts,
        "rows": rows,
        "last_action": last_action,
        "current_exposure": current_exposure,
        "equity": equity_value,
    }
