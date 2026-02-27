"""Stage-5 paper trading playback page."""

from __future__ import annotations

import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from buffmini.constants import RAW_DATA_DIR, RUNS_DIR
from buffmini.data.storage import load_parquet
from buffmini.ui.components.paper_playback import load_playback_artifacts, playback_snapshot


st.title("Stage-5 Paper Trading Playback")
st.write("Bar-by-bar playback driven strictly by bundled artifacts.")

runs = [path.name for path in sorted(Path(RUNS_DIR).iterdir(), key=lambda item: item.stat().st_mtime, reverse=True) if path.is_dir()]
if not runs:
    st.info("No runs found.")
    st.stop()

run_id = st.selectbox("Run ID", options=runs, index=0)
run_dir = Path(RUNS_DIR) / run_id
summary, playback, events, warnings = load_playback_artifacts(run_dir)
for warning in warnings:
    st.caption(warning)

if playback.empty:
    st.info("No playback_state.csv available for this run.")
    st.stop()

symbol_options = sorted({str(item) for item in playback["symbol"].astype(str).tolist() if str(item) != "ALL"})
if not symbol_options:
    symbol_options = ["BTC/USDT"]
symbol = st.selectbox("Symbol", options=symbol_options, index=0)

bar_key = f"paper_bar_{run_id}"
play_key = f"paper_play_{run_id}"

if bar_key not in st.session_state:
    st.session_state[bar_key] = 0
if play_key not in st.session_state:
    st.session_state[play_key] = False

max_idx = max(0, len(playback) - 1)
current_bar = st.slider("Current bar", min_value=0, max_value=max_idx, value=int(st.session_state[bar_key]), step=1)
st.session_state[bar_key] = current_bar

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("Play"):
        st.session_state[play_key] = True
with c2:
    if st.button("Pause"):
        st.session_state[play_key] = False
with c3:
    speed = st.slider("Speed (bars/sec)", min_value=1, max_value=20, value=4, step=1)

snapshot = playback_snapshot(playback, st.session_state[bar_key])
current_ts = snapshot["timestamp"]

st.subheader("Current State")
st.write(
    {
        "run_id": run_id,
        "timestamp": None if current_ts is None else pd.Timestamp(current_ts).isoformat(),
        "current_exposure": snapshot["current_exposure"],
        "equity": snapshot["equity"],
        "last_action": snapshot["last_action"],
    }
)

st.subheader("Current Actions")
if snapshot["rows"].empty:
    st.info("No actions for current bar.")
else:
    st.dataframe(snapshot["rows"])

st.subheader("Price Playback")
timeframe = str(summary.get("timeframe", "1h"))
try:
    price = load_parquet(symbol=symbol, timeframe=timeframe, data_dir=RAW_DATA_DIR)
except Exception as exc:
    st.info(f"Price data unavailable: {exc}")
    price = pd.DataFrame()

if not price.empty and current_ts is not None:
    price["timestamp"] = pd.to_datetime(price["timestamp"], utc=True, errors="coerce")
    price = price[price["timestamp"] <= pd.Timestamp(current_ts)].copy()

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(price["timestamp"], price["close"], linewidth=1.1, label="close")

    marker_frame = playback[
        (playback["timestamp"] <= pd.Timestamp(current_ts))
        & (playback["symbol"].astype(str) == symbol)
        & (playback["action"].astype(str).isin(["open", "close"]))
    ].copy()
    if not marker_frame.empty:
        open_rows = marker_frame[marker_frame["action"].astype(str) == "open"]
        close_rows = marker_frame[marker_frame["action"].astype(str) == "close"]
        if not open_rows.empty:
            y_open = [_nearest_close(price, ts) for ts in open_rows["timestamp"]]
            ax.scatter(open_rows["timestamp"], y_open, marker="^", color="green", s=30, label="open")
        if not close_rows.empty:
            y_close = [_nearest_close(price, ts) for ts in close_rows["timestamp"]]
            ax.scatter(close_rows["timestamp"], y_close, marker="x", color="red", s=30, label="close")

    ax.set_title(f"Playback | {symbol}")
    ax.set_xlabel("timestamp")
    ax.set_ylabel("price")
    ax.legend(loc="upper left")
    fig.tight_layout()
    st.pyplot(fig)
else:
    st.info("Price playback unavailable for current snapshot.")

st.subheader("Kill-switch State")
if events.empty:
    st.info("No kill-switch events found.")
else:
    events_frame = events.copy()
    ts_col = "ts" if "ts" in events_frame.columns else "timestamp"
    events_frame[ts_col] = pd.to_datetime(events_frame[ts_col], utc=True, errors="coerce")
    if current_ts is not None:
        active = events_frame[events_frame[ts_col] <= pd.Timestamp(current_ts)].tail(5)
    else:
        active = events_frame.tail(5)
    st.dataframe(active)

if st.session_state.get(play_key) and st.session_state[bar_key] < max_idx:
    st.session_state[bar_key] = min(max_idx, st.session_state[bar_key] + int(speed))
    time.sleep(1.0 / float(max(speed, 1)))
    st.rerun()


def _nearest_close(price: pd.DataFrame, ts: pd.Timestamp) -> float:
    idx = (price["timestamp"] - ts).abs().idxmin()
    return float(price.loc[idx, "close"])
