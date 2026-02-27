"""Stage-5 Strategy Library page."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from buffmini.constants import PROJECT_ROOT
from buffmini.ui.components.library import load_library_index, load_strategy_params
from buffmini.ui.components.markdown_viewer import render_markdown_file


LIBRARY_DIR = PROJECT_ROOT / "library"

st.title("Stage-5 Strategy Library")
st.write("Browse reusable strategy cards exported from completed runs.")

index_payload = load_library_index(LIBRARY_DIR)
strategies = index_payload.get("strategies", [])
if not strategies:
    st.info("Library is empty. Save a run from Results Studio first.")
    st.stop()

symbols_options = sorted({symbol for item in strategies for symbol in item.get("symbols", [])})
timeframe_options = sorted({str(item.get("timeframe", "")) for item in strategies if item.get("timeframe")})
method_options = sorted({str(item.get("method", "")) for item in strategies if item.get("method")})
execution_options = sorted({str(item.get("execution_mode", "")) for item in strategies if item.get("execution_mode")})

col1, col2, col3, col4 = st.columns(4)
with col1:
    symbol_filter = st.selectbox("Symbol", options=["All", *symbols_options], index=0)
with col2:
    timeframe_filter = st.selectbox("Timeframe", options=["All", *timeframe_options], index=0)
with col3:
    method_filter = st.selectbox("Method", options=["All", *method_options], index=0)
with col4:
    execution_filter = st.selectbox("Execution", options=["All", *execution_options], index=0)

filtered = []
for card in strategies:
    if symbol_filter != "All" and symbol_filter not in card.get("symbols", []):
        continue
    if timeframe_filter != "All" and str(card.get("timeframe")) != timeframe_filter:
        continue
    if method_filter != "All" and str(card.get("method")) != method_filter:
        continue
    if execution_filter != "All" and str(card.get("execution_mode")) != execution_filter:
        continue
    filtered.append(card)

st.caption(f"Showing {len(filtered)} strategy cards")

for card in filtered:
    strategy_id = str(card["strategy_id"])
    strategy_dir = LIBRARY_DIR / "strategies" / strategy_id
    with st.expander(f"{card.get('display_name', strategy_id)} ({strategy_id})", expanded=False):
        st.write(
            f"method={card.get('method')} | leverage={card.get('leverage')} | "
            f"symbols={','.join(card.get('symbols', []))} | timeframe={card.get('timeframe')}"
        )
        st.write(f"origin_run_id={card.get('origin_run_id')}")

        params = load_strategy_params(strategy_id, library_dir=LIBRARY_DIR)
        st.json(params)

        if st.button("Use this strategy", key=f"use_{strategy_id}"):
            st.session_state["strategy_lab_defaults"] = {
                "symbols": card.get("symbols", ["BTC/USDT", "ETH/USDT"]),
                "timeframe": card.get("timeframe", "1h"),
                "method": card.get("method"),
                "leverage": card.get("leverage", 1.0),
                **params,
            }
            st.success("Strategy loaded into Strategy Lab defaults.")
            if hasattr(st, "switch_page"):
                st.switch_page("pages/20_strategy_lab.py")

        spec_path = strategy_dir / "strategy_spec.md"
        render_markdown_file(spec_path, title="Spec Preview", fallback="No strategy_spec.md found")
