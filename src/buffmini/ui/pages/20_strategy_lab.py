"""Stage-5 Strategy Lab page."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from buffmini.config import load_config
from buffmini.ui.components.run_exec import start_pipeline
from buffmini.ui.components.run_lock import get_active_run


ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = ROOT / "configs" / "default.yaml"


def _stage5_presets(config: dict) -> tuple[int, int, int]:
    stage1_default = int(config["evaluation"]["stage1"]["candidate_count"])
    ui_stage5 = config.get("ui", {}).get("stage5", {})
    presets = ui_stage5.get("presets", {}) if isinstance(ui_stage5, dict) else {}
    quick = int((presets.get("quick") or {}).get("candidate_count", min(1000, stage1_default)))
    full = int((presets.get("full") or {}).get("candidate_count", stage1_default))
    default_sim = int((presets.get("quick") or {}).get("run_stage4_simulate", 0))
    return quick, full, default_sim


config = load_config(CONFIG_PATH)
quick_count, full_count, default_sim = _stage5_presets(config)

defaults = st.session_state.get("strategy_lab_defaults", {})
default_symbols = defaults.get("symbols") or list(config["universe"]["symbols"])
default_timeframe = defaults.get("timeframe", "1h")
default_method = defaults.get("method")
default_leverage = defaults.get("leverage")

st.title("Stage-5 Strategy Lab")
st.write("Run the end-to-end pipeline with safe presets. One active run is allowed at a time.")

active = get_active_run()
if active:
    st.warning(f"Active run detected: `{active.get('run_id')}` (pid={active.get('pid')})")

with st.form("strategy_lab_form"):
    symbols = st.multiselect(
        "Symbols",
        options=["BTC/USDT", "ETH/USDT"],
        default=default_symbols,
    )
    timeframe = st.selectbox("Timeframe", options=["1h"], index=0 if default_timeframe == "1h" else 0)
    window_months = st.selectbox("Evaluation window (months)", options=[3, 6, 12, 36], index=2)
    fees_round_trip_pct = st.number_input("Round-trip fee (%)", min_value=0.0, max_value=100.0, value=float(config["costs"]["round_trip_cost_pct"]), step=0.01)

    exec_mode_display = st.selectbox("Execution mode", options=["Auto", "Net", "Hedge", "Isolated"], index=0)
    mode_preset = st.selectbox("Preset", options=["Quick", "Full"], index=0)
    autosave_to_library = st.checkbox("Auto-save best result to Library", value=False)
    autosave_display_name = st.text_input("Auto-save display name (optional)", value="")

    with st.expander("Advanced"):
        seed = st.number_input("Seed", min_value=0, value=int(config["search"]["seed"]), step=1)
        default_candidate_count = quick_count if mode_preset == "Quick" else full_count
        candidate_count = st.number_input("Candidate count", min_value=1, value=int(default_candidate_count), step=100)
        run_stage4_simulate = st.selectbox("Run Stage-4 simulation", options=[0, 1], index=default_sim)

    if mode_preset == "Quick":
        st.caption(f"Estimated workload: light to medium (about {quick_count} candidates by default).")
    else:
        st.caption(f"Estimated workload: heavy (about {full_count} candidates by default).")

    if default_method is not None:
        st.info(
            f"Prefilled from Library: method={default_method}, leverage={default_leverage}, "
            f"symbols={','.join(default_symbols)} timeframe={default_timeframe}"
        )

    submitted = st.form_submit_button("RUN", use_container_width=True, disabled=active is not None)

if active:
    if st.button("Open Run Monitor") and hasattr(st, "switch_page"):
        st.switch_page("pages/21_run_monitor.py")

if submitted:
    if not symbols:
        st.error("Select at least one symbol.")
    else:
        execution_mode = exec_mode_display.lower()
        if execution_mode == "auto":
            execution_mode = "net"

        params = {
            "symbols": symbols,
            "timeframe": timeframe,
            "window_months": int(window_months),
            "candidate_count": int(candidate_count),
            "mode": mode_preset.lower(),
            "execution_mode": execution_mode,
            "fees_round_trip_pct": float(fees_round_trip_pct),
            "seed": int(seed),
            "run_stage4_simulate": int(run_stage4_simulate),
        }
        try:
            run_id, pid = start_pipeline(params=params)
        except Exception as exc:
            st.error(f"Failed to start pipeline: {exc}")
        else:
            st.session_state["last_pipeline_run_id"] = run_id
            if autosave_to_library:
                targets = st.session_state.setdefault("stage5_autosave_targets", {})
                targets[run_id] = {
                    "display_name": autosave_display_name.strip() or "",
                }
            st.success(f"Started run `{run_id}` (pid={pid}).")
            if hasattr(st, "switch_page"):
                st.switch_page("pages/21_run_monitor.py")
