"""Stage-2.5 walk-forward page."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from buffmini.constants import DEFAULT_WALKFORWARD_FORWARD_DAYS, DEFAULT_WALKFORWARD_NUM_WINDOWS
from buffmini.portfolio.walkforward import run_stage2_walkforward


root = Path(__file__).resolve().parents[4]
runs_dir = root / "runs"
data_dir = root / "data" / "raw"


def _latest_stage2_run_id() -> str:
    candidates = sorted([path for path in runs_dir.glob("*_stage2") if path.is_dir()], reverse=True)
    return candidates[0].name if candidates else ""


st.title("Stage-2.5 Walk-Forward")
st.write("Run rolling walk-forward validation on an existing Stage-2 portfolio run using cached local data.")

stage2_run_id = st.text_input("Stage-2 run id", value=_latest_stage2_run_id())
forward_days = st.number_input("Forward days", min_value=1, value=DEFAULT_WALKFORWARD_FORWARD_DAYS, step=1)
num_windows = st.number_input("Number of windows", min_value=1, value=DEFAULT_WALKFORWARD_NUM_WINDOWS, step=1)
reserve_forward_days = st.number_input(
    "Reserve forward days",
    min_value=0,
    value=int(forward_days) * int(num_windows),
    step=1,
    help="Bars in this reserved tail are excluded from the holdout so forward windows remain strictly out-of-sample.",
)
seed = st.number_input("Seed", min_value=0, value=42, step=1)

if st.button("Run Stage-2.5"):
    if not stage2_run_id:
        st.error("Stage-2 run id is required.")
    else:
        run_dir = run_stage2_walkforward(
            stage2_run_id=stage2_run_id,
            forward_days=int(forward_days),
            num_windows=int(num_windows),
            seed=int(seed),
            reserve_forward_days=int(reserve_forward_days),
            runs_dir=runs_dir,
            data_dir=data_dir,
            cli_command=(
                "python scripts/run_stage2_walkforward.py "
                f"--stage2-run-id {stage2_run_id} --forward-days {int(forward_days)} "
                f"--num-windows {int(num_windows)} --seed {int(seed)} "
                f"--reserve-forward-days {int(reserve_forward_days)}"
            ),
        )
        summary = json.loads((run_dir / "walkforward_summary.json").read_text(encoding="utf-8"))

        table_rows = []
        for method_key in ["equal", "vol", "corr-min"]:
            payload = summary["method_summaries"].get(method_key)
            if payload is None:
                continue
            stability = payload["stability"]
            table_rows.append(
                {
                    "method": method_key,
                    "pf_holdout": stability["pf_holdout"],
                    "pf_forward_mean": stability["pf_forward_mean"],
                    "degradation_ratio": stability["degradation_ratio"],
                    "worst_forward_pf": stability["worst_forward_pf"],
                    "dd_growth_ratio": stability["dd_growth_ratio"],
                    "usable_windows": stability["usable_windows"],
                    "classification": stability["classification"],
                }
            )

        st.success(f"Stage-2.5 completed: {summary['run_id']}")
        st.write(f"Recommendation: `{summary['overall_recommendation']}`")
        st.dataframe(pd.DataFrame(table_rows))
        st.write("Artifacts")
        for path in sorted(run_dir.iterdir()):
            if path.is_file():
                st.write(f"`{path}`")
