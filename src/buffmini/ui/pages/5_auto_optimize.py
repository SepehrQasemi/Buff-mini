"""Stage-1 Auto Optimize page."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import streamlit as st


root = Path(__file__).resolve().parents[4]
default_config = root / "configs" / "default.yaml"

st.title("Auto Optimize (Stage-1)")
st.write("Run seeded random-search optimization with funnel validation and display top 3 candidates.")

candidate_count = st.number_input("Candidate count", min_value=50, max_value=20000, value=500, step=50)
seed = st.number_input("Seed", min_value=0, max_value=2_147_483_647, value=42, step=1)
cost_pct = st.number_input(
    "Round-trip cost (%)",
    min_value=0.0,
    max_value=5.0,
    value=1.0,
    step=0.1,
    format="%.4f",
)
stage_a_months = st.number_input("Stage A months", min_value=3, max_value=36, value=9, step=1)
stage_b_months = st.number_input("Stage B months", min_value=6, max_value=72, value=24, step=1)
holdout_months = st.number_input("Holdout months", min_value=3, max_value=36, value=9, step=1)
dry_run = st.checkbox("Dry-run (synthetic/offline)", value=True)

if st.button("Run Stage-1"):
    run_id = f"stage1_ui_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    command = [
        sys.executable,
        "scripts/run_discovery.py",
        "--config",
        str(default_config),
        "--run-id",
        run_id,
        "--candidate-count",
        str(int(candidate_count)),
        "--seed",
        str(int(seed)),
        "--cost-pct",
        str(float(cost_pct)),
        "--stage-a-months",
        str(int(stage_a_months)),
        "--stage-b-months",
        str(int(stage_b_months)),
        "--holdout-months",
        str(int(holdout_months)),
    ]
    if dry_run:
        command.append("--dry-run")

    result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
    st.code(result.stdout if result.stdout else "(no output)")
    if result.returncode != 0:
        st.error(result.stderr if result.stderr else "Stage-1 run failed.")
    else:
        run_dir = root / "runs" / run_id
        strategies_path = run_dir / "strategies.json"
        if not strategies_path.exists():
            st.warning(f"Run finished but strategies.json was not found in {run_dir}.")
        else:
            strategies = json.loads(strategies_path.read_text(encoding="utf-8"))
            rows = []
            for item in strategies:
                m = item.get("metrics_holdout", {})
                rows.append(
                    {
                        "rank": item.get("rank"),
                        "family": item.get("family"),
                        "gating_mode": item.get("gating_mode"),
                        "exit_mode": item.get("exit_mode"),
                        "profit_factor": m.get("profit_factor"),
                        "expectancy": m.get("expectancy"),
                        "max_drawdown": m.get("max_drawdown"),
                        "trade_count": m.get("trade_count"),
                    }
                )
            st.success(f"Stage-1 completed. Run ID: {run_id}")
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.caption(f"Artifacts: {run_dir}")
