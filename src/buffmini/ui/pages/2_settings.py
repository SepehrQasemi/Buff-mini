"""Settings page."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from buffmini.config import load_config


root = Path(__file__).resolve().parents[4]
config_path = root / "configs" / "default.yaml"

st.title("Settings")

if config_path.exists():
    config = load_config(config_path)
    st.json(config)
else:
    st.error(f"Missing config file: {config_path}")
