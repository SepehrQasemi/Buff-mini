"""Markdown rendering helper for artifact-driven UI pages."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


def render_markdown_file(path: Path, title: str | None = None, fallback: str = "File not available.") -> None:
    """Render markdown file contents with safe fallback message."""

    if title:
        st.subheader(title)
    if not path.exists():
        st.info(f"{fallback} (`{path}`)")
        return
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        st.warning(f"Could not read `{path}`: {exc}")
        return
    if not text.strip():
        st.info(f"Empty markdown file: `{path}`")
        return
    st.markdown(text)

