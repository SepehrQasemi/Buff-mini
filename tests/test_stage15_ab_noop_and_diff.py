from __future__ import annotations

from buffmini.alpha_v2.ab_runner import run_ab_compare
from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH


def test_stage15_ab_noop_when_alpha_disabled() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    out = run_ab_compare(config=cfg, seed=42, dry_run=True, alpha_enabled=False, max_rows=600)
    summary = out["summary"]
    assert summary["classic"]["trades_hash"] == summary["alpha_v2"]["trades_hash"]
    assert summary["classic"]["equity_hash"] == summary["alpha_v2"]["equity_hash"]


def test_stage15_ab_diff_when_alpha_enabled() -> None:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    out = run_ab_compare(config=cfg, seed=42, dry_run=True, alpha_enabled=True, max_rows=600)
    summary = out["summary"]
    assert float(summary["activation_stats"]["pct_nonzero_confidence"]) > 0.0
    assert summary["classic"]["trades_hash"] != summary["alpha_v2"]["trades_hash"]

