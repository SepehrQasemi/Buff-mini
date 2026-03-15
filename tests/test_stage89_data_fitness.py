from __future__ import annotations

from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.research.data_fitness import evaluate_data_fitness


def test_stage89_data_fitness_reports_live_vs_canonical() -> None:
    cfg = load_config(Path(DEFAULT_CONFIG_PATH))
    summary = evaluate_data_fitness(cfg, symbols=["BTC/USDT"], timeframes=["1h"])
    assert len(summary["rows"]) == 1
    row = summary["rows"][0]
    assert row["snapshot_available"] is True
    assert row["evaluation_usable_class"] in {"evaluation_usable_live", "evaluation_usable_canonical_only", "evaluation_blocked"}
    assert isinstance(row["canonical_snapshot_match"], bool)
