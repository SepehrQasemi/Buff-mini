"""Stage-5 strategy library export tests."""

from __future__ import annotations

import json
from pathlib import Path

from buffmini.ui.components.library import export_run_to_library, load_library_index, load_strategy_params


def test_export_to_library_writes_strategy_folder_and_updates_index(tmp_path) -> None:
    runs_dir = tmp_path / "runs"
    library_dir = tmp_path / "library"

    pipeline_run_id = "pipeline_a"
    stage2_run_id = "stage2_a"
    stage3_run_id = "stage3_3_a"

    (runs_dir / pipeline_run_id).mkdir(parents=True, exist_ok=True)
    (runs_dir / stage2_run_id).mkdir(parents=True, exist_ok=True)
    (runs_dir / stage3_run_id).mkdir(parents=True, exist_ok=True)

    (runs_dir / pipeline_run_id / "pipeline_summary.json").write_text(
        json.dumps(
            {
                "run_id": pipeline_run_id,
                "stage2_run_id": stage2_run_id,
                "stage3_3_run_id": stage3_run_id,
                "stage1_run_id": "stage1_a",
                "chosen_method": "equal",
                "chosen_leverage": 2.0,
            }
        ),
        encoding="utf-8",
    )

    (runs_dir / stage3_run_id / "selector_summary.json").write_text(
        json.dumps(
            {
                "run_id": stage3_run_id,
                "stage2_run_id": stage2_run_id,
                "stage1_run_id": "stage1_a",
                "config_hash": "cfg_hash",
                "data_hash": "data_hash",
                "overall_choice": {"method": "vol", "chosen_leverage": 3.0},
            }
        ),
        encoding="utf-8",
    )

    (runs_dir / stage2_run_id / "portfolio_summary.json").write_text(
        json.dumps(
            {
                "run_id": stage2_run_id,
                "universe": {"symbols": ["BTC/USDT", "ETH/USDT"], "timeframe": "1h"},
            }
        ),
        encoding="utf-8",
    )
    (runs_dir / stage2_run_id / "weights_vol.csv").write_text("candidate_id,weight\nc1,1.0\n", encoding="utf-8")

    card = export_run_to_library(
        run_id=pipeline_run_id,
        display_name="My Library Strategy",
        runs_dir=runs_dir,
        library_dir=library_dir,
    )

    assert card["display_name"] == "My Library Strategy"
    strategy_id = card["strategy_id"]
    strategy_dir = library_dir / "strategies" / strategy_id
    assert strategy_dir.exists()
    assert (strategy_dir / "strategy_card.json").exists()
    assert (strategy_dir / "params.json").exists()
    assert (strategy_dir / "origin.json").exists()

    index_payload = load_library_index(library_dir)
    ids = [item["strategy_id"] for item in index_payload["strategies"]]
    assert strategy_id in ids

    params = load_strategy_params(strategy_id, library_dir=library_dir)
    assert params["method"] == "vol"
    assert float(params["leverage"]) == 3.0
