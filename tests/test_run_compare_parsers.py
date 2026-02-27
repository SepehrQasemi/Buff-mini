"""Run compare parser tests."""

from __future__ import annotations

import json

import pandas as pd

from buffmini.ui.components.run_compare import build_comparison_table


def test_run_compare_builds_side_by_side_table(tmp_path) -> None:
    runs_dir = tmp_path / "runs"
    run_a = runs_dir / "run_a" / "ui_bundle"
    run_b = runs_dir / "run_b" / "ui_bundle"
    run_a.mkdir(parents=True, exist_ok=True)
    run_b.mkdir(parents=True, exist_ok=True)

    selector_a = tmp_path / "selector_a.csv"
    selector_b = tmp_path / "selector_b.csv"
    pd.DataFrame(
        {
            "method": ["equal", "equal"],
            "leverage": [1.0, 2.0],
            "expected_log_growth": [0.01, 0.02],
            "return_p05": [0.0, 0.01],
            "return_median": [0.02, 0.03],
            "return_p95": [0.04, 0.05],
            "maxdd_p95": [0.10, 0.20],
            "p_ruin": [0.01, 0.02],
        }
    ).to_csv(selector_a, index=False)
    pd.DataFrame(
        {
            "method": ["equal", "equal"],
            "leverage": [1.0, 2.0],
            "expected_log_growth": [0.015, 0.025],
            "return_p05": [0.005, 0.015],
            "return_median": [0.025, 0.035],
            "return_p95": [0.045, 0.055],
            "maxdd_p95": [0.11, 0.21],
            "p_ruin": [0.012, 0.022],
        }
    ).to_csv(selector_b, index=False)

    (run_a / "summary_ui.json").write_text(
        json.dumps(
            {
                "run_id": "run_a",
                "chosen_method": "equal",
                "chosen_leverage": 2.0,
                "execution_mode": "net",
                "key_metrics": {"pf": 1.2, "maxdd": 0.2, "p_ruin": 0.02, "expected_log_growth": 0.02},
            }
        ),
        encoding="utf-8",
    )
    (run_b / "summary_ui.json").write_text(
        json.dumps(
            {
                "run_id": "run_b",
                "chosen_method": "equal",
                "chosen_leverage": 2.0,
                "execution_mode": "net",
                "key_metrics": {"pf": 1.25, "maxdd": 0.21, "p_ruin": 0.022, "expected_log_growth": 0.025},
            }
        ),
        encoding="utf-8",
    )

    (run_a / "charts_index.json").write_text(json.dumps({"selector_table": str(selector_a)}), encoding="utf-8")
    (run_b / "charts_index.json").write_text(json.dumps({"selector_table": str(selector_b)}), encoding="utf-8")

    table, warnings = build_comparison_table(runs_dir / "run_a", runs_dir / "run_b")

    assert not table.empty
    assert {"metric", "run_a", "run_b"}.issubset(set(table.columns))
    row = table[table["metric"] == "expected_log_growth"].iloc[0]
    assert float(row["run_a"]) == 0.02
    assert float(row["run_b"]) == 0.025
    assert isinstance(warnings, list)
