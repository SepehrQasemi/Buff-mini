from __future__ import annotations

import json
import importlib.util
from pathlib import Path

from buffmini.config import load_config


def _load_stage11_matrix_module():
    path = Path("scripts/run_stage11_matrix.py")
    spec = importlib.util.spec_from_file_location("stage11_matrix_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load scripts/run_stage11_matrix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage11_matrix_report_schema(tmp_path: Path) -> None:
    module = _load_stage11_matrix_module()
    config = load_config(Path("configs/default.yaml"))
    config["evaluation"]["stage10"]["evaluation"]["dry_run_rows"] = 500

    docs_dir = tmp_path / "docs"
    runs_dir = tmp_path / "runs"
    summary = module.run_stage11_matrix(
        config=config,
        seed=42,
        symbols=["BTC/USDT"],
        timeframe="1h",
        default_window_months=3,
        run_real=False,
        runs_dir=runs_dir,
        docs_dir=docs_dir,
        data_dir=Path("data/raw"),
        derived_dir=Path("data/derived"),
        allow_noop=False,
    )
    module.validate_stage11_1_summary_schema(summary)

    json_path = docs_dir / "stage11_1_report_summary.json"
    md_path = docs_dir / "stage11_1_report.md"
    assert json_path.exists()
    assert md_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    module.validate_stage11_1_summary_schema(payload)
    assert str(payload["stage"]) == "11.1"
    assert {"baseline", "bias", "confirm", "bias_confirm"}.issubset({str(row["name"]) for row in payload["modes"]})
