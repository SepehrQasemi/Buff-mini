from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH


def _load_bench_runner():
    module_path = Path("scripts") / "bench_engine_stage11_5.py"
    spec = importlib.util.spec_from_file_location("bench_engine_stage11_5", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load bench_engine_stage11_5 module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_stage11_5_bench


def test_stage11_5_bench_emits_required_fields(tmp_path: Path) -> None:
    run_stage11_5_bench = _load_bench_runner()
    data_dir = tmp_path / "raw"
    derived_dir = tmp_path / "derived"
    runs_dir = tmp_path / "runs"
    data_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    summary = run_stage11_5_bench(
        config=load_config(DEFAULT_CONFIG_PATH),
        symbols=["BTC/USDT"],
        base_timeframe="1m",
        tfs=["15m", "1h"],
        seed=42,
        data_dir=data_dir,
        derived_dir=derived_dir,
        runs_dir=runs_dir,
        dry_run=True,
        dry_run_rows=1800,
    )

    required = {
        "run_id",
        "first_run_seconds",
        "second_run_seconds",
        "derived_cache_hit_rate",
        "feature_cache_hit_rate",
        "first_breakdown",
        "second_breakdown",
        "first_cache",
        "second_cache",
        "data_hash",
        "config_hash",
    }
    assert required.issubset(summary.keys())
    assert float(summary["first_run_seconds"]) >= 0.0
    assert float(summary["second_run_seconds"]) >= 0.0
    assert 0.0 <= float(summary["derived_cache_hit_rate"]) <= 1.0
    assert 0.0 <= float(summary["feature_cache_hit_rate"]) <= 1.0

    perf_meta = runs_dir / str(summary["run_id"]) / "perf_meta.json"
    assert perf_meta.exists()
    payload = json.loads(perf_meta.read_text(encoding="utf-8"))
    assert payload["run_id"] == summary["run_id"]
    assert "features_compute_calls_per_tf" in payload["second_cache"]
