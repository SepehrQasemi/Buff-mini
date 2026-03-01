from __future__ import annotations

import importlib.util
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


def test_no_double_compute_regression(tmp_path: Path) -> None:
    run_stage11_5_bench = _load_bench_runner()
    summary = run_stage11_5_bench(
        config=load_config(DEFAULT_CONFIG_PATH),
        symbols=["BTC/USDT"],
        base_timeframe="1m",
        tfs=["15m", "1h", "2h"],
        seed=123,
        data_dir=tmp_path / "raw",
        derived_dir=tmp_path / "derived",
        feature_cache_dir=tmp_path / "features_cache",
        runs_dir=tmp_path / "runs",
        dry_run=True,
        dry_run_rows=1800,
    )

    cold_calls = summary["cold_run_feature_calls_per_tf"]
    assert all(int(cold_calls.get(tf, 0)) == 1 for tf in ["15m", "1h", "2h"])
    rerun_calls = summary["rerun_feature_calls_per_tf"]
    assert rerun_calls == {}
    assert float(summary["derived_cache_hit_rate"]) == 1.0
