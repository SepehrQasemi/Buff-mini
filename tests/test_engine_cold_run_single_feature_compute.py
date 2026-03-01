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


def test_cold_run_single_feature_compute_per_tf(tmp_path: Path) -> None:
    run_stage11_5_bench = _load_bench_runner()
    summary = run_stage11_5_bench(
        config=load_config(DEFAULT_CONFIG_PATH),
        symbols=["BTC/USDT"],
        base_timeframe="1m",
        tfs=["15m", "1h", "2h", "4h"],
        seed=42,
        data_dir=tmp_path / "raw",
        derived_dir=tmp_path / "derived",
        feature_cache_dir=tmp_path / "features_cache",
        runs_dir=tmp_path / "runs",
        dry_run=True,
        dry_run_rows=2400,
    )

    cold_calls = {str(k): int(v) for k, v in summary["cold_run_feature_calls_per_tf"].items()}
    rerun_calls = {str(k): int(v) for k, v in summary["rerun_feature_calls_per_tf"].items()}

    assert cold_calls == {"15m": 1, "1h": 1, "2h": 1, "4h": 1}
    assert rerun_calls == {}
    assert float(summary["derived_cache_hit_rate"]) == 1.0
