from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_run_stage28_module():
    path = Path("scripts/run_stage28.py")
    spec = importlib.util.spec_from_file_location("run_stage28_module", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage28_json_safe_handles_inf_values() -> None:
    module = _load_run_stage28_module()
    payload = {
        "a": float("inf"),
        "b": float("-inf"),
        "nested": {"c": [1.0, float("inf"), 2.0]},
    }
    out = module._json_safe(payload)
    assert out["a"] == 0.0
    assert out["b"] == 0.0
    assert out["nested"]["c"][1] == 0.0

