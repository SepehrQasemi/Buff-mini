from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from buffmini.stage65 import build_feature_frame_v3, compute_feature_attribution_v3


def test_stage65_builds_features_and_attribution() -> None:
    idx = pd.date_range("2025-01-01", periods=400, freq="h", tz="UTC")
    close = 100.0 + np.cumsum(np.sin(np.arange(len(idx)) / 10.0) * 0.1)
    bars = pd.DataFrame(
        {
            "timestamp": idx,
            "open": close,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close + 0.05,
            "volume": 1000 + np.cos(np.arange(len(idx)) / 8.0) * 40,
        }
    )
    features, tags = build_feature_frame_v3(bars)
    label = (features["ret_1"] > 0).astype(int)
    importance = compute_feature_attribution_v3(features=features, label=label)
    assert not features.empty
    assert len(tags) > 0
    assert not importance.empty
    assert set(importance.columns) == {"feature", "importance", "method"}


def test_stage65_runner_falls_back_without_repo_local_parquet(tmp_path: Path) -> None:
    docs_dir = tmp_path / "docs"
    config_path = tmp_path / "config.yaml"
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "default.yaml").read_text(encoding="utf-8"))
    config.setdefault("universe", {})["symbols"] = ["ZZZ/USDT"]
    config["universe"]["operational_timeframe"] = "1h"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    cmd = [
        sys.executable,
        "scripts/run_stage65.py",
        "--config",
        str(config_path),
        "--docs-dir",
        str(docs_dir),
    ]
    result = subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=False, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    summary = json.loads((docs_dir / "stage65_summary.json").read_text(encoding="utf-8"))
    assert summary["execution_status"] == "EXECUTED"
    assert summary["feature_count"] > 0
