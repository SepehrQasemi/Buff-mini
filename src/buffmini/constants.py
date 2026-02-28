"""Project constants and shared paths."""

from __future__ import annotations

import os
from pathlib import Path


def _detect_project_root() -> Path:
    """Detect repository root for both editable and wheel installs."""

    def _is_repo_root(path: Path) -> bool:
        return (path / "pyproject.toml").exists() and (path / "configs" / "default.yaml").exists()

    env_root = os.environ.get("BUFFMINI_PROJECT_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if _is_repo_root(candidate):
            return candidate

    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        if _is_repo_root(candidate):
            return candidate

    # Fallback for editable/development layout.
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT = _detect_project_root()
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DERIVED_DATA_DIR = DATA_DIR / "derived"
DATA_CACHE_DIR = DATA_DIR / "cache"
RUNS_DIR = PROJECT_ROOT / "runs"

DEFAULT_TIMEFRAME = "1h"
OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
DEFAULT_WALKFORWARD_FORWARD_DAYS = 7
DEFAULT_WALKFORWARD_NUM_WINDOWS = 8
DEFAULT_WALKFORWARD_MIN_USABLE_WINDOWS = 3
DEFAULT_WALKFORWARD_MIN_FORWARD_TRADES = 10
DEFAULT_WALKFORWARD_MIN_FORWARD_EXPOSURE = 0.01
DEFAULT_WALKFORWARD_PF_CLIP_MAX = 5.0
DEFAULT_WALKFORWARD_STABILITY_METRIC = "exp_lcb"
