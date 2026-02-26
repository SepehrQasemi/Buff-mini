"""Project constants and shared paths."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RUNS_DIR = PROJECT_ROOT / "runs"

DEFAULT_TIMEFRAME = "1h"
OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
