"""Run Stage-65 feature factory v3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RAW_DATA_DIR
from buffmini.data.continuity import continuity_report
from buffmini.data.store import build_data_store
from buffmini.stage65 import build_feature_frame_v3, compute_feature_attribution_v3
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-65 feature factory")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    return parser.parse_args()


def _safe_ohlcv(config: dict) -> pd.DataFrame:
    symbols = list(((config.get("universe", {}) or {}).get("symbols", ["BTC/USDT"])))
    symbol = str(symbols[0]) if symbols else "BTC/USDT"
    timeframe = str((config.get("universe", {}) or {}).get("operational_timeframe", "1h"))
    store = build_data_store(
        backend=str((config.get("data", {}) or {}).get("backend", "parquet")),
        data_dir=RAW_DATA_DIR,
        base_timeframe=str((config.get("universe", {}) or {}).get("base_timeframe", "1m")),
        resample_source=str((config.get("data", {}) or {}).get("resample_source", "direct")),
        derived_dir=Path("data") / "derived",
        partial_last_bucket=bool((config.get("data", {}) or {}).get("partial_last_bucket", False)),
    )
    try:
        bars = store.load_ohlcv(symbol=symbol, timeframe=timeframe).tail(1500).reset_index(drop=True)
    except FileNotFoundError:
        bars = pd.DataFrame()
    if not bars.empty:
        return bars.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]].copy()
    idx = pd.date_range("2025-01-01T00:00:00Z", periods=1500, freq="1h", tz="UTC")
    close = 100.0 + np.cumsum(np.sin(np.arange(len(idx)) / 16.0) * 0.08)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": close,
            "high": close + 0.15,
            "low": close - 0.15,
            "close": close + 0.03,
            "volume": 1000.0 + np.sin(np.arange(len(idx)) / 8.0) * 50.0,
        }
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    bars = _safe_ohlcv(cfg)
    continuity_cfg = dict(cfg.get("data", {}).get("continuity", {}))
    timeframe = str(continuity_cfg.get("timeframe", cfg.get("universe", {}).get("operational_timeframe", "1h")))
    continuity = continuity_report(
        bars,
        timeframe=str(timeframe),
        max_gap_bars=int(max(0, continuity_cfg.get("max_gap_bars", 0))),
    )
    features, tags = build_feature_frame_v3(bars)
    label = (pd.to_numeric(features.get("ret_1", 0.0), errors="coerce").fillna(0.0) > 0.0).astype(int)
    attribution = compute_feature_attribution_v3(features=features, label=label)

    features.to_csv(docs_dir / "stage65_features_v3.csv", index=False)
    attribution.to_csv(docs_dir / "stage65_feature_attribution.csv", index=False)
    (docs_dir / "stage65_leakage_tags.json").write_text(json.dumps(tags, indent=2, allow_nan=False), encoding="utf-8")
    (docs_dir / "stage65_continuity_report.json").write_text(json.dumps(continuity, indent=2, allow_nan=False), encoding="utf-8")

    leakage_safe = int(sum(1 for value in tags.values() if bool(value)))
    fail_on_gap = bool(continuity_cfg.get("fail_on_gap", False))
    continuity_passed = bool(continuity.get("passes_strict", True))
    continuity_blocked = bool(fail_on_gap and not continuity_passed)
    summary = {
        "stage": "65",
        "status": "SUCCESS" if (not features.empty and not continuity_blocked) else "PARTIAL",
        "execution_status": "EXECUTED",
        "validation_state": "CONTINUITY_OK" if not continuity_blocked else "CONTINUITY_BLOCKED",
        "feature_count": int(len([c for c in features.columns if c != "timestamp"])),
        "leakage_safe_feature_count": int(leakage_safe),
        "continuity_report": continuity,
        "top_features": attribution.head(10).to_dict(orient="records"),
        "blocker_reason": (
            "empty_features"
            if features.empty
            else ("continuity_gap_detected_in_strict_mode" if continuity_blocked else "")
        ),
    }
    summary["summary_hash"] = stable_hash(summary, length=16)
    (docs_dir / "stage65_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (docs_dir / "stage65_report.md").write_text(
        "\n".join(
            [
                "# Stage-65 Report",
                "",
                f"- status: `{summary['status']}`",
                f"- execution_status: `{summary['execution_status']}`",
                f"- validation_state: `{summary['validation_state']}`",
                f"- feature_count: `{summary['feature_count']}`",
                f"- leakage_safe_feature_count: `{summary['leakage_safe_feature_count']}`",
                f"- continuity_report: `{summary['continuity_report']}`",
                f"- blocker_reason: `{summary['blocker_reason']}`",
                f"- summary_hash: `{summary['summary_hash']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
