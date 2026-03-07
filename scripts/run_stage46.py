"""Stage-46 analyst brain core part 2 runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RAW_DATA_DIR
from buffmini.data.store import build_data_store
from buffmini.stage46.part2 import (
    build_stage46_contract_records,
    compute_crowding_layer,
    compute_flow_regime_engine,
    compute_mtf_bias_completion,
    compute_trade_geometry_layer,
)
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-46 analyst brain part 2")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _safe_ohlcv(config: dict[str, Any]) -> pd.DataFrame:
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
    bars = store.load_ohlcv(symbol=symbol, timeframe=timeframe).tail(720).reset_index(drop=True)
    if not bars.empty:
        return bars[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    idx = pd.date_range("2025-01-01T00:00:00Z", periods=720, freq="1h", tz="UTC")
    base = 100.0 + np.cumsum(np.sin(np.arange(720) / 24.0) * 0.1)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": base,
            "high": base + 0.2,
            "low": base - 0.2,
            "close": base + 0.05,
            "volume": 900.0,
        }
    )


def _render(payload: dict[str, Any], *, notes: list[str]) -> str:
    lines = [
        "# Stage-46 Analyst Brain Part 2 Report",
        "",
        f"- status: `{payload.get('status', '')}`",
        f"- flow_regime_enabled: `{bool(payload.get('flow_regime_enabled', False))}`",
        f"- crowding_layer_enabled: `{bool(payload.get('crowding_layer_enabled', False))}`",
        f"- mtf_bias_enabled: `{bool(payload.get('mtf_bias_enabled', False))}`",
        f"- trade_geometry_enabled: `{bool(payload.get('trade_geometry_enabled', False))}`",
        f"- oi_short_only_guard_verified: `{bool(payload.get('oi_short_only_guard_verified', False))}`",
        f"- modules_contract_compliant: `{bool(payload.get('modules_contract_compliant', False))}`",
        "",
        "## Runtime Evidence",
        f"- flow_rows: `{int(payload.get('flow_rows', 0))}`",
        f"- crowding_rows: `{int(payload.get('crowding_rows', 0))}`",
        f"- mtf_rows: `{int(payload.get('mtf_rows', 0))}`",
        f"- geometry_rows: `{int(payload.get('geometry_rows', 0))}`",
        "",
    ]
    if notes:
        lines.append("## Partial Notes")
        lines.extend([f"- {note}" for note in notes])
        lines.append("")
    lines.append(f"- summary_hash: `{payload.get('summary_hash', '')}`")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(Path(args.config))
    bars = _safe_ohlcv(config)

    flow = compute_flow_regime_engine(bars)
    crowding, guard = compute_crowding_layer(
        bars,
        timeframe=str((config.get("universe", {}) or {}).get("operational_timeframe", "1h")),
        short_only_enabled=True,
        short_horizon_max="30m",
    )
    mtf = compute_mtf_bias_completion(bars)
    geometry = compute_trade_geometry_layer(bars)
    contract_rows = build_stage46_contract_records(bars)

    modules_contract_compliant = bool(len(contract_rows) == 4)
    oi_short_only_guard_verified = bool(
        guard.get("short_only_enabled", False)
        and (not bool(guard.get("timeframe_allowed", True)))
        and (not bool(guard.get("oi_allowed", True)))
    )

    status = "SUCCESS"
    notes: list[str] = []
    if not modules_contract_compliant:
        status = "PARTIAL"
        notes.append("Not all Part-2 modules produced Stage-44 contract rows.")
    if not oi_short_only_guard_verified:
        status = "PARTIAL"
        notes.append("OI short-only runtime guard verification failed.")

    payload = {
        "stage": "46",
        "status": status,
        "flow_regime_enabled": bool(not flow.empty),
        "crowding_layer_enabled": bool(not crowding.empty),
        "mtf_bias_enabled": bool(not mtf.empty),
        "trade_geometry_enabled": bool(not geometry.empty),
        "oi_short_only_guard_verified": oi_short_only_guard_verified,
        "modules_contract_compliant": modules_contract_compliant,
        "flow_rows": int(flow.shape[0]),
        "crowding_rows": int(crowding.shape[0]),
        "mtf_rows": int(mtf.shape[0]),
        "geometry_rows": int(geometry.shape[0]),
    }
    payload["summary_hash"] = stable_hash(
        {
            "stage": payload["stage"],
            "status": payload["status"],
            "flow_regime_enabled": payload["flow_regime_enabled"],
            "crowding_layer_enabled": payload["crowding_layer_enabled"],
            "mtf_bias_enabled": payload["mtf_bias_enabled"],
            "trade_geometry_enabled": payload["trade_geometry_enabled"],
            "oi_short_only_guard_verified": payload["oi_short_only_guard_verified"],
            "modules_contract_compliant": payload["modules_contract_compliant"],
        },
        length=16,
    )

    summary_path = docs_dir / "stage46_analyst_brain_part2_summary.json"
    report_path = docs_dir / "stage46_analyst_brain_part2_report.md"
    summary_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(_render(payload, notes=notes), encoding="utf-8")

    print(f"status: {payload['status']}")
    print(f"summary_hash: {payload['summary_hash']}")
    print(f"report: {report_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()

