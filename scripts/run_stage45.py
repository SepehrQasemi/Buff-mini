"""Stage-45 analyst brain core part 1 runner."""

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
from buffmini.stage45.part1 import (
    build_stage45_contract_records,
    compute_htf_bias_skeleton,
    compute_liquidity_map,
    compute_market_structure_engine,
    compute_volatility_regime_engine,
)
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-45 analyst brain part 1")
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


def _render(payload: dict[str, Any], *, partial_notes: list[str]) -> str:
    lines = [
        "# Stage-45 Analyst Brain Part 1 Report",
        "",
        f"- status: `{payload.get('status', '')}`",
        f"- structure_engine_enabled: `{bool(payload.get('structure_engine_enabled', False))}`",
        f"- liquidity_map_enabled: `{bool(payload.get('liquidity_map_enabled', False))}`",
        f"- volatility_regime_enabled: `{bool(payload.get('volatility_regime_enabled', False))}`",
        f"- htf_bias_enabled: `{bool(payload.get('htf_bias_enabled', False))}`",
        f"- modules_contract_compliant: `{bool(payload.get('modules_contract_compliant', False))}`",
        f"- synthetic_tests_passed: `{bool(payload.get('synthetic_tests_passed', False))}`",
        "",
        "## Runtime Evidence",
        f"- structure_rows: `{int(payload.get('structure_rows', 0))}`",
        f"- liquidity_rows: `{int(payload.get('liquidity_rows', 0))}`",
        f"- volatility_rows: `{int(payload.get('volatility_rows', 0))}`",
        f"- htf_rows: `{int(payload.get('htf_rows', 0))}`",
        "",
    ]
    if partial_notes:
        lines.append("## Partial Notes")
        lines.extend([f"- {note}" for note in partial_notes])
        lines.append("")
    lines.append(f"- summary_hash: `{payload.get('summary_hash', '')}`")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(Path(args.config))
    bars = _safe_ohlcv(config)

    structure = compute_market_structure_engine(bars)
    liquidity = compute_liquidity_map(bars)
    volatility = compute_volatility_regime_engine(bars)
    htf = compute_htf_bias_skeleton(bars)
    contract_rows = build_stage45_contract_records(bars)

    # deterministic synthetic check: same input must hash identically
    hash_a = stable_hash(
        {
            "structure": structure.head(32).to_dict(orient="records"),
            "liquidity": liquidity.head(32).to_dict(orient="records"),
            "volatility": volatility.head(32).to_dict(orient="records"),
            "htf": htf.head(32).to_dict(orient="records"),
        },
        length=16,
    )
    hash_b = stable_hash(
        {
            "structure": compute_market_structure_engine(bars).head(32).to_dict(orient="records"),
            "liquidity": compute_liquidity_map(bars).head(32).to_dict(orient="records"),
            "volatility": compute_volatility_regime_engine(bars).head(32).to_dict(orient="records"),
            "htf": compute_htf_bias_skeleton(bars).head(32).to_dict(orient="records"),
        },
        length=16,
    )
    synthetic_tests_passed = bool(hash_a == hash_b)
    modules_contract_compliant = bool(len(contract_rows) == 4)

    partial_notes: list[str] = []
    status = "SUCCESS"
    if not modules_contract_compliant:
        status = "PARTIAL"
        partial_notes.append("Not all Part-1 modules produced Stage-44 compatible rows.")
    if not synthetic_tests_passed:
        status = "PARTIAL"
        partial_notes.append("Deterministic synthetic hash check failed.")

    payload = {
        "stage": "45",
        "status": status,
        "structure_engine_enabled": bool(not structure.empty),
        "liquidity_map_enabled": bool(not liquidity.empty),
        "volatility_regime_enabled": bool(not volatility.empty),
        "htf_bias_enabled": bool(not htf.empty),
        "modules_contract_compliant": modules_contract_compliant,
        "synthetic_tests_passed": synthetic_tests_passed,
        "structure_rows": int(structure.shape[0]),
        "liquidity_rows": int(liquidity.shape[0]),
        "volatility_rows": int(volatility.shape[0]),
        "htf_rows": int(htf.shape[0]),
    }
    payload["summary_hash"] = stable_hash(
        {
            "stage": payload["stage"],
            "status": payload["status"],
            "structure_engine_enabled": payload["structure_engine_enabled"],
            "liquidity_map_enabled": payload["liquidity_map_enabled"],
            "volatility_regime_enabled": payload["volatility_regime_enabled"],
            "htf_bias_enabled": payload["htf_bias_enabled"],
            "modules_contract_compliant": payload["modules_contract_compliant"],
            "synthetic_tests_passed": payload["synthetic_tests_passed"],
        },
        length=16,
    )

    summary_path = docs_dir / "stage45_analyst_brain_part1_summary.json"
    report_path = docs_dir / "stage45_analyst_brain_part1_report.md"
    summary_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(_render(payload, partial_notes=partial_notes), encoding="utf-8")

    print(f"status: {payload['status']}")
    print(f"summary_hash: {payload['summary_hash']}")
    print(f"report: {report_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()

