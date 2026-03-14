"""Transfer diagnostics and matrix utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from buffmini.constants import RAW_DATA_DIR


def discover_transfer_symbols(config: dict[str, Any], *, primary_symbol: str = "") -> list[str]:
    """Discover locally available liquid symbols from config and raw parquet files."""

    configured = [str(v).strip() for v in config.get("universe", {}).get("symbols", []) if str(v).strip()]
    detected: set[str] = set(configured)
    for path in Path(RAW_DATA_DIR).glob("*_1h.parquet"):
        stem = path.stem.replace("_1h", "")
        if "-" in stem:
            base, quote = stem.split("-", 1)
            detected.add(f"{base}/{quote}")
    primary = str(primary_symbol).strip()
    ordered = [symbol for symbol in configured if symbol in detected]
    extras = sorted(symbol for symbol in detected if symbol not in set(ordered))
    symbols = ordered + extras
    if primary and primary in symbols:
        symbols = [primary, *[symbol for symbol in symbols if symbol != primary]]
    return symbols


def classify_transfer_outcome(
    *,
    primary_metrics: dict[str, Any],
    transfer_metrics: dict[str, Any],
    min_trades: int = 8,
) -> dict[str, Any]:
    """Classify transfer behavior and surface useful diagnostics."""

    primary_lcb = float(primary_metrics.get("exp_lcb", 0.0))
    primary_trades = int(primary_metrics.get("trade_count", 0))
    trade_count = int(transfer_metrics.get("trade_count", 0))
    exp_lcb = float(transfer_metrics.get("exp_lcb", 0.0))
    max_dd = float(transfer_metrics.get("maxDD", transfer_metrics.get("max_drawdown", 1.0)))
    diagnostics: list[str] = []
    if trade_count < int(min_trades):
        diagnostics.append("insufficient_trades")
    if trade_count < max(1, int(primary_trades * 0.35)):
        diagnostics.append("trigger_rarity")
    if exp_lcb <= 0.0 and primary_lcb > 0.0:
        diagnostics.append("cost_collapse")
    if exp_lcb <= primary_lcb * 0.25 and trade_count >= int(min_trades):
        diagnostics.append("regime_mismatch")
    if max_dd > 0.30 and trade_count >= int(min_trades):
        diagnostics.append("timing_instability")

    if trade_count >= int(min_trades) and exp_lcb > max(0.0, primary_lcb * 0.65) and max_dd <= 0.25:
        outcome = "transferable"
    elif trade_count >= max(4, int(min_trades * 0.75)) and exp_lcb > 0.0 and max_dd <= 0.30:
        outcome = "partially_transferable"
    elif trade_count >= int(min_trades) and exp_lcb >= -0.0015 and max_dd <= 0.35:
        outcome = "regime_local"
    elif trade_count > 0 and primary_lcb > 0.0:
        outcome = "source_local"
    else:
        outcome = "not_transferable"

    return {
        "classification": outcome,
        "diagnostics": diagnostics,
        "trade_count": trade_count,
        "exp_lcb": exp_lcb,
        "maxDD": max_dd,
        "primary_exp_lcb": primary_lcb,
        "primary_trade_count": primary_trades,
    }
