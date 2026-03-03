"""Stage-27 coverage gating helpers for deterministic run eligibility."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from buffmini.constants import RAW_DATA_DIR
from buffmini.stage26.coverage import audit_symbol_coverage


@dataclass(frozen=True)
class CoverageGateDecision:
    requested_symbols: list[str]
    used_symbols: list[str]
    disabled_symbols: list[str]
    insufficient_symbols: list[str]
    coverage_years_by_symbol: dict[str, float]
    rows: list[dict[str, Any]]
    required_years: float
    min_years_to_run: float
    fail_if_below_min: bool
    allow_insufficient_data: bool
    auto_btc_fallback: bool
    can_run: bool
    status: str
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_symbols": list(self.requested_symbols),
            "used_symbols": list(self.used_symbols),
            "disabled_symbols": list(self.disabled_symbols),
            "insufficient_symbols": list(self.insufficient_symbols),
            "coverage_years_by_symbol": dict(self.coverage_years_by_symbol),
            "rows": list(self.rows),
            "required_years": float(self.required_years),
            "min_years_to_run": float(self.min_years_to_run),
            "fail_if_below_min": bool(self.fail_if_below_min),
            "allow_insufficient_data": bool(self.allow_insufficient_data),
            "auto_btc_fallback": bool(self.auto_btc_fallback),
            "can_run": bool(self.can_run),
            "status": str(self.status),
            "notes": list(self.notes),
        }


def evaluate_coverage_gate(
    *,
    config: dict[str, Any],
    symbols: list[str],
    timeframe: str = "1m",
    data_dir: Path = RAW_DATA_DIR,
    exchange: str = "binance",
    allow_insufficient_data: bool = False,
    auto_btc_fallback: bool = True,
) -> CoverageGateDecision:
    """Evaluate coverage constraints and optionally apply deterministic BTC-only fallback."""

    requested = [str(item) for item in symbols if str(item)]
    requested_unique = list(dict.fromkeys(requested))
    data_cfg = dict((config.get("data", {}) or {}).get("coverage", {}))
    required_years = float(data_cfg.get("required_years", 4.0))
    min_years_to_run = float(data_cfg.get("min_years_to_run", 1.0))
    fail_if_below_min = bool(data_cfg.get("fail_if_below_min", True))
    rows = [
        audit_symbol_coverage(
            symbol=symbol,
            timeframe=str(timeframe),
            data_dir=data_dir,
            end_mode="latest",
            exchange=str(exchange),
        ).to_dict()
        for symbol in requested_unique
    ]
    coverage_by_symbol = {str(row.get("symbol", "")): float(row.get("coverage_years", 0.0)) for row in rows}
    exists_by_symbol = {str(row.get("symbol", "")): bool(row.get("exists", False)) for row in rows}
    insufficient = [
        symbol
        for symbol in requested_unique
        if (not bool(exists_by_symbol.get(symbol, False))) or float(coverage_by_symbol.get(symbol, 0.0)) < min_years_to_run
    ]

    used_symbols = list(requested_unique)
    disabled: list[str] = []
    notes: list[str] = []

    btc_symbol = "BTC/USDT"
    btc_ok = btc_symbol in requested_unique and btc_symbol not in insufficient
    if bool(auto_btc_fallback) and insufficient:
        fallback_candidates = [symbol for symbol in insufficient if symbol != btc_symbol]
        if fallback_candidates and btc_ok:
            used_symbols = [symbol for symbol in used_symbols if symbol not in fallback_candidates]
            disabled = sorted(fallback_candidates)
            notes.append(f"auto_fallback_disabled:{','.join(disabled)}")

    insuff_after_fallback = [
        symbol
        for symbol in used_symbols
        if (not bool(exists_by_symbol.get(symbol, False))) or float(coverage_by_symbol.get(symbol, 0.0)) < min_years_to_run
    ]

    can_run = True
    status = "OK"
    if not used_symbols:
        can_run = False
        status = "NO_SYMBOLS_AFTER_COVERAGE_GATE"
    elif insuff_after_fallback and bool(fail_if_below_min) and not bool(allow_insufficient_data):
        can_run = False
        status = "INSUFFICIENT_DATA"
        notes.append(f"insufficient_symbols:{','.join(sorted(insuff_after_fallback))}")
    elif insuff_after_fallback and bool(allow_insufficient_data):
        status = "ALLOW_INSUFFICIENT_DATA"
        notes.append(f"allow_insufficient_data_symbols:{','.join(sorted(insuff_after_fallback))}")
    elif insuff_after_fallback:
        status = "INSUFFICIENT_DATA_IGNORED"
        notes.append(f"insufficient_symbols_ignored:{','.join(sorted(insuff_after_fallback))}")

    if any(float(coverage_by_symbol.get(symbol, 0.0)) < required_years for symbol in used_symbols):
        notes.append("required_years_not_met_for_all_used_symbols")

    return CoverageGateDecision(
        requested_symbols=requested_unique,
        used_symbols=used_symbols,
        disabled_symbols=disabled,
        insufficient_symbols=sorted(set(insufficient)),
        coverage_years_by_symbol=coverage_by_symbol,
        rows=rows,
        required_years=float(required_years),
        min_years_to_run=float(min_years_to_run),
        fail_if_below_min=bool(fail_if_below_min),
        allow_insufficient_data=bool(allow_insufficient_data),
        auto_btc_fallback=bool(auto_btc_fallback),
        can_run=bool(can_run),
        status=str(status),
        notes=notes,
    )

