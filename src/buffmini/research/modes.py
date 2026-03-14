"""Run-mode and interpretation-grade evaluation controls."""

from __future__ import annotations

from copy import deepcopy
from datetime import timezone
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.config import compute_config_hash
from buffmini.constants import DERIVED_DATA_DIR, RAW_DATA_DIR
from buffmini.data.store import build_data_store
from buffmini.utils.hashing import stable_hash


STANDARD_RUN_TYPES: dict[str, dict[str, Any]] = {
    "smoke": {
        "budget_mode_selected": "smoke",
        "evaluation_mode": False,
    },
    "exploration": {
        "budget_mode_selected": "search",
        "evaluation_mode": False,
    },
    "evaluation": {
        "budget_mode_selected": "validate",
        "evaluation_mode": True,
        "seed_bundle": [11, 19, 23, 29, 31],
    },
    "audit": {
        "budget_mode_selected": "full_audit",
        "evaluation_mode": True,
        "seed_bundle": [11, 19, 23, 29, 31, 37, 43],
    },
}


def resolve_run_mode(config: dict[str, Any], requested_mode: str | None = None) -> str:
    configured = str(((config.get("research_run", {}) or {}).get("mode", ""))).strip().lower()
    mode = str(requested_mode or configured or "exploration").strip().lower()
    if mode not in STANDARD_RUN_TYPES:
        raise ValueError(f"Unsupported run mode: {mode}")
    return mode


def infer_resolved_end_ts(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Infer a conservative common end timestamp from locally available market data."""

    universe = dict(config.get("universe", {}))
    symbols = [str(v) for v in universe.get("symbols", []) if str(v).strip()]
    if not symbols:
        symbols = [str(v) for v in ((config.get("research_scope", {}) or {}).get("primary_symbols", ["BTC/USDT"]))]
    timeframes = [str(universe.get("timeframe", "1h")).strip() or "1h"]
    for tf in universe.get("htf_timeframes", []) or []:
        text = str(tf).strip()
        if text and text not in timeframes:
            timeframes.append(text)

    data_cfg = dict(config.get("data", {}))
    store = build_data_store(
        backend=str(data_cfg.get("backend", "parquet")),
        data_dir=RAW_DATA_DIR,
        base_timeframe=str(universe.get("base_timeframe", "1h")),
        resample_source=str(data_cfg.get("resample_source", "direct")),
        derived_dir=DERIVED_DATA_DIR,
        partial_last_bucket=bool(data_cfg.get("partial_last_bucket", False)),
    )
    coverage_rows: list[dict[str, Any]] = []
    end_values: list[pd.Timestamp] = []
    for symbol in symbols:
        for timeframe in timeframes:
            coverage = dict(store.coverage(symbol=symbol, timeframe=timeframe))
            coverage_rows.append(coverage)
            end_text = str(coverage.get("end", "") or "").strip()
            if end_text:
                end_values.append(pd.Timestamp(end_text, tz="UTC") if pd.Timestamp(end_text).tzinfo is None else pd.Timestamp(end_text).tz_convert("UTC"))
    inferred = min(end_values).isoformat() if end_values else ""
    meta = {
        "coverage_rows": coverage_rows,
        "series_count": len(coverage_rows),
        "available_series": int(sum(1 for row in coverage_rows if bool(row.get("exists", False)))),
        "inferred_from_data": bool(inferred),
    }
    return inferred, meta


def build_mode_context(
    config: dict[str, Any],
    *,
    requested_mode: str | None = None,
    auto_pin_resolved_end: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return effective config plus interpretation/evaluation summary."""

    base = deepcopy(config)
    mode = resolve_run_mode(base, requested_mode=requested_mode)
    mode_cfg = dict(STANDARD_RUN_TYPES[mode])
    effective = deepcopy(base)
    effective.setdefault("research_run", {})
    effective["research_run"]["mode"] = mode
    effective["research_run"]["run_type"] = mode
    effective["research_run"]["evaluation_mode"] = bool(mode_cfg.get("evaluation_mode", False))
    effective["budget_mode"] = dict(effective.get("budget_mode", {}))
    effective["budget_mode"]["selected"] = str(mode_cfg.get("budget_mode_selected", effective["budget_mode"].get("selected", "search")))
    effective.setdefault("reproducibility", {})
    effective.setdefault("data", {})
    effective["data"].setdefault("continuity", {})
    effective.setdefault("campaign_memory", {})

    if bool(mode_cfg.get("evaluation_mode", False)):
        effective["reproducibility"]["frozen_research_mode"] = True
        effective["reproducibility"]["require_resolved_end_ts"] = True
        effective["data"]["continuity"]["strict_mode"] = True
        effective["data"]["continuity"]["fail_on_gap"] = True
        effective["campaign_memory"]["cold_start_each_run"] = True
        effective["campaign_memory"]["mode"] = "allocation_only"
        effective["research_run"]["seed_bundle"] = list(mode_cfg.get("seed_bundle", []))

    inferred_end_ts = ""
    inference_meta: dict[str, Any] = {"coverage_rows": [], "series_count": 0, "available_series": 0, "inferred_from_data": False}
    if bool(auto_pin_resolved_end) and bool(effective.get("reproducibility", {}).get("require_resolved_end_ts", False)):
        existing = str(((effective.get("universe", {}) or {}).get("resolved_end_ts", "")) or "").strip()
        if not existing:
            inferred_end_ts, inference_meta = infer_resolved_end_ts(effective)
            if inferred_end_ts:
                effective.setdefault("universe", {})
                effective["universe"]["resolved_end_ts"] = inferred_end_ts

    blockers: list[str] = []
    evaluation_mode = bool(effective.get("research_run", {}).get("evaluation_mode", False))
    if not evaluation_mode:
        blockers.append("EXPLORATION_MODE_NOT_INTERPRETABLE_BY_DEFAULT")
    if evaluation_mode and not bool(effective.get("reproducibility", {}).get("frozen_research_mode", False)):
        blockers.append("FROZEN_RESEARCH_MODE_DISABLED")
    if evaluation_mode and not bool(effective.get("reproducibility", {}).get("require_resolved_end_ts", False)):
        blockers.append("REQUIRE_RESOLVED_END_TS_DISABLED")
    resolved_end_ts = str(((effective.get("universe", {}) or {}).get("resolved_end_ts", "")) or "").strip()
    if evaluation_mode and not resolved_end_ts:
        blockers.append("RESOLVED_END_TS_MISSING")
    continuity_cfg = dict((effective.get("data", {}) or {}).get("continuity", {}))
    if evaluation_mode and not bool(continuity_cfg.get("strict_mode", False)):
        blockers.append("STRICT_CONTINUITY_DISABLED")
    if evaluation_mode and not bool(continuity_cfg.get("fail_on_gap", False)):
        blockers.append("FAIL_ON_GAP_DISABLED")
    if evaluation_mode and not bool((effective.get("campaign_memory", {}) or {}).get("cold_start_each_run", False)):
        blockers.append("CAMPAIGN_MEMORY_NOT_COLD_START")
    if evaluation_mode and not bool((effective.get("reproducibility", {}) or {}).get("deterministic_sorting", False)):
        blockers.append("DETERMINISTIC_SORTING_DISABLED")
    interpretation_allowed = bool(evaluation_mode and not blockers)
    coverage_rows = list(inference_meta.get("coverage_rows", []))
    data_scope_hash = stable_hash(
        {
            "resolved_end_ts": resolved_end_ts,
            "coverage_rows": [
                {
                    "symbol": row.get("symbol"),
                    "timeframe": row.get("timeframe"),
                    "exists": row.get("exists"),
                    "rows": row.get("rows"),
                    "end": row.get("end"),
                }
                for row in coverage_rows
            ],
        },
        length=16,
    )
    summary = {
        "mode": mode,
        "run_type": mode,
        "evaluation_mode": evaluation_mode,
        "interpretation_allowed": interpretation_allowed,
        "validation_state": "EVALUATION_READY" if interpretation_allowed else ("EVALUATION_BLOCKED" if evaluation_mode else "EXPLORATORY_ONLY"),
        "canonical_status": "CANONICAL" if interpretation_allowed else ("EVALUATION_BLOCKED" if evaluation_mode else "EXPLORATORY"),
        "resolved_end_ts": resolved_end_ts,
        "resolved_end_ts_status": "PINNED" if resolved_end_ts else "MISSING",
        "resolved_end_ts_auto_pinned": bool(inferred_end_ts and resolved_end_ts == inferred_end_ts),
        "continuity_status": "STRICT" if bool(continuity_cfg.get("strict_mode", False)) else "RELAXED",
        "data_hash_status": "PRESENT" if data_scope_hash else "MISSING",
        "data_scope_hash": data_scope_hash,
        "config_hash_effective": compute_config_hash(effective),
        "blocked_reasons": blockers,
        "seed_bundle": list((effective.get("research_run", {}) or {}).get("seed_bundle", [])),
        "campaign_memory_cold_start": bool((effective.get("campaign_memory", {}) or {}).get("cold_start_each_run", False)),
        "effective_values": {
            "frozen_research_mode": bool((effective.get("reproducibility", {}) or {}).get("frozen_research_mode", False)),
            "require_resolved_end_ts": bool((effective.get("reproducibility", {}) or {}).get("require_resolved_end_ts", False)),
            "deterministic_sorting": bool((effective.get("reproducibility", {}) or {}).get("deterministic_sorting", False)),
            "strict_continuity": bool(continuity_cfg.get("strict_mode", False)),
            "fail_on_gap": bool(continuity_cfg.get("fail_on_gap", False)),
            "budget_mode_selected": str((effective.get("budget_mode", {}) or {}).get("selected", "")),
        },
        "coverage_rows": coverage_rows,
    }
    summary["summary_hash"] = stable_hash(
        {
            "mode": summary["mode"],
            "interpretation_allowed": summary["interpretation_allowed"],
            "resolved_end_ts": summary["resolved_end_ts"],
            "blocked_reasons": summary["blocked_reasons"],
            "data_scope_hash": summary["data_scope_hash"],
            "config_hash_effective": summary["config_hash_effective"],
        },
        length=16,
    )
    return effective, summary
