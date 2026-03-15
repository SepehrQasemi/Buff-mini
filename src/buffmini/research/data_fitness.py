"""Data fitness and canonical comparison diagnostics."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.constants import PROJECT_ROOT
from buffmini.data.snapshot import snapshot_metadata_from_config
from buffmini.research.modes import build_mode_context
from buffmini.validation.candidate_runtime import load_candidate_market_frame
from buffmini.validation.walkforward_v2 import build_windows


def evaluate_data_fitness(
    config: dict[str, Any],
    *,
    symbols: list[str],
    timeframes: list[str],
) -> dict[str, Any]:
    snapshot_payload = _load_snapshot_payload(config)
    rows: list[dict[str, Any]] = []
    for symbol in symbols:
        for timeframe in timeframes:
            live_exploration_cfg, live_exploration_mode = build_mode_context(
                config,
                requested_mode="exploration",
                auto_pin_resolved_end=False,
            )
            live_strict_cfg, live_strict_mode = build_mode_context(
                config,
                requested_mode="evaluation",
                auto_pin_resolved_end=True,
            )
            canonical_cfg, canonical_mode = build_mode_context(
                _canonical_config(config, snapshot_payload, symbol=symbol, timeframe=timeframe),
                requested_mode="evaluation",
                auto_pin_resolved_end=False,
            )

            _, live_relaxed_meta = load_candidate_market_frame(live_exploration_cfg, symbol=symbol, timeframe=timeframe)
            live_eval_frame, live_strict_meta = load_candidate_market_frame(live_strict_cfg, symbol=symbol, timeframe=timeframe)
            canonical_frame, canonical_meta = load_candidate_market_frame(canonical_cfg, symbol=symbol, timeframe=timeframe)
            snapshot_row = _snapshot_row(snapshot_payload, symbol=symbol, timeframe=timeframe)

            rows.append(
                {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "live_relaxed_usable": bool(not live_relaxed_meta.get("runtime_truth_blocked", False)),
                    "live_relaxed_gap_count": int((live_relaxed_meta.get("continuity_report") or {}).get("gap_count", 0)),
                    "live_relaxed_largest_gap_bars": int((live_relaxed_meta.get("continuity_report") or {}).get("largest_gap_bars", 0)),
                    "live_strict_usable": bool(not live_strict_meta.get("runtime_truth_blocked", False) and not live_strict_meta.get("continuity_blocked", False)),
                    "live_strict_gap_count": int((live_strict_meta.get("continuity_report") or {}).get("gap_count", 0)),
                    "live_strict_largest_gap_bars": int((live_strict_meta.get("continuity_report") or {}).get("largest_gap_bars", 0)),
                    "live_strict_reason": str(live_strict_meta.get("continuity_reason", "") or live_strict_meta.get("runtime_truth_reason", "")),
                    "canonical_usable": bool(not canonical_meta.get("runtime_truth_blocked", False) and not canonical_meta.get("continuity_blocked", False)),
                    "canonical_gap_count": int((canonical_meta.get("continuity_report") or {}).get("gap_count", 0)),
                    "canonical_largest_gap_bars": int((canonical_meta.get("continuity_report") or {}).get("largest_gap_bars", 0)),
                    "canonical_reason": str(canonical_meta.get("continuity_reason", "") or canonical_meta.get("runtime_truth_reason", "")),
                    "live_evaluation_windows": int(_possible_window_count(live_eval_frame, config)),
                    "canonical_evaluation_windows": int(_possible_window_count(canonical_frame, config)),
                    "snapshot_available": bool(snapshot_row),
                    "snapshot_end_ts": str(snapshot_row.get("end_ts", "")),
                    "snapshot_candle_count": int(snapshot_row.get("candle_count", 0)),
                    "canonical_row_count": int(canonical_meta.get("row_count", 0)),
                    "canonical_end_ts": _frame_end_iso(canonical_frame),
                    "canonical_snapshot_match": bool(
                        snapshot_row
                        and int(snapshot_row.get("candle_count", 0)) == int(canonical_meta.get("row_count", 0))
                        and str(snapshot_row.get("end_ts", "")) == _frame_end_iso(canonical_frame)
                    ),
                    "evaluation_usable_class": _evaluation_usable_class(
                        live_strict_meta=live_strict_meta,
                        canonical_meta=canonical_meta,
                    ),
                    "mode_labels": {
                        "live_relaxed": str(live_exploration_mode.get("mode", "")),
                        "live_strict": str(live_strict_mode.get("mode", "")),
                        "canonical": str(canonical_mode.get("mode", "")),
                    },
                }
            )
    return {
        "rows": rows,
        "snapshot_meta": snapshot_metadata_from_config(config),
        "summary_hash": _stable_summary_hash(rows),
    }


def build_campaign_comparison(
    *,
    stage85_summary: dict[str, Any],
    stage89_summary: dict[str, Any],
) -> list[dict[str, Any]]:
    environments = dict((stage85_summary.get("environments") or {}))
    data_rows = list((stage89_summary.get("rows") or []))
    comparison: list[dict[str, Any]] = []
    env_map = {
        "live_relaxed": "live_relaxed",
        "live_strict": "live_strict",
        "canonical_snapshot": "canonical",
    }
    data_lookup = {
        (str(row.get("symbol", "")), str(row.get("timeframe", ""))): row
        for row in data_rows
    }
    for environment, _data_key in env_map.items():
        stage85 = dict(environments.get(environment, {}))
        representative = data_lookup.get(("BTC/USDT", "1h"), {})
        comparison.append(
            {
                "environment": environment,
                "candidate_count": int(stage85.get("candidate_count", 0)),
                "promising_count": int(stage85.get("promising_count", 0)),
                "validated_count": int(stage85.get("validated_count", 0)),
                "robust_count": int(stage85.get("robust_count", 0)),
                "blocked_count": int(stage85.get("blocked_count", 0)),
                "dominant_failure_reasons": dict(stage85.get("dominant_failure_reasons", {})),
                "data_gate_status": representative.get(
                    "evaluation_usable_class",
                    "unknown",
                ) if environment in {"live_strict", "canonical_snapshot"} else "exploration_usable",
                "data_gate_reason": representative.get(
                    "canonical_reason" if environment == "canonical_snapshot" else "live_strict_reason",
                    "",
                ),
            }
        )
    return comparison


def _canonical_config(
    config: dict[str, Any],
    snapshot_payload: dict[str, Any],
    *,
    symbol: str,
    timeframe: str,
) -> dict[str, Any]:
    cfg = deepcopy(config)
    cfg.setdefault("research_run", {})["data_source"] = "canonical"
    snapshot_row = _snapshot_row(snapshot_payload, symbol=symbol, timeframe=timeframe)
    if snapshot_row:
        cfg.setdefault("universe", {})["resolved_end_ts"] = str(snapshot_row.get("end_ts", ""))
    return cfg


def _snapshot_row(payload: dict[str, Any], *, symbol: str, timeframe: str) -> dict[str, Any]:
    per_symbol = dict(payload.get("per_symbol_per_tf", {}))
    tf_map = dict(per_symbol.get(str(symbol), {}))
    row = dict(tf_map.get(str(timeframe), {}))
    return row


def _load_snapshot_payload(config: dict[str, Any]) -> dict[str, Any]:
    snapshot_cfg = dict((config.get("data", {}) or {}).get("snapshot", {}))
    path = Path(snapshot_cfg.get("path", PROJECT_ROOT / "data" / "snapshots" / "DATA_FROZEN_v1.json"))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _possible_window_count(frame: pd.DataFrame, config: dict[str, Any]) -> int:
    if frame is None or frame.empty or "timestamp" not in frame.columns:
        return 0
    wf_cfg = dict((((config.get("evaluation", {}) or {}).get("stage8", {}) or {}).get("walkforward_v2", {})))
    windows = build_windows(
        start_ts=pd.to_datetime(frame["timestamp"].min(), utc=True),
        end_ts=pd.to_datetime(frame["timestamp"].max(), utc=True),
        train_days=int(max(30, wf_cfg.get("train_days", 180))),
        holdout_days=int(max(7, wf_cfg.get("holdout_days", 30))),
        forward_days=int(max(7, wf_cfg.get("forward_days", 30))),
        step_days=int(max(7, wf_cfg.get("step_days", 30))),
        reserve_tail_days=int(max(0, wf_cfg.get("reserve_tail_days", 0))),
    )
    return int(len(windows))


def _frame_end_iso(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty or "timestamp" not in frame.columns:
        return ""
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
    return str(ts.max().isoformat()) if not ts.empty else ""


def _evaluation_usable_class(*, live_strict_meta: dict[str, Any], canonical_meta: dict[str, Any]) -> str:
    if not bool(live_strict_meta.get("runtime_truth_blocked", False)) and not bool(live_strict_meta.get("continuity_blocked", False)):
        return "evaluation_usable_live"
    if not bool(canonical_meta.get("runtime_truth_blocked", False)) and not bool(canonical_meta.get("continuity_blocked", False)):
        return "evaluation_usable_canonical_only"
    return "evaluation_blocked"


def _stable_summary_hash(rows: list[dict[str, Any]]) -> str:
    payload = [
        {
            "symbol": str(row.get("symbol", "")),
            "timeframe": str(row.get("timeframe", "")),
            "live_strict_usable": bool(row.get("live_strict_usable", False)),
            "canonical_usable": bool(row.get("canonical_usable", False)),
            "evaluation_usable_class": str(row.get("evaluation_usable_class", "")),
        }
        for row in rows
    ]
    from buffmini.utils.hashing import stable_hash

    return stable_hash(payload, length=16)
