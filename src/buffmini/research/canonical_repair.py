"""Stage-96 canonical evaluation repair helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.constants import CANONICAL_DATA_DIR, CANONICAL_EVAL_DATA_DIR, DERIVED_DATA_DIR, PROJECT_ROOT
from buffmini.data.continuity import continuity_report
from buffmini.data.derived_tf import canonical_meta_path, canonical_tf_path, get_timeframe
from buffmini.data.snapshot import build_snapshot_payload, write_snapshot_file
from buffmini.research.data_fitness import evaluate_data_fitness
from buffmini.utils.hashing import stable_hash


def repair_canonical_evaluation_data(
    config: dict[str, Any],
    *,
    symbols: list[str],
    timeframes: list[str],
    exchange: str = "binance",
    target_snapshot_id: str = "DATA_FROZEN_EVAL_v2",
) -> dict[str, Any]:
    repair_rows: list[dict[str, Any]] = []
    eval_dir = CANONICAL_EVAL_DATA_DIR / str(exchange).strip().lower()
    eval_dir.mkdir(parents=True, exist_ok=True)

    for symbol in symbols:
        for timeframe in timeframes:
            loaded = get_timeframe(
                symbol=str(symbol),
                timeframe=str(timeframe),
                exchange=str(exchange),
                canonical_dir=CANONICAL_DATA_DIR,
                derived_dir=DERIVED_DATA_DIR,
            )
            source_frame = loaded.frame.copy()
            repaired_frame, trim_meta = build_contiguous_evaluation_suffix(source_frame, timeframe=str(timeframe))
            before_report = continuity_report(source_frame, timeframe=str(timeframe), max_gap_bars=0)
            after_report = continuity_report(repaired_frame, timeframe=str(timeframe), max_gap_bars=0)
            out_path = canonical_tf_path(
                symbol=str(symbol),
                timeframe=str(timeframe),
                exchange=str(exchange),
                canonical_dir=CANONICAL_EVAL_DATA_DIR,
            )
            out_meta = canonical_meta_path(
                symbol=str(symbol),
                timeframe=str(timeframe),
                exchange=str(exchange),
                canonical_dir=CANONICAL_EVAL_DATA_DIR,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            repaired_frame.to_parquet(out_path, index=False, compression="zstd")
            payload = {
                "symbol": str(symbol),
                "timeframe": str(timeframe),
                "exchange": str(exchange),
                "source_hash": stable_hash(source_frame[["timestamp", "open", "high", "low", "close", "volume"]].to_dict(orient="list"), length=24),
                "target_hash": stable_hash(repaired_frame[["timestamp", "open", "high", "low", "close", "volume"]].to_dict(orient="list"), length=24),
                "sha256": stable_hash(repaired_frame[["timestamp", "open", "high", "low", "close", "volume"]].to_dict(orient="list"), length=64),
                "repair_mode": "contiguous_suffix_after_last_gap",
                "trim_start_ts": str(trim_meta.get("trim_start_ts", "")),
                "source_rows": int(len(source_frame)),
                "rows": int(len(repaired_frame)),
                "candle_count": int(len(repaired_frame)),
                "start_ts": _frame_start(repaired_frame),
                "end_ts": _frame_end(repaired_frame),
                "before_continuity": before_report,
                "after_continuity": after_report,
            }
            out_meta.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
            repair_rows.append(
                {
                    "symbol": str(symbol),
                    "timeframe": str(timeframe),
                    "source_rows": int(len(source_frame)),
                    "repaired_rows": int(len(repaired_frame)),
                    "trim_start_ts": str(trim_meta.get("trim_start_ts", "")),
                    "before_gap_count": int(before_report.get("gap_count", 0)),
                    "before_largest_gap_bars": int(before_report.get("largest_gap_bars", 0)),
                    "after_gap_count": int(after_report.get("gap_count", 0)),
                    "after_largest_gap_bars": int(after_report.get("largest_gap_bars", 0)),
                    "strict_usable_after": bool(after_report.get("passes_strict", False)),
                }
            )

    snapshot_path = PROJECT_ROOT / "data" / "snapshots" / f"{target_snapshot_id}.json"
    snapshot_payload = build_snapshot_payload(
        snapshot_id=str(target_snapshot_id),
        symbols=list(symbols),
        timeframes=list(timeframes),
        exchange=str(exchange),
        canonical_dir=CANONICAL_EVAL_DATA_DIR,
    )
    write_snapshot_file(snapshot_path, snapshot_payload)

    repaired_config = _evaluation_repair_config(config, snapshot_id=str(target_snapshot_id), snapshot_path=snapshot_path)
    fitness_after = evaluate_data_fitness(repaired_config, symbols=list(symbols), timeframes=list(timeframes))
    critical_rows = [row for row in repair_rows if str(row.get("symbol")) in {"BTC/USDT", "ETH/USDT"} and str(row.get("timeframe")) in {"1h", "4h"}]
    stage96d_required = not all(bool(row.get("strict_usable_after", False)) for row in critical_rows)
    return {
        "repair_rows": repair_rows,
        "snapshot_path": str(snapshot_path.relative_to(PROJECT_ROOT).as_posix()),
        "snapshot_id": str(target_snapshot_id),
        "snapshot_hash": str(snapshot_payload.get("snapshot_hash", "")),
        "fitness_after": fitness_after,
        "stage96d_required": bool(stage96d_required),
        "stage96d_reason": "" if not stage96d_required else "target_scope_still_not_strict_usable",
        "summary_hash": stable_hash(
            {
                "repair_rows": repair_rows,
                "snapshot_id": target_snapshot_id,
                "stage96d_required": stage96d_required,
            },
            length=16,
        ),
    }


def build_contiguous_evaluation_suffix(frame: pd.DataFrame, *, timeframe: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    work = frame.copy()
    work["timestamp"] = pd.to_datetime(work.get("timestamp"), utc=True, errors="coerce")
    work = work.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
    report = continuity_report(work, timeframe=str(timeframe), max_gap_bars=0)
    gaps = list(report.get("gaps", []))
    if not gaps:
        return work.reset_index(drop=True), {"trim_start_ts": _frame_start(work), "repair_mode": "already_contiguous"}
    last_gap_end = str(gaps[-1].get("end_ts", ""))
    trimmed = work.loc[work["timestamp"] >= pd.to_datetime(last_gap_end, utc=True, errors="coerce")].copy().reset_index(drop=True)
    return trimmed, {"trim_start_ts": last_gap_end, "repair_mode": "contiguous_suffix_after_last_gap"}


def _evaluation_repair_config(config: dict[str, Any], *, snapshot_id: str, snapshot_path: Path) -> dict[str, Any]:
    repaired = json.loads(json.dumps(config))
    repaired.setdefault("research_run", {})["evaluation_data_source"] = "canonical_eval"
    repaired.setdefault("data", {}).setdefault("snapshot", {})["evaluation_id"] = str(snapshot_id)
    repaired["data"]["snapshot"]["evaluation_path"] = str(snapshot_path.relative_to(PROJECT_ROOT).as_posix())
    return repaired


def _frame_start(frame: pd.DataFrame) -> str:
    ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce").dropna()
    return str(ts.min().isoformat()) if not ts.empty else ""


def _frame_end(frame: pd.DataFrame) -> str:
    ts = pd.to_datetime(frame.get("timestamp"), utc=True, errors="coerce").dropna()
    return str(ts.max().isoformat()) if not ts.empty else ""
