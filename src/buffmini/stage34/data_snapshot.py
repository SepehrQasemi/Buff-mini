"""Stage-34 fixed snapshot audit and deterministic timeframe completion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.constants import CANONICAL_DATA_DIR, DERIVED_DATA_DIR, RAW_DATA_DIR
from buffmini.data.canonical_raw import prepare_frame, resolve_raw_path
from buffmini.data.derived_tf import (
    canonical_tf_path,
    get_timeframe,
)
from buffmini.stage26.coverage import timeframe_seconds
from buffmini.utils.hashing import stable_hash


REQUIRED_TFS_DEFAULT: tuple[str, ...] = ("1m", "5m", "15m", "30m", "1h", "4h", "1d")


@dataclass(frozen=True)
class SnapshotAuditRow:
    symbol: str
    timeframe: str
    storage_type: str
    path: str
    row_count: int
    start_ts: str | None
    end_ts: str | None
    coverage_years: float
    duplicate_timestamps: int
    non_monotonic: bool
    gaps_detected: int
    max_gap_minutes: int
    missing_bars_estimate: int
    data_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "storage_type": self.storage_type,
            "path": self.path,
            "row_count": int(self.row_count),
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "coverage_years": float(self.coverage_years),
            "duplicate_timestamps": int(self.duplicate_timestamps),
            "non_monotonic": bool(self.non_monotonic),
            "gaps_detected": int(self.gaps_detected),
            "max_gap_minutes": int(self.max_gap_minutes),
            "missing_bars_estimate": int(self.missing_bars_estimate),
            "data_hash": str(self.data_hash),
        }


def audit_and_complete_snapshot(
    *,
    symbols: list[str],
    timeframes: list[str] | None = None,
    exchange: str = "binance",
    raw_dir: Path = RAW_DATA_DIR,
    canonical_dir: Path = CANONICAL_DATA_DIR,
    derived_dir: Path = DERIVED_DATA_DIR,
    drop_incomplete_last: bool = True,
) -> dict[str, Any]:
    tfs = [str(v) for v in (timeframes or list(REQUIRED_TFS_DEFAULT))]
    rows: list[SnapshotAuditRow] = []
    canonical_available: dict[str, list[str]] = {}

    for symbol in [str(v) for v in symbols]:
        _ensure_canonical_1m(
            symbol=symbol,
            exchange=str(exchange),
            raw_dir=Path(raw_dir),
            canonical_dir=Path(canonical_dir),
        )
        per_symbol_available: list[str] = []
        for tf in tfs:
            result = get_timeframe(
                symbol=str(symbol),
                timeframe=str(tf),
                exchange=str(exchange),
                canonical_dir=Path(canonical_dir),
                derived_dir=Path(derived_dir),
                drop_incomplete_last=bool(drop_incomplete_last),
                generator_version="stage34",
            )
            frame = result.frame.copy()
            row = _audit_frame(
                frame=frame,
                symbol=str(symbol),
                timeframe=str(tf),
                storage_type=_storage_type_for_path(result.cache_path, canonical_dir=Path(canonical_dir), derived_dir=Path(derived_dir)),
                path=result.cache_path,
            )
            rows.append(row)
            if row.storage_type == "canonical":
                per_symbol_available.append(str(tf))
        canonical_available[str(symbol)] = sorted(per_symbol_available, key=_tf_sort_key)

    rows_sorted = sorted(rows, key=lambda r: (r.symbol, _tf_sort_key(r.timeframe)))
    end_candidates = [pd.to_datetime(row.end_ts, utc=True, errors="coerce") for row in rows_sorted if row.end_ts]
    end_candidates = [ts for ts in end_candidates if pd.notna(ts)]
    resolved_end_ts = min(end_candidates).isoformat() if end_candidates else None
    payload = {
        "symbols": sorted([str(v) for v in symbols]),
        "required_timeframes": tfs,
        "canonical_available": canonical_available,
        "rows": [row.to_dict() for row in rows_sorted],
        "resolved_end_ts": resolved_end_ts,
    }
    payload["snapshot_hash"] = stable_hash(payload, length=24)
    return payload


def write_snapshot_audit_docs(
    *,
    audit: dict[str, Any],
    md_path: Path,
    json_path: Path,
) -> None:
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(md_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    lines = [
        "# Stage-34 Data Snapshot Audit",
        "",
        f"- snapshot_hash: `{audit.get('snapshot_hash', '')}`",
        f"- resolved_end_ts: `{audit.get('resolved_end_ts', '')}`",
        f"- symbols: `{audit.get('symbols', [])}`",
        f"- required_timeframes: `{audit.get('required_timeframes', [])}`",
        "",
        "| symbol | tf | storage | rows | coverage_years | duplicates | gaps | max_gap_min | missing_est | hash |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in audit.get("rows", []):
        lines.append(
            "| {symbol} | {timeframe} | {storage_type} | {row_count} | {coverage_years:.4f} | {duplicate_timestamps} | {gaps_detected} | {max_gap_minutes} | {missing_bars_estimate} | `{data_hash}` |".format(
                symbol=row.get("symbol", ""),
                timeframe=row.get("timeframe", ""),
                storage_type=row.get("storage_type", ""),
                row_count=int(row.get("row_count", 0)),
                coverage_years=float(row.get("coverage_years", 0.0)),
                duplicate_timestamps=int(row.get("duplicate_timestamps", 0)),
                gaps_detected=int(row.get("gaps_detected", 0)),
                max_gap_minutes=int(row.get("max_gap_minutes", 0)),
                missing_bars_estimate=int(row.get("missing_bars_estimate", 0)),
                data_hash=row.get("data_hash", ""),
            )
        )
    Path(md_path).write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _ensure_canonical_1m(*, symbol: str, exchange: str, raw_dir: Path, canonical_dir: Path) -> None:
    target = canonical_tf_path(symbol=str(symbol), timeframe="1m", exchange=str(exchange), canonical_dir=Path(canonical_dir))
    if target.exists():
        return
    raw = resolve_raw_path(data_dir=Path(raw_dir), exchange=str(exchange), symbol=str(symbol), timeframe="1m")
    if not raw.exists():
        raise FileNotFoundError(f"Missing base raw 1m data for {symbol}: {raw}")
    frame = prepare_frame(pd.read_parquet(raw))
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False, compression="zstd")
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
    meta = {
        "timeframe": "1m",
        "source_timeframe": "1m_raw",
        "exchange": str(exchange),
        "symbol": str(symbol),
        "start_ts": ts.iloc[0].isoformat() if not ts.empty else None,
        "end_ts": ts.iloc[-1].isoformat() if not ts.empty else None,
        "candle_count": int(frame.shape[0]),
        "sha256": stable_hash(frame.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]].to_dict(orient="list"), length=64),
        "generator_version": "stage34",
    }
    target.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8")


def _audit_frame(
    *,
    frame: pd.DataFrame,
    symbol: str,
    timeframe: str,
    storage_type: str,
    path: Path,
) -> SnapshotAuditRow:
    work = frame.copy()
    ts = pd.to_datetime(work.get("timestamp"), utc=True, errors="coerce").dropna().sort_values().reset_index(drop=True)
    if ts.empty:
        return SnapshotAuditRow(
            symbol=str(symbol),
            timeframe=str(timeframe),
            storage_type=str(storage_type),
            path=Path(path).as_posix(),
            row_count=0,
            start_ts=None,
            end_ts=None,
            coverage_years=0.0,
            duplicate_timestamps=0,
            non_monotonic=False,
            gaps_detected=0,
            max_gap_minutes=0,
            missing_bars_estimate=0,
            data_hash=stable_hash({"empty": True, "symbol": symbol, "timeframe": timeframe}, length=24),
        )
    step_seconds = int(timeframe_seconds(str(timeframe)))
    span_seconds = max(0.0, float((ts.iloc[-1] - ts.iloc[0]).total_seconds()))
    span_seconds_inclusive = span_seconds + float(step_seconds)
    expected_bars = int(np.floor(span_seconds / max(1, step_seconds)) + 1)
    observed_bars = int(ts.shape[0])
    duplicate_timestamps = int(ts.duplicated().sum())
    diffs = ts.diff().dropna()
    non_monotonic = bool((diffs < pd.Timedelta(0)).any())
    gap_minutes = (diffs.dt.total_seconds() / 60.0).to_numpy(dtype=float) if not diffs.empty else np.asarray([], dtype=float)
    expected_minutes = float(step_seconds / 60.0)
    gap_mask = gap_minutes > expected_minutes + 1e-9
    gaps_detected = int(np.count_nonzero(gap_mask))
    max_gap_minutes = int(np.max(gap_minutes[gap_mask])) if gaps_detected > 0 else 0
    missing_bars_estimate = int(max(0, expected_bars - observed_bars))
    cols = [c for c in ("timestamp", "open", "high", "low", "close", "volume") if c in work.columns]
    data_hash = stable_hash(work.loc[:, cols].to_dict(orient="list"), length=24) if cols else stable_hash({"rows": int(work.shape[0])}, length=24)
    return SnapshotAuditRow(
        symbol=str(symbol),
        timeframe=str(timeframe),
        storage_type=str(storage_type),
        path=Path(path).as_posix(),
        row_count=int(work.shape[0]),
        start_ts=ts.iloc[0].isoformat(),
        end_ts=ts.iloc[-1].isoformat(),
        coverage_years=float(span_seconds_inclusive / 86400.0 / 365.25),
        duplicate_timestamps=int(duplicate_timestamps),
        non_monotonic=bool(non_monotonic),
        gaps_detected=int(gaps_detected),
        max_gap_minutes=int(max_gap_minutes),
        missing_bars_estimate=int(missing_bars_estimate),
        data_hash=str(data_hash),
    )


def _storage_type_for_path(path: Path, *, canonical_dir: Path, derived_dir: Path) -> str:
    p = Path(path).resolve()
    try:
        p.relative_to(Path(canonical_dir).resolve())
        return "canonical"
    except Exception:
        pass
    try:
        p.relative_to(Path(derived_dir).resolve())
        return "derived"
    except Exception:
        pass
    return "unknown"


def _tf_sort_key(tf: str) -> tuple[int, int]:
    text = str(tf).strip().lower()
    if text == "1m":
        return (0, 1)
    if text.endswith("m"):
        return (1, int(text[:-1]))
    if text.endswith("h"):
        return (2, int(text[:-1]) * 60)
    if text.endswith("d"):
        return (3, int(text[:-1]) * 1440)
    if text.endswith("w"):
        return (4, int(text[:-1]) * 10080)
    if text in {"1mo", "1m0", "1month", "1M".lower()}:
        return (5, 0)
    return (9, 0)

