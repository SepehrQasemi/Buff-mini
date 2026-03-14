"""Run Stage-79 candidate-specific ranking and pre-edge diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RAW_DATA_DIR, RUNS_DIR
from buffmini.data.store import build_data_store
from buffmini.stage48.tradability_learning import Stage48Config, compute_stage48_labels, score_candidates_with_ranker
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-79 candidate-specific ranking diagnostics")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--stage28-run-id", type=str, default="")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _resolve_stage28_run_id(args: argparse.Namespace, docs_dir: Path) -> str:
    if str(args.stage28_run_id).strip():
        return str(args.stage28_run_id).strip()
    stage47 = _load_json(docs_dir / "stage47_signal_gen2_summary.json")
    if str(stage47.get("stage28_run_id", "")).strip():
        return str(stage47["stage28_run_id"]).strip()
    stage39 = _load_json(docs_dir / "stage39_signal_generation_summary.json")
    return str(stage39.get("stage28_run_id", "")).strip()


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


def _render(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage-79 Report",
        "",
        f"- status: `{summary['status']}`",
        f"- execution_status: `{summary['execution_status']}`",
        f"- stage_role: `{summary['stage_role']}`",
        f"- validation_state: `{summary['validation_state']}`",
        f"- candidate_count: `{summary['candidate_count']}`",
        f"- promising_count: `{summary['promising_count']}`",
        f"- validated_count: `{summary['validated_count']}`",
        f"- fingerprint_diversity: `{summary['fingerprint_diversity']}`",
        f"- mean_aggregate_risk: `{summary['mean_aggregate_risk']}`",
        "",
        "## Candidate Classes",
    ]
    for key, value in sorted((summary.get("candidate_class_counts") or {}).items()):
        lines.append(f"- {key}: `{int(value)}`")
    lines.extend(["", "## Mean Risk Card"])
    for key, value in sorted((summary.get("mean_risk_card") or {}).items()):
        lines.append(f"- {key}: `{float(value):.6f}`")
    lines.extend(["", f"- summary_hash: `{summary['summary_hash']}`"])
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(Path(args.config))
    stage28_run_id = _resolve_stage28_run_id(args, docs_dir)
    if not stage28_run_id:
        raise SystemExit("unable to resolve stage28_run_id for Stage-79")

    shortlist_path = Path(args.runs_dir) / stage28_run_id / "stage47" / "setup_shortlist.csv"
    if not shortlist_path.exists():
        raise SystemExit(f"missing Stage-47 shortlist: {shortlist_path}")
    shortlist = pd.read_csv(shortlist_path)
    bars = _safe_ohlcv(config)
    labels = compute_stage48_labels(bars, cfg=Stage48Config(round_trip_cost_pct=float((config.get("costs", {}) or {}).get("round_trip_cost_pct", 0.1)) / 100.0))
    ranked = score_candidates_with_ranker(shortlist, labels, market_frame=bars)
    out_path = docs_dir / "stage79_ranking_cards.csv"
    ranked.to_csv(out_path, index=False)

    class_counts = {str(key): int(value) for key, value in ranked.get("candidate_class", pd.Series(dtype=str)).astype(str).value_counts(dropna=False).to_dict().items()}
    risk_cols = [
        "trade_density_risk",
        "cost_fragility_risk",
        "regime_concentration_risk",
        "hold_sanity_risk",
        "overlap_duplication_risk",
        "clustering_risk",
        "thin_evidence_risk",
        "transfer_risk_prior",
        "aggregate_risk",
    ]
    mean_risk_card = {
        column: float(round(pd.to_numeric(ranked.get(column, 0.0), errors="coerce").fillna(0.0).mean(), 6))
        for column in risk_cols
        if column in ranked.columns
    }
    promising_count = int(class_counts.get("promising_but_unproven", 0))
    validated_count = int(class_counts.get("validated_candidate", 0))
    fingerprint_diversity = (
        float(round(ranked["behavioral_fingerprint"].astype(str).nunique() / max(1, len(ranked)), 6))
        if not ranked.empty and "behavioral_fingerprint" in ranked.columns
        else 0.0
    )
    status = "SUCCESS" if promising_count > 0 and fingerprint_diversity >= 0.70 else "PARTIAL"
    summary = {
        "stage": "79",
        "status": status,
        "execution_status": "EXECUTED",
        "stage_role": "heuristic_filter",
        "validation_state": "PROMISING_CANDIDATES_SURFACED" if promising_count > 0 else "RANKING_TOO_BLUNT",
        "candidate_count": int(len(ranked)),
        "candidate_class_counts": class_counts,
        "promising_count": int(promising_count),
        "validated_count": int(validated_count),
        "fingerprint_diversity": float(fingerprint_diversity),
        "mean_aggregate_risk": float(mean_risk_card.get("aggregate_risk", 0.0)),
        "mean_risk_card": mean_risk_card,
        "artifact_path": str(out_path).replace("\\", "/"),
    }
    summary["summary_hash"] = stable_hash(summary, length=16)
    (docs_dir / "stage79_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (docs_dir / "stage79_report.md").write_text(_render(summary), encoding="utf-8")
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
