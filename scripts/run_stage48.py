"""Stage-48 tradability learning runner."""

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
from buffmini.stage48.tradability_learning import (
    Stage48Config,
    compute_stage48_labels,
    route_stage_a_stage_b,
    score_candidates_with_ranker,
)
from buffmini.utils.hashing import stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-48 tradability learning")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
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


def _render(payload: dict[str, Any], *, notes: list[str]) -> str:
    lines = [
        "# Stage-48 Tradability Learning Report",
        "",
        f"- status: `{payload.get('status', '')}`",
        f"- strict_direct_survivors_before: `{int(payload.get('strict_direct_survivors_before', 0))}`",
        f"- stage_a_survivors_after: `{int(payload.get('stage_a_survivors_after', 0))}`",
        f"- stage_b_survivors_after: `{int(payload.get('stage_b_survivors_after', 0))}`",
        f"- tradable_rate: `{float(payload.get('tradable_rate', 0.0)):.6f}`",
        f"- net_return_after_cost_mean: `{float(payload.get('net_return_after_cost_mean', 0.0)):.6f}`",
        f"- ranker_enabled: `{bool(payload.get('ranker_enabled', False))}`",
        f"- strongest_bottleneck: `{payload.get('strongest_bottleneck', '')}`",
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
    stage28_run_id = _resolve_stage28_run_id(args, docs_dir)
    if not stage28_run_id:
        raise SystemExit("unable to resolve stage28_run_id for Stage-48")

    shortlist_path = Path(args.runs_dir) / stage28_run_id / "stage47" / "setup_shortlist.csv"
    if not shortlist_path.exists():
        raise SystemExit(f"missing Stage-47 shortlist: {shortlist_path}")
    shortlist = pd.read_csv(shortlist_path)
    finalists_path = Path(args.runs_dir) / stage28_run_id / "stage28" / "finalists_stageC.csv"
    finalists = pd.read_csv(finalists_path) if finalists_path.exists() else pd.DataFrame()
    exp_map = {
        str(row.get("candidate_id", "")): float(row.get("exp_lcb", 0.0))
        for row in finalists.to_dict(orient="records")
    }
    shortlist["exp_lcb_proxy"] = shortlist.get("source_candidate_id", "").astype(str).map(lambda cid: float(exp_map.get(str(cid), 0.0)))

    bars = _safe_ohlcv(config)
    cfg = Stage48Config(
        horizon_bars=12,
        tp_pct=0.004,
        sl_pct=0.003,
        round_trip_cost_pct=float((config.get("costs", {}) or {}).get("round_trip_cost_pct", 0.1)) / 100.0,
        max_adverse_excursion_pct=0.004,
        min_rr=1.2,
        stage_a_threshold=0.42,
        stage_b_threshold=0.0,
    )
    labels = compute_stage48_labels(bars, cfg=cfg)
    ranked = score_candidates_with_ranker(shortlist, labels)
    work = shortlist.merge(ranked[["candidate_id", "rank_score", "replay_worthiness"]], on="candidate_id", how="left")
    work["beam_score"] = pd.to_numeric(work.get("beam_score", 0.0), errors="coerce").fillna(0.0)
    work["beam_score"] = work["beam_score"] + pd.to_numeric(work.get("rank_score", 0.0), errors="coerce").fillna(0.0) * 0.5

    routed = route_stage_a_stage_b(work, labels=labels, cfg=cfg)
    strict_before = int(routed.get("strict_direct_survivors_before", 0))
    stage_a = pd.DataFrame(routed.get("stage_a_survivors", pd.DataFrame()))
    stage_b = pd.DataFrame(routed.get("stage_b_survivors", pd.DataFrame()))
    tradable_rate = float(pd.to_numeric(labels.get("tradable", 0), errors="coerce").fillna(0).astype(int).mean()) if not labels.empty else 0.0
    net_mean = float(pd.to_numeric(labels.get("net_return_after_cost", 0.0), errors="coerce").fillna(0.0).mean()) if not labels.empty else 0.0

    status = "SUCCESS"
    notes: list[str] = []
    if shortlist.empty:
        status = "PARTIAL"
        notes.append("Stage-47 shortlist is empty.")
    if stage_a.empty:
        status = "PARTIAL"
        notes.append("Stage-A produced zero survivors; tradability filter likely too strict.")

    payload = {
        "stage": "48",
        "status": status,
        "strict_direct_survivors_before": strict_before,
        "stage_a_survivors_after": int(stage_a.shape[0]),
        "stage_b_survivors_after": int(stage_b.shape[0]),
        "tradable_rate": tradable_rate,
        "net_return_after_cost_mean": net_mean,
        "ranker_enabled": True,
        "strongest_bottleneck": str(routed.get("strongest_bottleneck", "stage_a_tradability")),
    }
    payload["summary_hash"] = stable_hash(
        {
            "stage": payload["stage"],
            "status": payload["status"],
            "strict_direct_survivors_before": payload["strict_direct_survivors_before"],
            "stage_a_survivors_after": payload["stage_a_survivors_after"],
            "stage_b_survivors_after": payload["stage_b_survivors_after"],
            "tradable_rate": payload["tradable_rate"],
            "net_return_after_cost_mean": payload["net_return_after_cost_mean"],
            "ranker_enabled": payload["ranker_enabled"],
            "strongest_bottleneck": payload["strongest_bottleneck"],
        },
        length=16,
    )

    out_dir = Path(args.runs_dir) / stage28_run_id / "stage48"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out_dir / "stage48_labels.csv", index=False)
    ranked.to_csv(out_dir / "stage48_ranked_candidates.csv", index=False)
    stage_a.to_csv(out_dir / "stage48_stage_a_survivors.csv", index=False)
    stage_b.to_csv(out_dir / "stage48_stage_b_survivors.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    summary_path = docs_dir / "stage48_tradability_learning_summary.json"
    report_path = docs_dir / "stage48_tradability_learning_report.md"
    summary_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(_render(payload, notes=notes), encoding="utf-8")

    print(f"status: {payload['status']}")
    print(f"summary_hash: {payload['summary_hash']}")
    print(f"report: {report_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()

