"""Stage-40 tradability objective redesign runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.config import load_config
from buffmini.constants import RAW_DATA_DIR, RUNS_DIR
from buffmini.data.store import build_data_store
from buffmini.stage40.objective import TradabilityConfig, compute_tradability_labels, route_two_stage_objective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-40 tradability objective redesign")
    parser.add_argument("--config", type=Path, default=Path("configs") / "default.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--stage28-run-id", type=str, default="")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _resolve_stage28_run_id(args: argparse.Namespace, docs_dir: Path) -> str:
    direct = str(args.stage28_run_id).strip()
    if direct:
        return direct
    stage39 = _load_json(docs_dir / "stage39_signal_generation_summary.json")
    if str(stage39.get("stage28_run_id", "")).strip():
        return str(stage39["stage28_run_id"]).strip()
    stage38 = _load_json(docs_dir / "stage38_master_summary.json")
    return str(stage38.get("stage28_run_id", "")).strip()


def _render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage-40 Tradability Objective Report",
        "",
        "## Objective Shift",
        "- Stage-A objective targets activation + tradability before robustness.",
        "- Stage-B objective keeps strict robustness filtering.",
        "",
        "## Candidate Survival",
        f"- input_candidates: `{int(payload.get('counts', {}).get('input', 0))}`",
        f"- stage_a_survivors: `{int(payload.get('counts', {}).get('stage_a', 0))}`",
        f"- stage_b_survivors: `{int(payload.get('counts', {}).get('stage_b', 0))}`",
        f"- before_strict_direct_survivors: `{int(payload.get('counts', {}).get('before_strict_direct', 0))}`",
        "",
        "## Label Stats",
        f"- tradable_rate: `{float(payload.get('label_stats', {}).get('tradable_rate', 0.0)):.6f}`",
        f"- tp_before_sl_rate: `{float(payload.get('label_stats', {}).get('tp_before_sl_rate', 0.0)):.6f}`",
        f"- net_return_after_cost_mean: `{float(payload.get('label_stats', {}).get('net_return_after_cost_mean', 0.0)):.6f}`",
        "",
        "## Bottleneck",
        f"- strongest_bottleneck_step: `{payload.get('strongest_bottleneck_step', '')}`",
        "",
        "## Before vs After",
        f"- before (strict-direct): `{int(payload.get('counts', {}).get('before_strict_direct', 0))}`",
        f"- after (two-stage, final): `{int(payload.get('counts', {}).get('stage_b', 0))}`",
    ]
    return "\n".join(lines).strip() + "\n"


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
    bars = store.load_ohlcv(symbol=symbol, timeframe=timeframe).tail(600).reset_index(drop=True)
    if not bars.empty:
        return bars[["timestamp", "open", "high", "low", "close", "volume"]].copy()

    idx = pd.date_range("2025-01-01T00:00:00Z", periods=600, freq="1h", tz="UTC")
    base = 100.0 + np.cumsum(np.sin(np.arange(600) / 24.0) * 0.1)
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


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(Path(args.config))
    stage28_run_id = _resolve_stage28_run_id(args, docs_dir)
    if not stage28_run_id:
        raise SystemExit("unable to resolve stage28_run_id for Stage-40")

    stage39_summary = _load_json(docs_dir / "stage39_signal_generation_summary.json")
    stage39_dir = Path(args.runs_dir) / stage28_run_id / "stage39"
    layer_c_path = stage39_dir / "layer_c_candidates.csv"
    if not layer_c_path.exists():
        raise SystemExit(f"missing Stage-39 shortlist file: {layer_c_path}")
    layer_c = pd.read_csv(layer_c_path)

    finalists = pd.read_csv(Path(args.runs_dir) / stage28_run_id / "stage28" / "finalists_stageC.csv")
    exp_map = {
        str(row["candidate_id"]): float(row.get("exp_lcb", 0.0))
        for row in finalists.to_dict(orient="records")
    }
    layer_c["exp_lcb_proxy"] = layer_c.get("source_candidate_id", "").astype(str).map(lambda cid: float(exp_map.get(str(cid), 0.0)))

    bars = _safe_ohlcv(config)
    trad_cfg = TradabilityConfig(
        horizon_bars=12,
        tp_pct=0.004,
        sl_pct=0.003,
        round_trip_cost_pct=float((config.get("costs", {}) or {}).get("round_trip_cost_pct", 0.1)) / 100.0,
        max_adverse_excursion_pct=0.004,
        stage_a_threshold=0.35,
        stage_b_threshold=0.0,
    )
    labels = compute_tradability_labels(bars, cfg=trad_cfg)
    routed = route_two_stage_objective(layer_c, labels=labels, cfg=trad_cfg)
    counts = dict(routed.get("counts", {}))
    label_stats = dict(routed.get("label_stats", {}))

    payload = {
        "stage": "40",
        "seed": int(args.seed),
        "stage28_run_id": stage28_run_id,
        "stage39_raw_candidate_count": int(stage39_summary.get("raw_candidate_count", 0)),
        "stage39_shortlisted_count": int(stage39_summary.get("shortlisted_count", 0)),
        "counts": counts,
        "label_stats": label_stats,
        "strongest_bottleneck_step": str(routed.get("bottleneck_step", "stage_a_activation")),
    }

    out_dir = Path(args.runs_dir) / stage28_run_id / "stage40"
    out_dir.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out_dir / "tradability_labels.csv", index=False)
    pd.DataFrame(routed.get("stage_a_survivors", pd.DataFrame())).to_csv(out_dir / "stage_a_survivors.csv", index=False)
    pd.DataFrame(routed.get("stage_b_survivors", pd.DataFrame())).to_csv(out_dir / "stage_b_survivors.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    report_md = docs_dir / "stage40_tradability_objective_report.md"
    report_json = docs_dir / "stage40_tradability_objective_summary.json"
    report_md.write_text(_render_report(payload), encoding="utf-8")
    report_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    print(f"stage28_run_id: {stage28_run_id}")
    print(f"stage_a_survivors: {counts.get('stage_a', 0)}")
    print(f"stage_b_survivors: {counts.get('stage_b', 0)}")
    print(f"report_md: {report_md}")
    print(f"report_json: {report_json}")


if __name__ == "__main__":
    main()
