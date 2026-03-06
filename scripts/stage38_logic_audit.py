"""Stage-38.2 hunt-vs-engine contradiction audit with lineage table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from buffmini.config import load_config
from buffmini.constants import RAW_DATA_DIR, RUNS_DIR
from buffmini.data.features import calculate_features
from buffmini.data.store import build_data_store
from buffmini.stage38.audit import build_lineage_table_from_stage28, oi_runtime_usage
from buffmini.stage38.reporting import render_stage38_logic_audit_report
from buffmini.stage9.oi_overlay import OI_DEPENDENT_COLUMNS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-38 hunt-vs-engine lineage audit")
    parser.add_argument("--stage28-run-id", type=str, required=True)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--docs-dir", type=Path, default=Path("docs"))
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--quality-floor", type=float, default=-0.02)
    parser.add_argument("--activation-summary", type=Path, default=Path("docs") / "stage37_activation_hunt_summary.json")
    parser.add_argument("--registry-path", type=Path, default=Path(""))
    parser.add_argument("--config", type=Path, default=Path("configs") / "default.yaml")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    payload = json.loads(p.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    stage28_dir = Path(args.runs_dir) / str(args.stage28_run_id) / "stage28"
    if not stage28_dir.exists():
        raise SystemExit(f"missing stage28 directory: {stage28_dir}")

    activation_payload = _load_json(Path(args.activation_summary))
    cfg = load_config(Path(args.config))
    lineage = build_lineage_table_from_stage28(
        stage28_dir=stage28_dir,
        threshold=float(args.threshold),
        quality_floor=float(args.quality_floor),
    )
    root_cause = (
        "Activation hunt counted NaN active-candidate cells as non-empty strings ('nan'), "
        "inflating raw_signal_count while engine final_signal stayed zero."
    )
    fix_summary = (
        "Normalized active_candidates now maps NaN/None/'nan' to empty before raw-signal gating, "
        "and lineage now tracks composer_signal_count explicitly."
    )
    oi_usage_payload = _runtime_oi_usage(cfg=cfg)
    payload = {
        "stage": "38.2",
        "stage28_run_id": str(args.stage28_run_id),
        "lineage_table": lineage,
        "collapse_reason": str(lineage.get("collapse_reason", "")),
        "contradiction_fixed": bool(lineage.get("contradiction_fixed", False)),
        "root_cause": root_cause,
        "fix_summary": fix_summary,
        "before_after": [
            {
                "metric": "raw_signal_count",
                "before": float(lineage.get("legacy_raw_signal_count", 0)),
                "after": float(lineage.get("raw_signal_count", 0)),
            },
            {
                "metric": "composer_vs_engine_delta",
                "before": float(lineage.get("legacy_raw_signal_count", 0) - lineage.get("engine_raw_signal_count", 0)),
                "after": float(lineage.get("composer_signal_count", 0) - lineage.get("engine_raw_signal_count", 0)),
            },
        ],
        "activation_summary_source": str(Path(args.activation_summary).as_posix()),
        "activation_summary_present": bool(activation_payload),
        "self_learning": {
            "registry_path": str(args.registry_path.as_posix()) if str(args.registry_path).strip() else "",
            "registry_rows": 0,
            "elites_count": 0,
            "dead_family_count": 0,
            "failure_motif_tags_non_empty": False,
        },
        "oi_usage": oi_usage_payload,
    }

    report_md = docs_dir / "stage38_logic_audit_report.md"
    report_json = docs_dir / "stage38_logic_audit_summary.json"
    report_md.write_text(render_stage38_logic_audit_report(payload), encoding="utf-8")
    report_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    print(f"stage28_run_id: {args.stage28_run_id}")
    print(f"collapse_reason: {payload['collapse_reason']}")
    print(f"contradiction_fixed: {payload['contradiction_fixed']}")
    print(f"report_md: {report_md}")
    print(f"report_json: {report_json}")


def _runtime_oi_usage(*, cfg: dict[str, Any]) -> dict[str, Any]:
    data_cfg = dict(cfg.get("data", {}))
    futures_cfg = dict(data_cfg.get("futures_extras", {}))
    oi_cfg = dict(futures_cfg.get("open_interest", {}))
    symbols = [str(v) for v in futures_cfg.get("symbols", ["BTC/USDT", "ETH/USDT"])]
    symbol = str(symbols[0]) if symbols else "BTC/USDT"
    timeframe = "1h"

    store = build_data_store(
        backend=str(data_cfg.get("backend", "parquet")),
        data_dir=RAW_DATA_DIR,
        base_timeframe=str(cfg.get("universe", {}).get("base_timeframe") or "1m"),
        resample_source=str(data_cfg.get("resample_source", "direct")),
        derived_dir=Path("data") / "derived",
        partial_last_bucket=bool(data_cfg.get("partial_last_bucket", False)),
    )
    bars = store.load_ohlcv(symbol=symbol, timeframe=timeframe)
    if bars.empty:
        return {
            "short_only_enabled": bool(oi_cfg.get("short_horizon_only", False)),
            "short_horizon_max": str(oi_cfg.get("short_horizon_max", "30m")),
            "timeframe": timeframe,
            "timeframe_allowed": False,
            "oi_columns_present": [],
            "oi_non_null_rows": 0,
            "oi_active_runtime": False,
            "rule": "no_bars_available",
        }

    audit_cfg = dict(cfg)
    audit_cfg.setdefault("data", {})
    audit_cfg["data"]["include_futures_extras"] = True
    audit_cfg["data"].setdefault("futures_extras", {})
    audit_cfg["data"]["futures_extras"].setdefault("open_interest", {})
    audit_cfg["data"]["futures_extras"]["open_interest"]["short_horizon_only"] = True
    audit_cfg["data"]["futures_extras"]["open_interest"]["short_horizon_max"] = str(oi_cfg.get("short_horizon_max", "30m"))

    feat = calculate_features(
        bars.tail(200).reset_index(drop=True),
        config=audit_cfg,
        symbol=symbol,
        timeframe=timeframe,
        derived_data_dir=Path("data") / "derived",
    )
    usage = oi_runtime_usage(
        frame=feat,
        oi_columns=list(OI_DEPENDENT_COLUMNS),
        timeframe=timeframe,
        short_horizon_max=str(oi_cfg.get("short_horizon_max", "30m")),
        short_only_enabled=True,
    )
    attrs_usage = dict(feat.attrs.get("oi_usage", {}))
    if attrs_usage:
        usage["attrs_oi_usage"] = attrs_usage
        usage["runtime_vs_attrs_match"] = bool(
            bool(attrs_usage.get("oi_active", False)) == bool(usage.get("oi_active_runtime", False))
        )
    return usage


if __name__ == "__main__":
    main()
