"""Stage-41 derivatives feature completion and contribution metrics runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.constants import RUNS_DIR
from buffmini.stage41.contribution import compute_family_contribution_metrics, oi_short_only_runtime_guard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-41 derivatives contribution analysis")
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
    if str(args.stage28_run_id).strip():
        return str(args.stage28_run_id).strip()
    stage40 = _load_json(docs_dir / "stage40_tradability_objective_summary.json")
    if str(stage40.get("stage28_run_id", "")).strip():
        return str(stage40["stage28_run_id"]).strip()
    stage39 = _load_json(docs_dir / "stage39_signal_generation_summary.json")
    return str(stage39.get("stage28_run_id", "")).strip()


def _render(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage-41 Derivatives Completion Report",
        "",
        "## Availability",
        f"- funding_available: `{bool(payload.get('funding_available', False))}`",
        f"- taker_buy_sell_available: `{bool(payload.get('taker_available', False))}`",
        f"- long_short_ratio_available: `{bool(payload.get('long_short_available', False))}`",
        f"- oi_short_only_mode_enabled: `{bool(payload.get('oi_short_only_mode_enabled', False))}`",
        "",
        "## OI Runtime Guard",
        f"- timeframe: `{payload.get('oi_runtime_guard', {}).get('timeframe', '')}`",
        f"- timeframe_allowed: `{bool(payload.get('oi_runtime_guard', {}).get('timeframe_allowed', False))}`",
        f"- oi_allowed: `{bool(payload.get('oi_runtime_guard', {}).get('oi_allowed', False))}`",
        "",
        "## Family Contribution Metrics",
        "| family | layer_a | layer_c | stage_a | stage_b | candidate_lift | activation_lift | tradability_lift | final_policy_share |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload.get("family_contributions", []):
        lines.append(
            "| {family} | {a} | {c} | {sa} | {sb} | {cl:.6f} | {al:.6f} | {tl:.6f} | {fps:.6f} |".format(
                family=str(row.get("family", "")),
                a=int(row.get("layer_a_count", 0)),
                c=int(row.get("layer_c_count", 0)),
                sa=int(row.get("stage_a_count", 0)),
                sb=int(row.get("stage_b_count", 0)),
                cl=float(row.get("candidate_lift", 0.0)),
                al=float(row.get("activation_lift", 0.0)),
                tl=float(row.get("tradability_lift", 0.0)),
                fps=float(row.get("final_policy_share", 0.0)),
            )
        )
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    args = parse_args()
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    stage28_run_id = _resolve_stage28_run_id(args, docs_dir)
    if not stage28_run_id:
        raise SystemExit("unable to resolve stage28_run_id for Stage-41")

    stage39_dir = Path(args.runs_dir) / stage28_run_id / "stage39"
    stage40_dir = Path(args.runs_dir) / stage28_run_id / "stage40"
    layer_a = pd.read_csv(stage39_dir / "layer_a_candidates.csv") if (stage39_dir / "layer_a_candidates.csv").exists() else pd.DataFrame()
    layer_c = pd.read_csv(stage39_dir / "layer_c_candidates.csv") if (stage39_dir / "layer_c_candidates.csv").exists() else pd.DataFrame()
    stage_a = pd.read_csv(stage40_dir / "stage_a_survivors.csv") if (stage40_dir / "stage_a_survivors.csv").exists() else pd.DataFrame()
    stage_b = pd.read_csv(stage40_dir / "stage_b_survivors.csv") if (stage40_dir / "stage_b_survivors.csv").exists() else pd.DataFrame()

    families = ["funding", "taker_buy_sell", "long_short_ratio", "open_interest", "flow", "volatility", "price"]
    contributions = compute_family_contribution_metrics(
        layer_a=layer_a,
        layer_c=layer_c,
        stage_a_survivors=stage_a,
        stage_b_survivors=stage_b,
        families=families,
    )

    stage37_deriv = _load_json(docs_dir / "stage37_derivatives_expansion_summary.json")
    stage38 = _load_json(docs_dir / "stage38_master_summary.json")
    oi_cfg = ((stage38.get("oi_usage", {}) or {}))
    oi_guard = oi_short_only_runtime_guard(
        timeframe=str(oi_cfg.get("timeframe", "1h")),
        short_only_enabled=bool(oi_cfg.get("short_only_enabled", True)),
        short_horizon_max=str(oi_cfg.get("short_horizon_max", "30m")),
    )

    payload = {
        "stage": "41",
        "seed": int(args.seed),
        "stage28_run_id": stage28_run_id,
        "funding_available": bool(stage37_deriv.get("funding_available", True)),
        "taker_available": bool(stage37_deriv.get("taker_available", True)),
        "long_short_available": bool(stage37_deriv.get("long_short_available", True)),
        "oi_short_only_mode_enabled": bool(oi_cfg.get("short_only_enabled", True)),
        "oi_runtime_guard": oi_guard,
        "family_contributions": contributions,
    }

    out_dir = Path(args.runs_dir) / stage28_run_id / "stage41"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    report_md = docs_dir / "stage41_derivatives_completion_report.md"
    report_json = docs_dir / "stage41_derivatives_completion_summary.json"
    report_md.write_text(_render(payload), encoding="utf-8")
    report_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    print(f"stage28_run_id: {stage28_run_id}")
    print(f"families_reported: {len(contributions)}")
    print(f"report_md: {report_md}")
    print(f"report_json: {report_json}")


if __name__ == "__main__":
    main()

