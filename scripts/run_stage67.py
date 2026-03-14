"""Run Stage-67 validation protocol v3 with real executable evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.utils.hashing import stable_hash
from buffmini.validation import (
    estimate_trade_monte_carlo,
    evaluate_candidate_walkforward,
    evaluate_cross_perturbation,
    load_candidate_market_frame,
    resolve_validation_candidate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-67 validation protocol v3")
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
    direct = str(args.stage28_run_id).strip()
    if direct:
        return direct
    for name in ("stage62_summary.json", "stage60_summary.json", "stage52_summary.json"):
        payload = _load_json(docs_dir / name)
        run_id = str(payload.get("stage28_run_id", "")).strip()
        if run_id:
            return run_id
    return ""


def _primary_symbol(cfg: dict[str, Any]) -> str:
    primary = [str(v).strip() for v in cfg.get("research_scope", {}).get("primary_symbols", []) if str(v).strip()]
    if primary:
        return primary[0]
    universe = [str(v).strip() for v in cfg.get("universe", {}).get("symbols", []) if str(v).strip()]
    return universe[0] if universe else "BTC/USDT"


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    stage28_run_id = _resolve_stage28_run_id(args, docs_dir)
    if not stage28_run_id:
        raise SystemExit("unable to resolve stage28_run_id for Stage-67")

    candidate = resolve_validation_candidate(
        runs_dir=Path(args.runs_dir),
        stage28_run_id=stage28_run_id,
        docs_dir=docs_dir,
    )
    if not candidate:
        raise SystemExit(f"missing executable validation candidate for Stage-67: {stage28_run_id}")

    symbol = _primary_symbol(cfg)
    frame, market_meta = load_candidate_market_frame(
        cfg,
        symbol=symbol,
        timeframe=str(candidate.get("timeframe", "1h")),
    )

    walkforward = evaluate_candidate_walkforward(
        candidate=candidate,
        config=cfg,
        symbol=symbol,
        frame=frame,
        market_meta=market_meta,
    )

    out_dir = Path(args.runs_dir) / stage28_run_id / "stage67"
    out_dir.mkdir(parents=True, exist_ok=True)
    stage57_dir = Path(args.runs_dir) / stage28_run_id / "stage57"
    stage57_dir.mkdir(parents=True, exist_ok=True)

    windows = pd.DataFrame(walkforward.get("window_metrics", []))
    windows_path = out_dir / "walkforward_windows_real.csv"
    if not windows.empty:
        windows.to_csv(windows_path, index=False)
    forward_trades = walkforward.get("forward_trades", pd.DataFrame())
    if isinstance(forward_trades, pd.DataFrame) and not forward_trades.empty:
        forward_trades.to_csv(out_dir / "walkforward_forward_trades_real.csv", index=False)

    walkforward_summary = dict(walkforward.get("summary", {}))
    walkforward_metrics_path = out_dir / "walkforward_metrics_real.json"
    walkforward_metrics_payload = {
        "metric_source_type": "real_walkforward",
        "artifact_path": str(windows_path) if windows_path.exists() else "",
        "candidate_id": str(candidate.get("candidate_id", "")),
        "symbol": symbol,
        "timeframe": str(candidate.get("timeframe", "")),
        "execution_status": str(walkforward.get("execution_status", "BLOCKED")),
        "validation_state": str(walkforward.get("validation_state", "REAL_VALIDATION_FAILED")),
        "decision_use_allowed": bool(walkforward.get("decision_use_allowed", False)),
        "evidence_quality": "artifact_backed_real" if str(walkforward.get("execution_status", "")) == "EXECUTED" else "real_but_blocked",
        "split_count": int(walkforward_summary.get("total_windows", len(windows))),
        "usable_windows": int(walkforward_summary.get("usable_windows", 0)),
        "min_usable_windows": int(cfg.get("promotion_gates", {}).get("walkforward", {}).get("min_usable_windows", 5)),
        "median_forward_exp_lcb": float(walkforward_summary.get("median_forward_exp_lcb", 0.0)),
        "mean_forward_exp_lcb": float(dict(walkforward_summary.get("forward_expectancy", {})).get("mean", 0.0)),
        "classification": str(walkforward_summary.get("classification", "")),
        "classification_explanation": str(walkforward_summary.get("classification_explanation", "")),
        "market_meta": market_meta,
    }
    walkforward_metrics_path.write_text(json.dumps(walkforward_metrics_payload, indent=2, allow_nan=False), encoding="utf-8")
    (out_dir / "runtime_market_meta.json").write_text(json.dumps(market_meta, indent=2, allow_nan=False), encoding="utf-8")

    mc_cfg = dict(cfg.get("portfolio", {}).get("leverage_selector", {}))
    monte_carlo = estimate_trade_monte_carlo(
        forward_trades if isinstance(forward_trades, pd.DataFrame) else pd.DataFrame(),
        seed=int(mc_cfg.get("seed", cfg.get("search", {}).get("seed", 42))),
        n_paths=int(min(5000, max(100, mc_cfg.get("n_paths", 1000)))),
        block_size=int(max(4, mc_cfg.get("block_size_trades", 10))),
    )
    monte_carlo_payload = {
        "metric_source_type": "real_monte_carlo",
        "artifact_path": str(stage57_dir / "monte_carlo_metrics_real.json"),
        "candidate_id": str(candidate.get("candidate_id", "")),
        "symbol": symbol,
        "timeframe": str(candidate.get("timeframe", "")),
        "evidence_quality": "artifact_backed_real" if str(monte_carlo.get("execution_status", "")) == "EXECUTED" else "real_but_blocked",
        **{key: value for key, value in monte_carlo.items() if key not in {"trades", "equity_curve"}},
    }
    (stage57_dir / "monte_carlo_metrics_real.json").write_text(json.dumps(monte_carlo_payload, indent=2, allow_nan=False), encoding="utf-8")

    perturbation = evaluate_cross_perturbation(
        candidate=candidate,
        config=cfg,
        symbol=symbol,
        frame=frame,
        market_meta=market_meta,
    )
    perturb_df = pd.DataFrame(perturbation.get("rows", []))
    perturb_csv = stage57_dir / "cross_perturbation_windows_real.csv"
    if not perturb_df.empty:
        perturb_df.to_csv(perturb_csv, index=False)
    perturb_payload = {
        "metric_source_type": "real_cross_perturbation",
        "artifact_path": str(perturb_csv) if perturb_csv.exists() else "",
        "candidate_id": str(candidate.get("candidate_id", "")),
        "symbol": symbol,
        "timeframe": str(candidate.get("timeframe", "")),
        "evidence_quality": "artifact_backed_real" if str(perturbation.get("execution_status", "")) == "EXECUTED" else "real_but_blocked",
        **{key: value for key, value in perturbation.items() if key not in {"rows"}},
    }
    (stage57_dir / "cross_perturbation_metrics_real.json").write_text(json.dumps(perturb_payload, indent=2, allow_nan=False), encoding="utf-8")

    stage8_cfg = dict(cfg.get("evaluation", {}).get("stage8", {}).get("walkforward_v2", {}))
    promotion_walkforward = dict(cfg.get("promotion_gates", {}).get("walkforward", {}))
    portfolio_walkforward = dict(cfg.get("portfolio", {}).get("walkforward", {}))
    cross_cfg = dict(cfg.get("promotion_gates", {}).get("cross_seed", {}))
    status = (
        "SUCCESS"
        if str(walkforward_summary.get("status", "PARTIAL")) == "SUCCESS"
        and str(monte_carlo.get("execution_status", "")) == "EXECUTED"
        and str(perturbation.get("execution_status", "")) == "EXECUTED"
        else "PARTIAL"
    )
    validation_state = "REAL_VALIDATION_PASSED" if status == "SUCCESS" else "REAL_VALIDATION_FAILED"
    blocker_parts = []
    if str(walkforward_summary.get("status", "PARTIAL")) != "SUCCESS":
        blocker_parts.append(str(walkforward_summary.get("blocker_reason", "walkforward_gate_not_met")))
    if str(monte_carlo.get("execution_status", "")) != "EXECUTED":
        blocker_parts.append(str(monte_carlo.get("validation_state", "monte_carlo_blocked")).lower())
    if str(perturbation.get("execution_status", "")) != "EXECUTED":
        blocker_parts.append(str(perturbation.get("validation_state", "cross_perturbation_blocked")).lower())

    summary = {
        "stage": "67",
        "status": status,
        "execution_status": "EXECUTED",
        "stage_role": "real_validation",
        "validation_state": validation_state,
        "stage28_run_id": stage28_run_id,
        "candidate_id": str(candidate.get("candidate_id", "")),
        "symbol": symbol,
        "timeframe": str(candidate.get("timeframe", "")),
        "split_count": int(walkforward_metrics_payload.get("split_count", 0)),
        "usable_windows": int(walkforward_metrics_payload.get("usable_windows", 0)),
        "mean_score": float(walkforward_metrics_payload.get("mean_forward_exp_lcb", 0.0)),
        "mean_label": float(walkforward_summary.get("forward_expectancy", {}).get("mean", 0.0)),
        "median_forward_exp_lcb": float(walkforward_metrics_payload.get("median_forward_exp_lcb", 0.0)),
        "gates_effective": bool(walkforward.get("decision_use_allowed", False)),
        "metric_source_type": "real_walkforward",
        "walkforward_artifact_path": str(walkforward_metrics_path),
        "monte_carlo_artifact_path": str(stage57_dir / "monte_carlo_metrics_real.json"),
        "cross_perturbation_artifact_path": str(stage57_dir / "cross_perturbation_metrics_real.json"),
        "continuity_blocked": bool(market_meta.get("continuity_blocked", False)),
        "continuity_report": dict(market_meta.get("continuity_report", {})),
        "used_config_keys": [
            "evaluation.stage8.walkforward_v2.train_days",
            "evaluation.stage8.walkforward_v2.holdout_days",
            "evaluation.stage8.walkforward_v2.forward_days",
            "evaluation.stage8.walkforward_v2.step_days",
            "evaluation.stage8.walkforward_v2.reserve_tail_days",
            "portfolio.walkforward.min_forward_trades",
            "portfolio.walkforward.min_forward_exposure",
            "promotion_gates.walkforward.min_median_forward_exp_lcb",
            "promotion_gates.walkforward.min_usable_windows",
            "portfolio.leverage_selector.block_size_trades",
            "portfolio.leverage_selector.n_paths",
            "portfolio.leverage_selector.seed",
            "promotion_gates.cross_seed.min_passing_seeds",
            "data.continuity.strict_mode",
            "data.continuity.fail_on_gap",
            "reproducibility.frozen_research_mode",
        ],
        "effective_values": {
            "train_days": int(stage8_cfg.get("train_days", 180)),
            "holdout_days": int(stage8_cfg.get("holdout_days", 30)),
            "forward_days": int(stage8_cfg.get("forward_days", 30)),
            "step_days": int(stage8_cfg.get("step_days", 30)),
            "reserve_tail_days": int(stage8_cfg.get("reserve_tail_days", 0)),
            "min_forward_trades": int(portfolio_walkforward.get("min_forward_trades", 10)),
            "min_forward_exposure": float(portfolio_walkforward.get("min_forward_exposure", 0.01)),
            "min_median_forward_exp_lcb": float(promotion_walkforward.get("min_median_forward_exp_lcb", 0.0)),
            "min_usable_windows": int(promotion_walkforward.get("min_usable_windows", 5)),
            "monte_carlo_n_paths": int(min(5000, max(100, mc_cfg.get("n_paths", 1000)))),
            "monte_carlo_block_size": int(max(4, mc_cfg.get("block_size_trades", 10))),
            "cross_perturbation_min_survivors": int(cross_cfg.get("min_passing_seeds", 3)),
            "frozen_research_mode": bool(cfg.get("reproducibility", {}).get("frozen_research_mode", False)),
        },
        "monte_carlo_execution_status": str(monte_carlo.get("execution_status", "")),
        "cross_perturbation_execution_status": str(perturbation.get("execution_status", "")),
        "blocker_reason": ",".join([part for part in blocker_parts if part]),
    }
    summary["summary_hash"] = stable_hash(
        {
            "summary": summary,
            "walkforward_hash": walkforward_summary.get("summary_hash", ""),
            "monte_carlo": monte_carlo_payload,
            "cross_perturbation": perturb_payload,
        },
        length=16,
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (docs_dir / "stage67_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (docs_dir / "stage67_report.md").write_text(
        "\n".join(
            [
                "# Stage-67 Report",
                "",
                f"- status: `{summary['status']}`",
                f"- execution_status: `{summary['execution_status']}`",
                f"- stage_role: `{summary['stage_role']}`",
                f"- validation_state: `{summary['validation_state']}`",
                f"- stage28_run_id: `{summary['stage28_run_id']}`",
                f"- candidate_id: `{summary['candidate_id']}`",
                f"- symbol: `{summary['symbol']}`",
                f"- timeframe: `{summary['timeframe']}`",
                f"- split_count: `{summary['split_count']}`",
                f"- usable_windows: `{summary['usable_windows']}`",
                f"- mean_score: `{summary['mean_score']}`",
                f"- mean_label: `{summary['mean_label']}`",
                f"- median_forward_exp_lcb: `{summary['median_forward_exp_lcb']}`",
                f"- gates_effective: `{summary['gates_effective']}`",
                f"- metric_source_type: `{summary['metric_source_type']}`",
                f"- walkforward_artifact_path: `{summary['walkforward_artifact_path']}`",
                f"- monte_carlo_artifact_path: `{summary['monte_carlo_artifact_path']}`",
                f"- cross_perturbation_artifact_path: `{summary['cross_perturbation_artifact_path']}`",
                f"- continuity_blocked: `{summary['continuity_blocked']}`",
                f"- continuity_report: `{summary['continuity_report']}`",
                f"- used_config_keys: `{summary['used_config_keys']}`",
                f"- effective_values: `{summary['effective_values']}`",
                f"- monte_carlo_execution_status: `{summary['monte_carlo_execution_status']}`",
                f"- cross_perturbation_execution_status: `{summary['cross_perturbation_execution_status']}`",
                f"- blocker_reason: `{summary['blocker_reason']}`",
                f"- summary_hash: `{summary['summary_hash']}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"status: {summary['status']}")
    print(f"summary_hash: {summary['summary_hash']}")


if __name__ == "__main__":
    main()
