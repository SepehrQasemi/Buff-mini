"""Run Stage-34 end-to-end offline on fixed local snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from buffmini.config import compute_config_hash, load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RUNS_DIR
from buffmini.data.snapshot import snapshot_metadata_from_config
from buffmini.stage34.data_snapshot import audit_and_complete_snapshot, write_snapshot_audit_docs
from buffmini.stage34.dataset_builder import DatasetConfig, build_stage34_dataset, write_stage34_dataset
from buffmini.stage34.eval import EvalConfig, evaluate_models_strict
from buffmini.stage34.evolution import EvolutionConfig, run_evolution
from buffmini.stage34.policy import PolicyConfig, replay_policy, select_best_policy
from buffmini.stage34.reporting import render_next_actions_md, render_stage34_report
from buffmini.stage34.train import TrainConfig, save_models, train_stage34_models
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-34 offline self-improving AI trading engine")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--budget-small", action="store_true")
    parser.add_argument("--budget-medium", action="store_true")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not bool(args.offline):
        raise SystemExit("Stage-34 only supports offline mode. Re-run with --offline.")
    budget = "medium" if bool(args.budget_medium) else "small"
    cfg = load_config(args.config)
    stage_cfg = ((cfg.get("evaluation", {}) or {}).get("stage34", {})) or {}
    seed = int(args.seed)
    symbols = [str(v) for v in stage_cfg.get("symbols", ["BTC/USDT", "ETH/USDT"])]
    required_tfs = [str(v) for v in stage_cfg.get("required_timeframes", ["1m", "5m", "15m", "30m", "1h", "4h", "1d"])]
    model_tfs = [str(v) for v in stage_cfg.get("timeframes", ["15m", "30m", "1h", "4h"])]
    dataset_cfg = (stage_cfg.get("dataset", {})) or {}
    training_cfg = (stage_cfg.get("training", {})) or {}
    eval_cfg = (stage_cfg.get("evaluation", {})) or {}
    evo_cfg = (stage_cfg.get("evolution", {})) or {}

    run_id = (
        f"{utc_now_compact()}_"
        f"{stable_hash({'seed': seed, 'cfg': compute_config_hash(cfg), 'symbols': symbols, 'tfs': model_tfs, 'generations': int(args.generations), 'budget': budget}, length=12)}"
        "_stage34"
    )
    run_root = Path(args.runs_dir) / run_id / "stage34"
    run_root.mkdir(parents=True, exist_ok=True)

    # 34.1 snapshot audit
    snapshot = audit_and_complete_snapshot(symbols=symbols, timeframes=required_tfs)
    write_snapshot_audit_docs(
        audit=snapshot,
        md_path=Path("docs/stage34_data_snapshot_audit.md"),
        json_path=Path("docs/stage34_data_snapshot_audit.json"),
    )

    # 34.2 dataset builder
    ds_cfg = DatasetConfig(
        symbols=tuple(symbols),
        timeframes=tuple(model_tfs),
        max_rows_per_symbol=int(dataset_cfg.get("max_rows_per_symbol", 300000)),
        max_features=int(dataset_cfg.get("max_features", 120)),
        horizons_hours=tuple(int(v) for v in dataset_cfg.get("window_horizons_hours", [24, 72])),
        resolved_end_ts=snapshot.get("resolved_end_ts"),
    )
    dataset, dataset_meta = build_stage34_dataset(cfg=ds_cfg)
    dataset_dir = run_root / "dataset"
    dataset_path, dataset_meta_path = write_stage34_dataset(dataset=dataset, meta=dataset_meta, out_dir=dataset_dir)
    Path("docs/stage34_ml_dataset_spec.md").write_text(_render_dataset_spec(dataset_meta, symbols, model_tfs), encoding="utf-8")

    # 34.3 training
    model_names = tuple(str(v) for v in training_cfg.get("models", ["logreg", "hgbt", "rf"]))
    models_bundle, train_summary = train_stage34_models(
        dataset,
        feature_columns=[str(v) for v in dataset_meta.get("feature_columns", [])],
        cfg=TrainConfig(seed=seed, models=model_names, calibration=str(training_cfg.get("calibration", "platt"))),
    )
    model_paths = save_models(models_bundle, out_dir=run_root / "models")
    train_payload = {
        "run_id": run_id,
        "models": train_summary.get("models", []),
        "model_paths": {k: str(v.as_posix()) for k, v in model_paths.items()},
        "train_summary_hash": train_summary.get("train_summary_hash", ""),
    }
    (run_root / "train_summary.json").write_text(json.dumps(train_payload, indent=2, allow_nan=False), encoding="utf-8")
    Path("docs/stage34_ml_model_card.md").write_text(_render_model_card(train_summary), encoding="utf-8")

    # 34.4 strict eval
    rows_eval, eval_summary = evaluate_models_strict(
        dataset,
        models=models_bundle,
        cfg=EvalConfig(
            threshold=float(eval_cfg.get("threshold", 0.55)),
            window_months=tuple(int(v) for v in eval_cfg.get("window_months", [3, 6])),
            step_months=int(eval_cfg.get("step_months", 1)),
            min_usable_windows=int(eval_cfg.get("min_usable_windows", 3)),
            mc_min_trades=int(eval_cfg.get("mc_min_trades", 30)),
            seed=seed,
        ),
    )
    (run_root / "eval_rows.csv").write_text(rows_eval.to_csv(index=False), encoding="utf-8")
    (run_root / "eval_summary.json").write_text(json.dumps(eval_summary, indent=2, allow_nan=False), encoding="utf-8")
    Path("docs/stage34_failure_modes.md").write_text(_render_failure_modes(rows_eval, eval_summary), encoding="utf-8")

    # 34.5 policy select + replay
    policy_cfg = PolicyConfig(
        threshold=float(eval_cfg.get("threshold", 0.55)),
        risk_cap=0.20,
        equity=10_000.0,
    )
    policy = select_best_policy(rows_eval, cfg=policy_cfg, seed=seed)
    chosen_name = str(policy.get("model_name", "")) if str(policy.get("model_name", "")) in models_bundle else sorted(models_bundle.keys())[0]
    policy["model_name"] = chosen_name
    heldout_start = int(max(10, round(dataset.shape[0] * 0.8)))
    heldout = dataset.iloc[heldout_start:, :].reset_index(drop=True)
    replay_research = replay_policy(heldout, model=models_bundle[chosen_name], policy=policy, mode="research", cfg=policy_cfg)
    replay_live = replay_policy(heldout, model=models_bundle[chosen_name], policy=policy, mode="live", cfg=policy_cfg)
    policy_payload = {"policy": policy, "research": replay_research, "live": replay_live}
    (run_root / "policy_snapshot.json").write_text(json.dumps(policy, indent=2, allow_nan=False), encoding="utf-8")
    (run_root / "policy_replay_summary.json").write_text(json.dumps(policy_payload, indent=2, allow_nan=False), encoding="utf-8")

    # 34.6-34.8 registry + evolution
    evo_result = run_evolution(
        _subset_for_evolution(dataset, budget=budget),
        feature_pool=[c for c in dataset.columns if c not in {"timestamp", "symbol", "timeframe", "label_primary", "label_auxiliary", "open", "high", "low", "close", "volume"}],
        registry_path=run_root / "model_registry.json",
        cfg=EvolutionConfig(
            generations=int(args.generations),
            max_models_per_generation=int(evo_cfg.get("max_models_per_generation", 12 if budget == "medium" else 8)),
            exploration_pct=float(evo_cfg.get("exploration_pct", 0.20)),
            seed=seed,
            budget=budget,
        ),
    )
    (run_root / "generation_summary.json").write_text(json.dumps(evo_result, indent=2, allow_nan=False), encoding="utf-8")

    summary = _build_summary(
        run_id=run_id,
        seed=seed,
        snapshot=snapshot,
        dataset_meta=dataset_meta,
        train_summary=train_summary,
        eval_rows=rows_eval,
        eval_summary=eval_summary,
        policy=policy,
        replay_research=replay_research,
        replay_live=replay_live,
        evolution=evo_result,
        config_hash=compute_config_hash(cfg),
        snapshot_meta=snapshot_metadata_from_config(cfg),
    )
    docs_summary = Path("docs/stage34_report_summary.json")
    docs_report = Path("docs/stage34_report.md")
    docs_next = Path("docs/stage34_next_actions.md")
    docs_summary.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    docs_report.write_text(render_stage34_report(summary), encoding="utf-8")
    docs_next.write_text(render_next_actions_md(summary), encoding="utf-8")

    print(f"run_id: {run_id}")
    print(f"summary_hash: {summary.get('summary_hash', '')}")
    print(f"report: {docs_report}")
    print(f"summary: {docs_summary}")
    print(f"next_actions: {docs_next}")


def _subset_for_evolution(dataset: pd.DataFrame, *, budget: str) -> pd.DataFrame:
    work = dataset.copy()
    if {"symbol", "timeframe"}.issubset(set(work.columns)):
        local = work.loc[(work["symbol"].astype(str) == "BTC/USDT") & (work["timeframe"].astype(str) == "1h")].copy()
        if local.empty:
            local = work.loc[work["timeframe"].astype(str) == "1h"].copy()
        work = local if not local.empty else work
    cap = 20_000 if str(budget) == "small" else 60_000
    if work.shape[0] > cap:
        work = work.tail(int(cap)).reset_index(drop=True)
    return work


def _render_dataset_spec(meta: dict[str, Any], symbols: list[str], timeframes: list[str]) -> str:
    lines = [
        "# Stage-34 ML Dataset Spec",
        "",
        f"- symbols: `{symbols}`",
        f"- timeframes: `{timeframes}`",
        f"- rows_total: `{int(meta.get('rows_total', 0))}`",
        f"- data_hash: `{meta.get('data_hash', '')}`",
        "",
        "## Features",
    ]
    lines.extend([f"- `{col}`" for col in meta.get("feature_columns", [])])
    lines.extend(
        [
            "",
            "## Labels",
            "- `label_primary`: triple-barrier-like directional label.",
            "- `label_auxiliary`: forward adverse excursion proxy.",
            "",
            "## Leakage Safety",
            "- Features are strictly based on current/past bars.",
            "- Labels use future horizon alignment in supervised-learning-safe form only.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _render_model_card(summary: dict[str, Any]) -> str:
    lines = [
        "# Stage-34 ML Model Card",
        "",
        f"- rows_total: `{int(summary.get('rows_total', 0))}`",
        f"- feature_count: `{int(summary.get('feature_count', 0))}`",
        f"- split: `{summary.get('splits', {})}`",
        "",
        "## Models",
    ]
    for row in summary.get("models", []):
        lines.append(
            "- `{model_name}` val_logloss={val_logloss:.6f} test_logloss={test_logloss:.6f} prob_std={prob_std:.6f}".format(
                model_name=str(row.get("model_name", "")),
                val_logloss=float(row.get("val_logloss", 0.0)),
                test_logloss=float(row.get("test_logloss", 0.0)),
                prob_std=float(row.get("prob_std", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Calibration",
            "- Time-safe calibration on validation split only.",
            "",
            "## Runtime",
            "- CPU deterministic training with bounded estimators.",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _render_failure_modes(rows: pd.DataFrame, summary: dict[str, Any]) -> str:
    lines = [
        "# Stage-34 Failure Modes",
        "",
        f"- wf_executed_pct: `{float(summary.get('wf_executed_pct', 0.0)):.2f}`",
        f"- mc_trigger_pct: `{float(summary.get('mc_trigger_pct', 0.0)):.2f}`",
        f"- final_verdict: `{summary.get('final_verdict', 'NO_EDGE')}`",
        "",
        "## Failure Mode Counts",
    ]
    counts = dict(summary.get("failure_mode_counts", {}))
    if counts:
        lines.extend([f"- `{k}`: `{int(v)}`" for k, v in sorted(counts.items())])
    else:
        lines.append("- none")
    if not rows.empty:
        top = rows.sort_values("exp_lcb", ascending=False).head(10)
        lines.extend(
            [
                "",
                "## Top Candidate Rows",
                "| model | cost | window | trades | exp_lcb | pf_adj | maxdd_p95 | failure |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for rec in top.to_dict(orient="records"):
            lines.append(
                "| {model_name} | {cost_mode} | {window_months} | {trade_count} | {exp_lcb:.6f} | {pf_adj:.6f} | {maxdd_p95:.6f} | {failure_mode} |".format(
                    model_name=str(rec.get("model_name", "")),
                    cost_mode=str(rec.get("cost_mode", "")),
                    window_months=int(rec.get("window_months", 0)),
                    trade_count=int(rec.get("trade_count", 0)),
                    exp_lcb=float(rec.get("exp_lcb", 0.0)),
                    pf_adj=float(rec.get("pf_adj", 0.0)),
                    maxdd_p95=float(rec.get("maxdd_p95", 0.0)),
                    failure_mode=str(rec.get("failure_mode", "")),
                )
            )
    return "\n".join(lines).strip() + "\n"


def _build_summary(
    *,
    run_id: str,
    seed: int,
    snapshot: dict[str, Any],
    dataset_meta: dict[str, Any],
    train_summary: dict[str, Any],
    eval_rows: pd.DataFrame,
    eval_summary: dict[str, Any],
    policy: dict[str, Any],
    replay_research: dict[str, Any],
    replay_live: dict[str, Any],
    evolution: dict[str, Any],
    config_hash: str,
    snapshot_meta: dict[str, Any],
) -> dict[str, Any]:
    live = eval_rows.loc[eval_rows["cost_mode"] == "live"].copy() if not eval_rows.empty and "cost_mode" in eval_rows.columns else pd.DataFrame()
    research = eval_rows.loc[eval_rows["cost_mode"] == "research"].copy() if not eval_rows.empty and "cost_mode" in eval_rows.columns else pd.DataFrame()
    live_best_exp = float(live["exp_lcb"].max()) if not live.empty else 0.0
    research_best_exp = float(research["exp_lcb"].max()) if not research.empty else 0.0
    verdict_eval = str(eval_summary.get("final_verdict", "NO_EDGE"))
    live_trades = int(replay_live.get("trade_count", 0))
    if verdict_eval == "EDGE" and live_trades > 0:
        final_verdict = "EDGE"
    elif verdict_eval in {"EDGE", "WEAK_EDGE"}:
        final_verdict = "WEAK_EDGE"
    elif int(dataset_meta.get("rows_total", 0)) <= 0:
        final_verdict = "INSUFFICIENT_DATA"
    else:
        final_verdict = "NO_EDGE"
    failure_counts = dict(eval_summary.get("failure_mode_counts", {}))
    top_failure = max(failure_counts.items(), key=lambda kv: kv[1])[0] if failure_counts else "unknown"
    top_reject = str((replay_live.get("top_reject_reasons", [{}]) or [{}])[0].get("reason", "")) if replay_live.get("top_reject_reasons") else ""
    if live_best_exp <= 0.0 and research_best_exp > 0.0:
        bottleneck = "cost_drag"
    elif live_trades == 0 and (top_reject or top_failure in {"no_trades_due_to_thresholds", "policy_zero_activation"}):
        bottleneck = "policy_thresholds"
    elif live_best_exp <= 0.0 and research_best_exp <= 0.0:
        bottleneck = "signal_quality"
    elif float(eval_summary.get("wf_executed_pct", 0.0)) <= 0.0 or float(eval_summary.get("mc_trigger_pct", 0.0)) <= 0.0:
        bottleneck = "insufficient_sample"
    elif top_failure in {"no_trades_due_to_thresholds", "policy_zero_activation"}:
        bottleneck = "policy_thresholds"
    else:
        bottleneck = "signal_quality"
    summary = {
        "run_id": str(run_id),
        "seed": int(seed),
        "config_hash": str(config_hash),
        "data_hash": str(dataset_meta.get("data_hash", "")),
        "resolved_end_ts": snapshot.get("resolved_end_ts"),
        "snapshot_hash": snapshot.get("snapshot_hash", ""),
        "dataset": {
            "rows_total": int(dataset_meta.get("rows_total", 0)),
            "feature_count": int(len(dataset_meta.get("feature_columns", []))),
            "timeframes": list(dataset_meta.get("timeframes", [])),
            "dataset_hash": str(dataset_meta.get("dataset_hash", "")),
        },
        "training": train_summary,
        "evaluation": {
            "wf_executed_pct": float(eval_summary.get("wf_executed_pct", 0.0)),
            "mc_trigger_pct": float(eval_summary.get("mc_trigger_pct", 0.0)),
            "failure_mode_counts": failure_counts,
            "research_best_exp_lcb": float(research_best_exp),
            "live_best_exp_lcb": float(live_best_exp),
        },
        "policy": {
            "policy_id": str(policy.get("policy_id", "")),
            "model_name": str(policy.get("model_name", "")),
            "status": str(policy.get("status", "")),
            "research_trade_count": int(replay_research.get("trade_count", 0)),
            "live_trade_count": int(live_trades),
            "accepted_rejected_breakdown": replay_live.get("accepted_rejected_breakdown", {}),
            "top_reject_reasons": replay_live.get("top_reject_reasons", []),
        },
        "generations": {
            "generation_count": int(len(evolution.get("generations", []))),
            "best_generation": int(evolution.get("best_generation", 0)),
            "trend": [
                {
                    "generation": int(item.get("generation", 0)),
                    "best_exp_lcb": float((item.get("best", {}) or {}).get("exp_lcb", 0.0)),
                    "median_exp_lcb": float((item.get("median", {}) or {}).get("exp_lcb", 0.0)),
                    "worst_exp_lcb": float((item.get("worst", {}) or {}).get("exp_lcb", 0.0)),
                }
                for item in evolution.get("generations", [])
            ],
        },
        "best_generation": int(evolution.get("best_generation", 0)),
        "best_combo": {
            "model_name": str(policy.get("model_name", "")),
            "threshold": float(policy.get("threshold", 0.0)),
        },
        "final_verdict": str(final_verdict),
        "top_bottleneck": str(bottleneck),
        "bottleneck_evidence": {
            "live_trade_count": int(live_trades),
            "top_reject_reason": top_reject,
            "failure_mode": top_failure,
            "wf_executed_pct": float(eval_summary.get("wf_executed_pct", 0.0)),
            "mc_trigger_pct": float(eval_summary.get("mc_trigger_pct", 0.0)),
            "live_best_exp_lcb": float(live_best_exp),
            "research_best_exp_lcb": float(research_best_exp),
        },
        "did_generations_improve": bool(evolution.get("did_generations_improve", False)),
        "if_false_why": (
            "Best generation exp_lcb did not improve across deterministic mutations."
            if not bool(evolution.get("did_generations_improve", False))
            else ""
        ),
        **snapshot_meta,
    }
    summary["summary_hash"] = stable_hash(
        {
            "seed": int(summary["seed"]),
            "data_hash": summary["data_hash"],
            "resolved_end_ts": summary["resolved_end_ts"],
            "training_hash": summary.get("training", {}).get("train_summary_hash", ""),
            "evaluation": summary["evaluation"],
            "policy": summary["policy"],
            "generations": summary["generations"],
            "final_verdict": summary["final_verdict"],
            "top_bottleneck": summary["top_bottleneck"],
        },
        length=16,
    )
    return summary


if __name__ == "__main__":
    main()
