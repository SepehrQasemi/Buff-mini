"""Stage-11 MTF engine wrapper over Stage-10 evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.config import compute_config_hash
from buffmini.constants import DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.mtf import (
    FEATURE_PACK_VERSION,
    MtfFeatureCache,
    MtfLayerSpec,
    build_cache_key,
    build_mtf_spec,
    compute_feature_pack,
    join_mtf_layer,
    resample_ohlcv,
    validate_resampled_schema,
)
from buffmini.stage10.evaluate import _build_features, run_stage10
from buffmini.stage11.hooks import build_noop_hooks
from buffmini.stage11.policy import DEFAULT_POLICY_CFG, build_stage11_policy_hooks
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact
from buffmini.validation.leakage_harness import run_registered_features_harness


STAGE11_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "mtf": {
        "base_timeframe": "1h",
        "layers": [
            {
                "name": "htf_4h",
                "timeframe": "4h",
                "role": "context",
                "features": [
                    "ema_50",
                    "ema_200",
                    "ema_slope_50",
                    "atr_14",
                    "atr_pct",
                    "atr_pct_rank_252",
                    "bb_mid_20",
                    "bb_upper_20_2",
                    "bb_lower_20_2",
                    "bb_bandwidth_20",
                    "volume_z_120",
                ],
                "tolerance_bars": 4,
                "enabled": True,
            },
            {
                "name": "ltf_15m",
                "timeframe": "15m",
                "role": "confirm",
                "features": [
                    "ema_50",
                    "ema_slope_50",
                    "atr_pct_rank_252",
                    "volume_z_120",
                ],
                "tolerance_bars": 2,
                "enabled": False,
            },
        ],
        "feature_pack_params": {},
        "hooks_enabled": {"bias": True, "confirm": False, "exit": False},
    },
    "hooks": json.loads(json.dumps(DEFAULT_POLICY_CFG)),
    "trade_count_guard": {
        "max_drop_pct": 15.0,
        "material_pf_improvement": 0.05,
        "material_exp_lcb_improvement": 0.5,
    },
}


def run_stage11(
    config: dict[str, Any],
    seed: int = 42,
    dry_run: bool = True,
    symbols: list[str] | None = None,
    timeframe: str = "1h",
    cost_mode: str = "v2",
    walkforward_v2_enabled: bool = True,
    runs_root: Path = RUNS_DIR,
    docs_dir: Path = Path("docs"),
    data_dir: Path = RAW_DATA_DIR,
    derived_dir: Path = DERIVED_DATA_DIR,
    write_docs: bool = True,
) -> dict[str, Any]:
    cfg = _normalize_stage11_config(config=config)
    stage11_cfg = cfg["evaluation"]["stage11"]
    mtf_spec = build_mtf_spec(stage11_cfg)
    enabled = bool(stage11_cfg.get("enabled", False))
    resolved_symbols = list(symbols or cfg.get("universe", {}).get("symbols", ["BTC/USDT", "ETH/USDT"]))
    resolved_timeframe = str(timeframe or mtf_spec.base_timeframe)
    if resolved_timeframe != "1h":
        raise ValueError("Stage-11 currently supports base timeframe=1h")

    base_features = _build_features(
        config=cfg,
        symbols=resolved_symbols,
        timeframe=resolved_timeframe,
        dry_run=bool(dry_run),
        seed=int(seed),
        data_dir=data_dir,
        derived_dir=derived_dir,
    )
    if not base_features:
        raise ValueError("No base features available for Stage-11")

    cache = MtfFeatureCache()
    if enabled:
        mtf_features, layer_stats = _build_mtf_features(
            base_features=base_features,
            mtf_spec=mtf_spec,
            cache=cache,
        )
    else:
        mtf_features = {symbol: frame.copy() for symbol, frame in base_features.items()}
        layer_stats = []

    hooks_cfg = _policy_cfg_from_stage11(stage11_cfg, enable_confirm=False, enable_exit=False)
    bias_hooks = build_stage11_policy_hooks(hooks_cfg) if enabled else build_noop_hooks()
    full_hooks_cfg = _policy_cfg_from_stage11(
        stage11_cfg,
        enable_confirm=bool(stage11_cfg.get("hooks", {}).get("confirm", {}).get("enabled", False)),
        enable_exit=bool(stage11_cfg.get("hooks", {}).get("exit", {}).get("enabled", False)),
    )
    full_hooks = build_stage11_policy_hooks(full_hooks_cfg) if enabled else build_noop_hooks()

    baseline = run_stage10(
        config=cfg,
        seed=int(seed),
        dry_run=bool(dry_run),
        symbols=resolved_symbols,
        timeframe=resolved_timeframe,
        cost_mode=cost_mode,
        walkforward_v2_enabled=walkforward_v2_enabled,
        runs_root=runs_root,
        docs_dir=docs_dir,
        data_dir=data_dir,
        derived_dir=derived_dir,
        write_docs=False,
        hooks=None,
        features_by_symbol_override=base_features,
    )
    bias_only = run_stage10(
        config=cfg,
        seed=int(seed),
        dry_run=bool(dry_run),
        symbols=resolved_symbols,
        timeframe=resolved_timeframe,
        cost_mode=cost_mode,
        walkforward_v2_enabled=walkforward_v2_enabled,
        runs_root=runs_root,
        docs_dir=docs_dir,
        data_dir=data_dir,
        derived_dir=derived_dir,
        write_docs=False,
        hooks=bias_hooks,
        features_by_symbol_override=mtf_features,
    )
    with_confirm = run_stage10(
        config=cfg,
        seed=int(seed),
        dry_run=bool(dry_run),
        symbols=resolved_symbols,
        timeframe=resolved_timeframe,
        cost_mode=cost_mode,
        walkforward_v2_enabled=walkforward_v2_enabled,
        runs_root=runs_root,
        docs_dir=docs_dir,
        data_dir=data_dir,
        derived_dir=derived_dir,
        write_docs=False,
        hooks=full_hooks,
        features_by_symbol_override=mtf_features,
    )

    comparison_rows = _build_comparison_rows(
        baseline=baseline,
        bias_only=bias_only,
        with_confirm=with_confirm,
    )
    guard = _trade_count_guard(
        baseline=baseline,
        candidate=with_confirm,
        cfg=stage11_cfg.get("trade_count_guard", {}),
    )
    walkforward = {
        "baseline_classification": str(baseline.get("walkforward_v2", {}).get("stage10_classification", "N/A")),
        "stage11_classification": str(with_confirm.get("walkforward_v2", {}).get("stage10_classification", "N/A")),
        "baseline_usable_windows": int(
            baseline.get("walkforward_v2", {}).get("stage10", {}).get("usable_windows", 0)
            if isinstance(baseline.get("walkforward_v2", {}).get("stage10"), dict)
            else 0
        ),
        "stage11_usable_windows": int(
            with_confirm.get("walkforward_v2", {}).get("stage10", {}).get("usable_windows", 0)
            if isinstance(with_confirm.get("walkforward_v2", {}).get("stage10"), dict)
            else 0
        ),
    }
    leakage = run_registered_features_harness(rows=520, seed=int(seed), shock_index=420, warmup_max=260)
    causality = _causality_status(layer_stats)
    leakage_status = "PASS" if int(leakage.get("leaks_found", 1)) == 0 else "FAIL"
    cache_stats = {
        "enabled": bool(enabled),
        "hits": int(cache.stats.hits),
        "misses": int(cache.stats.misses),
        "hit_rate": float(cache.stats.hit_rate),
    }

    config_hash = compute_config_hash(cfg)
    data_hash = _features_data_hash(mtf_features)
    deterministic_payload = {
        "baseline": baseline.get("baseline_vs_stage10", {}).get("stage10", {}),
        "bias_only": bias_only.get("baseline_vs_stage10", {}).get("stage10", {}),
        "with_confirm": with_confirm.get("baseline_vs_stage10", {}).get("stage10", {}),
        "mtf_layer_stats": layer_stats,
        "seed": int(seed),
        "config_hash": config_hash,
        "data_hash": data_hash,
    }
    signature = stable_hash(deterministic_payload, length=24)

    run_payload = {
        "seed": int(seed),
        "dry_run": bool(dry_run),
        "config_hash": config_hash,
        "data_hash": data_hash,
        "stage11_enabled": bool(enabled),
        "signature": signature,
    }
    run_id = f"{utc_now_compact()}_{stable_hash(run_payload, length=12)}_stage11"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    layer_stats_path = run_dir / "mtf_layer_stats.json"
    layer_stats_path.write_text(json.dumps(layer_stats, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(comparison_rows).to_csv(run_dir / "comparison_vs_stage10_7.csv", index=False)
    _write_layer_regime_distribution_csv(
        out_path=run_dir / "regime_distribution.csv",
        features_by_symbol=mtf_features,
    )

    summary = {
        "stage": "11",
        "run_id": run_id,
        "mtf_spec": _spec_to_dict(mtf_spec),
        "causality": causality,
        "leakage": {
            "status": leakage_status,
            "features_checked": int(leakage.get("features_checked", 0)),
            "leaks_found": int(leakage.get("leaks_found", 0)),
        },
        "cache": cache_stats,
        "comparisons": {
            "baseline_stage10_7": baseline.get("baseline_vs_stage10", {}).get("stage10", {}),
            "stage11_bias_only": bias_only.get("baseline_vs_stage10", {}).get("stage10", {}),
            "stage11_with_confirm": with_confirm.get("baseline_vs_stage10", {}).get("stage10", {}),
            "comparison_table_path": str(run_dir / "comparison_vs_stage10_7.csv"),
        },
        "trade_count_guard": guard,
        "walkforward": walkforward,
        "seed": int(seed),
        "config_hash": config_hash,
        "data_hash": data_hash,
        "determinism_signature": signature,
        "final_verdict": _final_verdict(guard=guard, baseline=baseline, candidate=with_confirm),
    }
    validate_stage11_summary_schema(summary)
    (run_dir / "stage11_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")

    if bool(write_docs):
        docs_dir.mkdir(parents=True, exist_ok=True)
        _write_stage11_report(
            summary=summary,
            out_md=docs_dir / "stage11_report.md",
            out_json=docs_dir / "stage11_report_summary.json",
        )
    return summary


def run_disabled_equivalence_snapshot(
    config: dict[str, Any],
    seed: int = 42,
    symbols: list[str] | None = None,
    timeframe: str = "1h",
    dry_run: bool = True,
    data_dir: Path = RAW_DATA_DIR,
    derived_dir: Path = DERIVED_DATA_DIR,
) -> dict[str, Any]:
    """Return strict trade/equity snapshots to verify Stage-11 disabled no-op behavior."""

    cfg = _normalize_stage11_config(config=config)
    resolved_symbols = list(symbols or cfg.get("universe", {}).get("symbols", ["BTC/USDT", "ETH/USDT"]))
    features = _build_features(
        config=cfg,
        symbols=resolved_symbols,
        timeframe=timeframe,
        dry_run=bool(dry_run),
        seed=int(seed),
        data_dir=data_dir,
        derived_dir=derived_dir,
    )
    from buffmini.stage10.evaluate import _evaluate_stage10

    left = _evaluate_stage10(features, cfg, hooks=None, return_paths=True)
    right = _evaluate_stage10(features, cfg, hooks=build_noop_hooks(), return_paths=True)

    snapshots: dict[str, Any] = {}
    for symbol in sorted(set(left.get("best_paths", {})) | set(right.get("best_paths", {}))):
        l_path = left.get("best_paths", {}).get(symbol, {})
        r_path = right.get("best_paths", {}).get(symbol, {})
        l_trades = l_path.get("trades", pd.DataFrame())
        r_trades = r_path.get("trades", pd.DataFrame())
        l_equity = l_path.get("equity_curve", pd.DataFrame())
        r_equity = r_path.get("equity_curve", pd.DataFrame())
        snapshots[symbol] = {
            "left_trade_times": _timestamp_list(l_trades, "entry_time"),
            "right_trade_times": _timestamp_list(r_trades, "entry_time"),
            "left_equity": _numeric_list(l_equity, "equity"),
            "right_equity": _numeric_list(r_equity, "equity"),
        }
    return snapshots


def validate_stage11_summary_schema(summary: dict[str, Any]) -> None:
    required = {
        "stage",
        "run_id",
        "mtf_spec",
        "causality",
        "leakage",
        "cache",
        "comparisons",
        "trade_count_guard",
        "walkforward",
        "final_verdict",
    }
    missing = required.difference(summary.keys())
    if missing:
        raise ValueError(f"Missing Stage-11 summary keys: {sorted(missing)}")
    if str(summary["stage"]) != "11":
        raise ValueError("stage must be '11'")
    if str(summary["causality"].get("status")) not in {"PASS", "FAIL"}:
        raise ValueError("causality.status must be PASS/FAIL")
    if str(summary["leakage"].get("status")) not in {"PASS", "FAIL"}:
        raise ValueError("leakage.status must be PASS/FAIL")
    if str(summary["final_verdict"]) not in {"IMPROVEMENT", "NEUTRAL", "REGRESSION"}:
        raise ValueError("final_verdict must be IMPROVEMENT/NEUTRAL/REGRESSION")


def _build_mtf_features(
    base_features: dict[str, pd.DataFrame],
    mtf_spec: Any,
    cache: MtfFeatureCache,
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    out: dict[str, pd.DataFrame] = {}
    stats: list[dict[str, Any]] = []

    for symbol, base in base_features.items():
        enriched = base.copy().sort_values("timestamp").reset_index(drop=True)
        symbol_hash = _features_data_hash({symbol: enriched})
        for layer in mtf_spec.layers:
            if not bool(layer.enabled):
                continue
            resampled = resample_ohlcv(base_df=enriched, target_timeframe=layer.timeframe)
            validate_resampled_schema(resampled)
            cache_key = build_cache_key(
                symbol=symbol,
                base_data_hash=symbol_hash,
                target_timeframe=layer.timeframe,
                feature_pack_version=FEATURE_PACK_VERSION,
                params=mtf_spec.feature_pack_params,
                layer_name=layer.name,
            )
            computed, cache_hit = cache.get_or_compute(
                cache_key,
                lambda r=resampled, l=layer.name, p=mtf_spec.feature_pack_params: compute_feature_pack(r, l, p),
                meta={
                    "symbol": symbol,
                    "layer_name": layer.name,
                    "timeframe": layer.timeframe,
                },
            )
            joined, align_stats = join_mtf_layer(
                base_df=enriched,
                layer_df=computed,
                layer_spec=layer,
                base_ts_col="timestamp",
                base_timeframe=mtf_spec.base_timeframe,
            )
            enriched = _inject_layer_aliases(joined, layer)
            row = {
                "symbol": symbol,
                "layer_name": layer.name,
                "layer_timeframe": layer.timeframe,
                "role": layer.role,
                "cache_key": cache_key,
                "cache_hit": bool(cache_hit),
            }
            row.update(align_stats)
            stats.append(row)
        out[symbol] = enriched
    return out, stats


def _inject_layer_aliases(frame: pd.DataFrame, layer: MtfLayerSpec) -> pd.DataFrame:
    out = frame.copy()
    prefix = f"{layer.name}__"
    if layer.role == "context":
        mapping = {
            f"{prefix}ema_slope_50": "stage11_context_ema_slope_50",
            f"{prefix}atr_pct_rank_252": "stage11_context_atr_pct_rank_252",
            f"{prefix}volume_z_120": "stage11_context_volume_z_120",
        }
    elif layer.role == "confirm":
        mapping = {
            f"{prefix}ema_slope_50": "stage11_confirm_ema_slope_50",
            f"{prefix}volume_z_120": "stage11_confirm_volume_z_120",
        }
    elif layer.role == "exit":
        mapping = {
            f"{prefix}atr_pct_rank_252": "stage11_exit_atr_pct_rank_252",
            f"{prefix}ema_slope_50": "stage11_exit_ema_slope_50",
        }
    else:
        mapping = {}

    for src, dst in mapping.items():
        if src in out.columns:
            out[dst] = pd.to_numeric(out[src], errors="coerce")
    return out


def _build_comparison_rows(
    baseline: dict[str, Any],
    bias_only: dict[str, Any],
    with_confirm: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, payload in (
        ("baseline_stage10_7", baseline),
        ("stage11_bias_only", bias_only),
        ("stage11_with_confirm", with_confirm),
    ):
        metrics = dict(payload.get("baseline_vs_stage10", {}).get("stage10", {}))
        rows.append(
            {
                "variant": name,
                "trade_count": float(metrics.get("trade_count", 0.0)),
                "profit_factor": float(metrics.get("profit_factor", 0.0)),
                "expectancy": float(metrics.get("expectancy", 0.0)),
                "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
                "pf_adj": float(metrics.get("pf_adj", 0.0)),
                "exp_lcb": float(metrics.get("exp_lcb", 0.0)),
                "walkforward_classification": str(payload.get("walkforward_v2", {}).get("stage10_classification", "N/A")),
            }
        )
    return rows


def _trade_count_guard(baseline: dict[str, Any], candidate: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    baseline_metrics = dict(baseline.get("baseline_vs_stage10", {}).get("stage10", {}))
    candidate_metrics = dict(candidate.get("baseline_vs_stage10", {}).get("stage10", {}))
    base_trades = float(baseline_metrics.get("trade_count", 0.0))
    cand_trades = float(candidate_metrics.get("trade_count", 0.0))
    max_drop_pct = float(cfg.get("max_drop_pct", 15.0))
    pf_threshold = float(cfg.get("material_pf_improvement", 0.05))
    exp_lcb_threshold = float(cfg.get("material_exp_lcb_improvement", 0.5))

    if base_trades <= 0:
        drop_pct = 0.0
    else:
        drop_pct = max(0.0, (base_trades - cand_trades) / base_trades * 100.0)
    pf_delta = float(candidate_metrics.get("profit_factor", 0.0) - baseline_metrics.get("profit_factor", 0.0))
    exp_lcb_delta = float(candidate_metrics.get("exp_lcb", 0.0) - baseline_metrics.get("exp_lcb", 0.0))
    material_improvement = bool(pf_delta >= pf_threshold and exp_lcb_delta >= exp_lcb_threshold)
    passed = bool(drop_pct <= max_drop_pct or material_improvement)
    return {
        "max_drop_pct": max_drop_pct,
        "observed_drop_pct": float(drop_pct),
        "pass": passed,
        "pf_delta": pf_delta,
        "exp_lcb_delta": exp_lcb_delta,
        "material_improvement": material_improvement,
    }


def _causality_status(layer_stats: list[dict[str, Any]]) -> dict[str, Any]:
    violations: list[str] = []
    for row in layer_stats:
        if float(row.get("max_lookback_bars", 0.0)) < 0:
            violations.append(f"{row.get('symbol')}:{row.get('layer_name')}:negative_lookback")
    status = "PASS" if not violations else "FAIL"
    return {
        "status": status,
        "notes": "causal merge_asof backward joins with assertions",
        "violations": violations,
    }


def _final_verdict(guard: dict[str, Any], baseline: dict[str, Any], candidate: dict[str, Any]) -> str:
    if not bool(guard.get("pass", False)):
        return "REGRESSION"
    base = dict(baseline.get("baseline_vs_stage10", {}).get("stage10", {}))
    cand = dict(candidate.get("baseline_vs_stage10", {}).get("stage10", {}))
    pf_delta = float(cand.get("profit_factor", 0.0) - base.get("profit_factor", 0.0))
    exp_delta = float(cand.get("expectancy", 0.0) - base.get("expectancy", 0.0))
    if pf_delta > 0 and exp_delta > 0:
        return "IMPROVEMENT"
    return "NEUTRAL"


def _features_data_hash(features_by_symbol: dict[str, pd.DataFrame]) -> str:
    payload: list[dict[str, Any]] = []
    for symbol in sorted(features_by_symbol):
        frame = features_by_symbol[symbol]
        ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
        payload.append(
            {
                "symbol": symbol,
                "rows": int(len(frame)),
                "start": ts.iloc[0].isoformat() if not ts.empty else None,
                "end": ts.iloc[-1].isoformat() if not ts.empty else None,
            }
        )
    return stable_hash(payload, length=16)


def _normalize_stage11_config(config: dict[str, Any]) -> dict[str, Any]:
    cfg = json.loads(json.dumps(config))
    evaluation = cfg.setdefault("evaluation", {})
    stage11 = _merge_defaults(STAGE11_DEFAULTS, evaluation.get("stage11", {}))
    evaluation["stage11"] = stage11
    cfg["evaluation"] = evaluation
    return cfg


def _policy_cfg_from_stage11(stage11_cfg: dict[str, Any], enable_confirm: bool, enable_exit: bool) -> dict[str, Any]:
    hooks_cfg = dict(stage11_cfg.get("hooks", {}))
    mtf_hooks_enabled = dict(stage11_cfg.get("mtf", {}).get("hooks_enabled", {}))
    policy_cfg = _merge_defaults(DEFAULT_POLICY_CFG, hooks_cfg)
    policy_cfg.setdefault("bias", {})
    policy_cfg.setdefault("confirm", {})
    policy_cfg.setdefault("exit", {})
    policy_cfg["bias"]["enabled"] = bool(
        stage11_cfg.get("hooks", {}).get("bias", {}).get("enabled", True) and mtf_hooks_enabled.get("bias", True)
    )
    policy_cfg["confirm"]["enabled"] = bool(enable_confirm and mtf_hooks_enabled.get("confirm", False))
    policy_cfg["exit"]["enabled"] = bool(enable_exit and mtf_hooks_enabled.get("exit", False))
    return policy_cfg


def _merge_defaults(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_defaults(dict(out[key]), value)
        else:
            out[key] = value
    return out


def _spec_to_dict(spec: Any) -> dict[str, Any]:
    return {
        "base_timeframe": str(spec.base_timeframe),
        "layers": [
            {
                "name": layer.name,
                "timeframe": layer.timeframe,
                "role": layer.role,
                "features": list(layer.features),
                "tolerance_bars": int(layer.tolerance_bars),
                "enabled": bool(layer.enabled),
            }
            for layer in spec.layers
        ],
        "feature_pack_params": dict(spec.feature_pack_params),
        "hooks_enabled": dict(spec.hooks_enabled),
    }


def _write_layer_regime_distribution_csv(out_path: Path, features_by_symbol: dict[str, pd.DataFrame]) -> None:
    rows: list[dict[str, Any]] = []
    for symbol, frame in features_by_symbol.items():
        if frame.empty:
            continue
        if "stage11_context_ema_slope_50" in frame.columns:
            trend = pd.to_numeric(frame["stage11_context_ema_slope_50"], errors="coerce")
        else:
            trend = pd.Series(np.nan, index=frame.index, dtype=float)
        if "stage11_context_atr_pct_rank_252" in frame.columns:
            atr_rank = pd.to_numeric(frame["stage11_context_atr_pct_rank_252"], errors="coerce")
        else:
            atr_rank = pd.Series(np.nan, index=frame.index, dtype=float)
        labels = np.where(atr_rank > 0.75, "VOL_EXPANSION", np.where(trend.abs() > 0.01, "TREND", "RANGE"))
        series = pd.Series(labels, index=frame.index, dtype="object").value_counts(normalize=True)
        rows.append(
            {
                "symbol": symbol,
                "TREND": float(series.get("TREND", 0.0) * 100.0),
                "RANGE": float(series.get("RANGE", 0.0) * 100.0),
                "VOL_EXPANSION": float(series.get("VOL_EXPANSION", 0.0) * 100.0),
            }
        )
    pd.DataFrame(rows).to_csv(out_path, index=False)


def _write_stage11_report(summary: dict[str, Any], out_md: Path, out_json: Path) -> None:
    comp = summary["comparisons"]
    baseline = comp["baseline_stage10_7"]
    bias = comp["stage11_bias_only"]
    confirm = comp["stage11_with_confirm"]
    lines: list[str] = []
    lines.append("# Stage-11 Report")
    lines.append("")
    lines.append("## What Stage-11 Adds")
    lines.append("- Config-driven MTF spec/layers with causal merge_asof alignment")
    lines.append("- Optional bias/confirm/exit hooks with no-op compatibility")
    lines.append("- Deterministic MTF feature cache keyed by data/config")
    lines.append("")
    lines.append("## Causality + Leakage")
    lines.append(f"- causality: `{summary['causality']['status']}`")
    lines.append(f"- leakage: `{summary['leakage']['status']}` (features_checked={summary['leakage']['features_checked']})")
    lines.append("")
    lines.append("## Cache")
    lines.append(f"- enabled: `{summary['cache']['enabled']}`")
    lines.append(f"- hits/misses: `{summary['cache']['hits']}/{summary['cache']['misses']}`")
    lines.append(f"- hit_rate: `{summary['cache']['hit_rate']:.6f}`")
    lines.append("")
    lines.append("## Baseline vs Stage-11")
    lines.append("| variant | trade_count | PF | expectancy | maxDD | exp_lcb |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for name, row in (
        ("baseline_stage10_7", baseline),
        ("stage11_bias_only", bias),
        ("stage11_with_confirm", confirm),
    ):
        lines.append(
            f"| {name} | {float(row.get('trade_count', 0.0)):.2f} | {float(row.get('profit_factor', 0.0)):.6f} | "
            f"{float(row.get('expectancy', 0.0)):.6f} | {float(row.get('max_drawdown', 0.0)):.6f} | "
            f"{float(row.get('exp_lcb', 0.0)):.6f} |"
        )
    lines.append("")
    lines.append("## Trade Count Guard")
    guard = summary["trade_count_guard"]
    lines.append(f"- pass: `{guard['pass']}`")
    lines.append(f"- observed_drop_pct: `{float(guard['observed_drop_pct']):.6f}`")
    lines.append(f"- max_drop_pct: `{float(guard['max_drop_pct']):.2f}`")
    lines.append("")
    lines.append("## Walkforward")
    lines.append(
        f"- classification: `{summary['walkforward']['baseline_classification']} -> {summary['walkforward']['stage11_classification']}`"
    )
    lines.append(
        f"- usable_windows: `{summary['walkforward']['baseline_usable_windows']} -> {summary['walkforward']['stage11_usable_windows']}`"
    )
    lines.append("")
    lines.append("## Final Verdict")
    lines.append(f"- {summary['final_verdict']}")
    lines.append("")
    lines.append(f"- run_id: `{summary['run_id']}`")
    lines.append(f"- config_hash: `{summary['config_hash']}`")
    lines.append(f"- data_hash: `{summary['data_hash']}`")
    lines.append(f"- seed: `{summary['seed']}`")
    lines.append(f"- determinism_signature: `{summary['determinism_signature']}`")
    out_md.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    out_json.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")


def _numeric_list(frame: pd.DataFrame, column: str) -> list[float]:
    if frame.empty or column not in frame.columns:
        return []
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0.0).astype(float)
    return [float(item) for item in values.tolist()]


def _timestamp_list(frame: pd.DataFrame, column: str) -> list[str]:
    if frame.empty or column not in frame.columns:
        return []
    series = pd.to_datetime(frame[column], utc=True, errors="coerce").dropna()
    return [str(item.isoformat()) for item in series.tolist()]
