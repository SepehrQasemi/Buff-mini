"""Stage-14 ML-lite weighting and calibration."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.backtest.engine import run_backtest
from buffmini.config import compute_config_hash
from buffmini.constants import DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.signals.family_base import FamilyContext
from buffmini.signals.registry import build_families
from buffmini.stage10.evaluate import _build_features
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact
from buffmini.validation.walkforward_v2 import build_windows


@dataclass
class FitResult:
    model: str
    l2: float
    coefficients: np.ndarray
    intercept: float
    holdout_exp_lcb: float
    holdout_tpm: float
    accepted: bool


def run_stage14_weighting(
    *,
    config: dict[str, Any],
    seed: int = 42,
    dry_run: bool = True,
    symbols: list[str] | None = None,
    timeframe: str = "1h",
    runs_root: Path = RUNS_DIR,
    docs_dir: Path = Path("docs"),
    data_dir: Path = RAW_DATA_DIR,
    derived_dir: Path = DERIVED_DATA_DIR,
    stage_tag: str = "14.1",
    report_name: str = "stage14_1_weighting",
) -> dict[str, Any]:
    cfg = json.loads(json.dumps(config))
    stage14 = dict(((cfg.get("evaluation", {}) or {}).get("stage14", {})))
    stage14["enabled"] = True
    data = _build_family_dataset(
        config=cfg,
        seed=int(seed),
        dry_run=bool(dry_run),
        symbols=symbols,
        timeframe=timeframe,
        data_dir=data_dir,
        derived_dir=derived_dir,
    )
    if data.empty:
        raise RuntimeError("Stage-14.1 dataset is empty")
    x_cols = [col for col in data.columns if col.startswith("score_")][: int(stage14.get("max_features", 20))]
    if not x_cols:
        raise RuntimeError("Stage-14.1 no score features available")
    X = data[x_cols].to_numpy(dtype=float)
    y = data["target"].to_numpy(dtype=float)
    train_idx, hold_idx, fwd_idx = _split_indices(len(data))
    X_train, y_train = X[train_idx], y[train_idx]
    X_hold, y_hold = X[hold_idx], y[hold_idx]
    X_fwd, y_fwd = X[fwd_idx], y[fwd_idx]
    models = [str(x) for x in stage14.get("models", {}).get("allowed", ["logreg_l2", "ridge"])]
    l2_grid = [float(v) for v in stage14.get("weighting", {}).get("l2_grid", [0.1, 0.3, 1.0])]
    coef_clip = float(stage14.get("weighting", {}).get("coef_clip", 3.0))
    trade_bounds = dict(stage14.get("trade_rate_bounds", {}))
    tpm_min = float(trade_bounds.get("min_tpm", 5.0))
    tpm_max = float(trade_bounds.get("max_tpm", 80.0))

    fits: list[FitResult] = []
    for model in models:
        for l2 in l2_grid:
            coef, intercept = _fit_model(model=model, X=X_train, y=y_train, l2=l2, seed=int(seed))
            coef = np.clip(coef, -coef_clip, coef_clip)
            hold_score = _predict(model=model, X=X_hold, coef=coef, intercept=intercept)
            hold_bt = _quick_backtest_from_score(data.iloc[hold_idx].copy(), hold_score, cfg=cfg, symbol="HOLDOUT")
            hold_exp_lcb = float(_exp_lcb(pd.to_numeric(hold_bt.trades.get("pnl", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)))
            hold_tpm = float(_metrics(hold_bt, data.iloc[hold_idx])["tpm"])
            accepted = bool(hold_tpm >= tpm_min and hold_tpm <= tpm_max)
            fits.append(
                FitResult(
                    model=model,
                    l2=float(l2),
                    coefficients=coef.astype(float),
                    intercept=float(intercept),
                    holdout_exp_lcb=hold_exp_lcb,
                    holdout_tpm=hold_tpm,
                    accepted=accepted,
                )
            )
    fits_sorted = sorted(fits, key=lambda item: (item.accepted, item.holdout_exp_lcb), reverse=True)
    best = fits_sorted[0]
    fwd_score = _predict(model=best.model, X=X_fwd, coef=best.coefficients, intercept=best.intercept)
    fwd_bt = _quick_backtest_from_score(data.iloc[fwd_idx].copy(), fwd_score, cfg=cfg, symbol="FORWARD")
    fwd_metrics = _metrics(fwd_bt, data.iloc[fwd_idx])
    drift = float(np.linalg.norm(best.coefficients, ord=2) / max(1e-12, np.sqrt(len(best.coefficients))))
    drift_ok = bool(drift <= float(stage14.get("weighting", {}).get("drift_threshold", 0.4)))
    classification = "WEAK_EDGE" if float(fwd_metrics["exp_lcb"]) > 0 and drift_ok else "NO_EDGE"

    run_payload = {
        "stage": str(stage_tag),
        "model": best.model,
        "l2": float(best.l2),
        "coefficients": best.coefficients.tolist(),
        "intercept": float(best.intercept),
        "holdout_exp_lcb": float(best.holdout_exp_lcb),
        "forward_metrics": fwd_metrics,
        "drift_metric": drift,
        "drift_ok": drift_ok,
        "classification": classification,
        "seed": int(seed),
        "config_hash": compute_config_hash(cfg),
        "data_hash": stable_hash(data[x_cols + ["target"]].to_dict(orient="list"), length=16),
    }
    run_id = f"{utc_now_compact()}_{stable_hash(run_payload, length=12)}_stage14_1"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "model": fit.model,
                "l2": fit.l2,
                "holdout_exp_lcb": fit.holdout_exp_lcb,
                "holdout_tpm": fit.holdout_tpm,
                "accepted": fit.accepted,
            }
            for fit in fits_sorted
        ]
    ).to_csv(run_dir / "stage14_1_candidates.csv", index=False)
    (run_dir / "stage14_1_summary.json").write_text(json.dumps(_json_safe(run_payload), indent=2, allow_nan=False), encoding="utf-8")
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_md = docs_dir / f"{report_name}_report.md"
    report_json = docs_dir / f"{report_name}_summary.json"
    report_json.write_text(json.dumps(_json_safe(run_payload), indent=2, allow_nan=False), encoding="utf-8")
    report_md.write_text(
        "\n".join(
            [
                f"# Stage-{stage_tag} ML-lite Weighting Report",
                "",
                "## 1) What changed",
                "- Trained deterministic regularized linear model on family scores.",
                "",
                "## 2) How to run (dry-run + real)",
                "- dry-run: `python scripts/run_stage13.py --substage 14.1 --dry-run --seed 42`",
                "- real: `python scripts/run_stage13.py --substage 14.1 --seed 42`",
                "",
                "## 3) Validation gates & results",
                f"- best model: `{best.model}`",
                f"- holdout_exp_lcb: `{best.holdout_exp_lcb:.6f}`",
                f"- forward_exp_lcb: `{float(fwd_metrics['exp_lcb']):.6f}`",
                f"- drift_ok: `{drift_ok}`",
                "",
                "## 4) Key metrics tables (trade_count, tpm, PF, expectancy, exp_lcb, maxDD, wf, mc)",
                f"- forward trade_count: `{float(fwd_metrics['trade_count']):.2f}`",
                f"- forward tpm: `{float(fwd_metrics['tpm']):.2f}`",
                "",
                "## 5) Failures + reasons",
                "- drift_threshold exceeded" if not drift_ok else "- none",
                "",
                "## 6) Next actions",
                "- Calibrate thresholds per regime (Stage-14.2).",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return {"summary": run_payload, "run_id": run_id, "run_dir": run_dir, "report_md": report_md, "report_json": report_json}


def run_stage14_threshold_calibration(
    *,
    config: dict[str, Any],
    seed: int = 42,
    dry_run: bool = True,
    symbols: list[str] | None = None,
    timeframe: str = "1h",
    runs_root: Path = RUNS_DIR,
    docs_dir: Path = Path("docs"),
    data_dir: Path = RAW_DATA_DIR,
    derived_dir: Path = DERIVED_DATA_DIR,
    stage_tag: str = "14.2",
    report_name: str = "stage14_2_threshold",
) -> dict[str, Any]:
    cfg = json.loads(json.dumps(config))
    weighting = run_stage14_weighting(
        config=cfg,
        seed=int(seed),
        dry_run=bool(dry_run),
        symbols=symbols,
        timeframe=timeframe,
        runs_root=runs_root,
        docs_dir=docs_dir,
        data_dir=data_dir,
        derived_dir=derived_dir,
        stage_tag="14.1",
        report_name="stage14_1_weighting",
    )
    data = _build_family_dataset(
        config=cfg,
        seed=int(seed),
        dry_run=bool(dry_run),
        symbols=symbols,
        timeframe=timeframe,
        data_dir=data_dir,
        derived_dir=derived_dir,
    )
    stage14 = dict(((cfg.get("evaluation", {}) or {}).get("stage14", {})))
    low_grid = [float(v) for v in stage14.get("threshold_calibration", {}).get("low_grid", [0.2, 0.25, 0.3])]
    high_grid = [float(v) for v in stage14.get("threshold_calibration", {}).get("high_grid", [0.35, 0.45, 0.55])]
    trade_bounds = dict(stage14.get("trade_rate_bounds", {}))
    tpm_min = float(trade_bounds.get("min_tpm", 5.0))
    tpm_max = float(trade_bounds.get("max_tpm", 80.0))

    w = weighting["summary"]
    coef = np.asarray(w["coefficients"], dtype=float)
    intercept = float(w["intercept"])
    x_cols = [col for col in data.columns if col.startswith("score_")][: len(coef)]
    X = data[x_cols].to_numpy(dtype=float)
    base_score = _predict(model=str(w["model"]), X=X, coef=coef, intercept=intercept)
    regime = data["regime_label_stage10"].fillna("UNKNOWN").astype(str)
    rows: list[dict[str, Any]] = []
    best_by_regime: dict[str, dict[str, Any]] = {}
    for reg in sorted(regime.unique()):
        mask = regime == reg
        if int(mask.sum()) < 20:
            continue
        best = None
        for low in low_grid:
            for high in high_grid:
                threshold = float((low + high) / 2.0)
                bt = _quick_backtest_from_score(data.loc[mask].copy(), base_score[mask.to_numpy(dtype=bool)], cfg=cfg, symbol=f"REG::{reg}", threshold=threshold)
                met = _metrics(bt, data.loc[mask])
                accepted = bool(met["tpm"] >= tpm_min and met["tpm"] <= tpm_max)
                row = {
                    "regime": reg,
                    "low": low,
                    "high": high,
                    "threshold": threshold,
                    "exp_lcb": float(met["exp_lcb"]),
                    "tpm": float(met["tpm"]),
                    "accepted": accepted,
                }
                rows.append(row)
                if best is None or (row["accepted"], row["exp_lcb"]) > (best["accepted"], best["exp_lcb"]):
                    best = row
        if best is not None:
            best_by_regime[reg] = best

    payload = {
        "stage": str(stage_tag),
        "best_by_regime": _json_safe(best_by_regime),
        "classification": "WEAK_EDGE" if any(float(v.get("exp_lcb", 0.0)) > 0 for v in best_by_regime.values()) else "NO_EDGE",
        "seed": int(seed),
    }
    run_id = f"{utc_now_compact()}_{stable_hash(payload, length=12)}_stage14_2"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(run_dir / "stage14_2_grid.csv", index=False)
    (run_dir / "stage14_2_summary.json").write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_md = docs_dir / f"{report_name}_report.md"
    report_json = docs_dir / f"{report_name}_summary.json"
    report_json.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")
    report_md.write_text(
        f"# Stage-{stage_tag} Threshold Calibration Report\n\n- classification: `{payload['classification']}`\n- regimes_calibrated: `{len(best_by_regime)}`\n",
        encoding="utf-8",
    )
    return {"summary": payload, "run_id": run_id, "run_dir": run_dir, "report_md": report_md, "report_json": report_json}


def run_stage14_nested_walkforward(
    *,
    config: dict[str, Any],
    seed: int = 42,
    dry_run: bool = True,
    symbols: list[str] | None = None,
    timeframe: str = "1h",
    runs_root: Path = RUNS_DIR,
    docs_dir: Path = Path("docs"),
    data_dir: Path = RAW_DATA_DIR,
    derived_dir: Path = DERIVED_DATA_DIR,
    stage_tag: str = "14.3",
    report_name: str = "stage14_3_nested_wf",
) -> dict[str, Any]:
    cfg = json.loads(json.dumps(config))
    calibr = run_stage14_threshold_calibration(
        config=cfg,
        seed=int(seed),
        dry_run=bool(dry_run),
        symbols=symbols,
        timeframe=timeframe,
        runs_root=runs_root,
        docs_dir=docs_dir,
        data_dir=data_dir,
        derived_dir=derived_dir,
        stage_tag="14.2",
        report_name="stage14_2_threshold",
    )
    data = _build_family_dataset(
        config=cfg,
        seed=int(seed),
        dry_run=bool(dry_run),
        symbols=symbols,
        timeframe=timeframe,
        data_dir=data_dir,
        derived_dir=derived_dir,
    )
    wf_cfg = (((cfg.get("evaluation", {}) or {}).get("stage8", {}) or {}).get("walkforward_v2", {}))
    windows = build_windows(
        start_ts=data["timestamp"].iloc[0],
        end_ts=data["timestamp"].iloc[-1],
        train_days=int(wf_cfg.get("train_days", 180)),
        holdout_days=int(wf_cfg.get("holdout_days", 30)),
        forward_days=int(wf_cfg.get("forward_days", 30)),
        step_days=int(wf_cfg.get("step_days", 30)),
        reserve_tail_days=int(wf_cfg.get("reserve_tail_days", 0)),
    )
    if not windows:
        windows = []
    best_by_regime = dict(calibr["summary"].get("best_by_regime", {}))
    rows: list[dict[str, Any]] = []
    ts = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    for window in windows:
        fwd = data.loc[(ts >= window.forward_start) & (ts < window.forward_end)].copy().reset_index(drop=True)
        if fwd.empty:
            continue
        threshold = np.array([float(best_by_regime.get(reg, {}).get("threshold", 0.3)) for reg in fwd["regime_label_stage10"]], dtype=float)
        score = pd.to_numeric(fwd["score_price"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        bt = _quick_backtest_from_score(fwd, score, cfg=cfg, symbol="NESTED", dynamic_threshold=threshold)
        met = _metrics(bt, fwd)
        rows.append(
            {
                "window_idx": int(window.window_idx),
                "trade_count": float(met["trade_count"]),
                "tpm": float(met["tpm"]),
                "exp_lcb": float(met["exp_lcb"]),
                "PF": float(met["PF"]),
            }
        )
    table = pd.DataFrame(rows)
    consistency = float((table["exp_lcb"] > 0).mean()) if not table.empty else 0.0
    classification = "WEAK_EDGE" if consistency > 0.4 else "NO_EDGE"
    payload = {
        "stage": str(stage_tag),
        "folds_evaluated": int(len(table)),
        "consistency": consistency,
        "classification": classification,
    }
    run_id = f"{utc_now_compact()}_{stable_hash(payload, length=12)}_stage14_3"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(run_dir / "stage14_3_windows.csv", index=False)
    (run_dir / "stage14_3_summary.json").write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_md = docs_dir / f"{report_name}_report.md"
    report_json = docs_dir / f"{report_name}_summary.json"
    report_json.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")
    report_md.write_text(f"# Stage-{stage_tag} Nested Walkforward Report\n\n- classification: `{classification}`\n- folds: `{len(table)}`\n", encoding="utf-8")
    return {"summary": payload, "run_id": run_id, "run_dir": run_dir, "report_md": report_md, "report_json": report_json}


def run_stage14_meta_family(
    *,
    config: dict[str, Any],
    seed: int = 42,
    dry_run: bool = True,
    symbols: list[str] | None = None,
    timeframe: str = "1h",
    runs_root: Path = RUNS_DIR,
    docs_dir: Path = Path("docs"),
    data_dir: Path = RAW_DATA_DIR,
    derived_dir: Path = DERIVED_DATA_DIR,
    stage_tag: str = "14.4",
    report_name: str = "stage14_4_meta_family",
) -> dict[str, Any]:
    cfg = json.loads(json.dumps(config))
    nested = run_stage14_nested_walkforward(
        config=cfg,
        seed=int(seed),
        dry_run=bool(dry_run),
        symbols=symbols,
        timeframe=timeframe,
        runs_root=runs_root,
        docs_dir=docs_dir,
        data_dir=data_dir,
        derived_dir=derived_dir,
        stage_tag="14.3",
        report_name="stage14_3_nested_wf",
    )
    data = _build_family_dataset(
        config=cfg,
        seed=int(seed),
        dry_run=bool(dry_run),
        symbols=symbols,
        timeframe=timeframe,
        data_dir=data_dir,
        derived_dir=derived_dir,
    )
    families = ["score_price", "score_volatility", "score_flow"]
    regime = data["regime_label_stage10"].fillna("UNKNOWN").astype(str)
    rows = []
    for reg in sorted(regime.unique()):
        part = data.loc[regime == reg]
        if part.empty:
            continue
        means = np.array([float(pd.to_numeric(part[col], errors="coerce").fillna(0.0).mean()) for col in families], dtype=float)
        exp = np.exp(means - np.max(means))
        weights = exp / np.maximum(exp.sum(), 1e-12)
        rows.append({"regime": reg, "w_price": float(weights[0]), "w_volatility": float(weights[1]), "w_flow": float(weights[2])})
    alloc = pd.DataFrame(rows)
    enabled = bool(((cfg.get("evaluation", {}) or {}).get("stage14", {}).get("meta_family", {}).get("enabled", False)))
    if len(alloc) < 2:
        enabled = False
    classification = "WEAK_EDGE" if enabled and nested["summary"]["classification"] in {"WEAK_EDGE", "ROBUST_EDGE"} else "NO_EDGE"
    payload = {"stage": str(stage_tag), "enabled": enabled, "classification": classification}
    run_id = f"{utc_now_compact()}_{stable_hash(payload, length=12)}_stage14_4"
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    alloc.to_csv(run_dir / "stage14_4_allocations.csv", index=False)
    (run_dir / "stage14_4_summary.json").write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")
    docs_dir.mkdir(parents=True, exist_ok=True)
    report_md = docs_dir / f"{report_name}_report.md"
    report_json = docs_dir / f"{report_name}_summary.json"
    report_json.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")
    report_md.write_text(f"# Stage-{stage_tag} Meta-Family Allocation Report\n\n- enabled: `{enabled}`\n- classification: `{classification}`\n", encoding="utf-8")
    return {"summary": payload, "run_id": run_id, "run_dir": run_dir, "report_md": report_md, "report_json": report_json}


def run_stage13_14_master_report(
    *,
    docs_dir: Path = Path("docs"),
    report_md_name: str = "stage13_14_master_report.md",
    report_json_name: str = "stage13_14_master_summary.json",
) -> dict[str, Any]:
    docs_dir.mkdir(parents=True, exist_ok=True)
    stage_files = {
        "13_2": docs_dir / "stage13_2_price_family_summary.json",
        "13_3": docs_dir / "stage13_3_volatility_family_summary.json",
        "13_4": docs_dir / "stage13_4_flow_family_summary.json",
        "13_5": docs_dir / "stage13_5_combined_summary.json",
        "13_6": docs_dir / "stage13_6_robustness_summary.json",
        "13_7": docs_dir / "stage13_7_multihorizon_summary.json",
        "14_1": docs_dir / "stage14_1_weighting_summary.json",
        "14_2": docs_dir / "stage14_2_threshold_summary.json",
        "14_3": docs_dir / "stage14_3_nested_wf_summary.json",
        "14_4": docs_dir / "stage14_4_meta_family_summary.json",
    }
    loaded: dict[str, Any] = {}
    for key, path in stage_files.items():
        if path.exists():
            loaded[key] = json.loads(path.read_text(encoding="utf-8"))
        else:
            loaded[key] = {}
    verdict_candidates = [
        str(loaded.get("13_6", {}).get("classification", "NO_EDGE")),
        str(loaded.get("13_7", {}).get("classification", "NO_EDGE")),
        str(loaded.get("14_3", {}).get("classification", "NO_EDGE")),
        str(loaded.get("14_4", {}).get("classification", "NO_EDGE")),
    ]
    if "ROBUST_EDGE" in verdict_candidates:
        final_verdict = "ROBUST_EDGE"
    elif "WEAK_EDGE" in verdict_candidates:
        final_verdict = "WEAK_EDGE"
    else:
        final_verdict = "NO_EDGE"
    payload = {
        "stage": "13_14",
        "best_per_family": {
            "price": loaded.get("13_2", {}).get("best_combo", {}),
            "volatility": loaded.get("13_3", {}).get("best_combo", {}),
            "flow": loaded.get("13_4", {}).get("best_combo", {}),
        },
        "combined": loaded.get("13_5", {}).get("best", {}),
        "ml_enhanced": loaded.get("14_3", {}),
        "final_verdict": final_verdict,
    }
    report_json = docs_dir / report_json_name
    report_md = docs_dir / report_md_name
    report_json.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")
    report_md.write_text(
        "\n".join(
            [
                "# Stage-13 + Stage-14 Master Report",
                "",
                f"- final_verdict: `{final_verdict}`",
                f"- price_best: `{loaded.get('13_2', {}).get('best_combo', {})}`",
                f"- vol_best: `{loaded.get('13_3', {}).get('best_combo', {})}`",
                f"- flow_best: `{loaded.get('13_4', {}).get('best_combo', {})}`",
                f"- combined_best: `{loaded.get('13_5', {}).get('best', {})}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def _build_family_dataset(
    *,
    config: dict[str, Any],
    seed: int,
    dry_run: bool,
    symbols: list[str] | None,
    timeframe: str,
    data_dir: Path,
    derived_dir: Path,
) -> pd.DataFrame:
    cfg = json.loads(json.dumps(config))
    resolved_symbols = list(symbols or cfg.get("universe", {}).get("symbols", ["BTC/USDT", "ETH/USDT"]))
    features_by_symbol = _build_features(
        config=cfg,
        symbols=resolved_symbols,
        timeframe=str(timeframe),
        dry_run=bool(dry_run),
        seed=int(seed),
        data_dir=data_dir,
        derived_dir=derived_dir,
    )
    frames: list[pd.DataFrame] = []
    fams = build_families(enabled=["price", "volatility", "flow"], cfg=cfg)
    for symbol, frame in sorted(features_by_symbol.items()):
        ctx = FamilyContext(symbol=str(symbol), timeframe=str(timeframe), seed=int(seed), config=cfg, params={})
        out = frame.copy()
        for name, family in fams.items():
            out[f"score_{name}"] = family.compute_scores(frame, ctx).to_numpy(dtype=float)
        fut = pd.to_numeric(out["close"], errors="coerce").shift(-6)
        cur = pd.to_numeric(out["close"], errors="coerce")
        out["target"] = np.sign((fut - cur).fillna(0.0).to_numpy(dtype=float))
        out["symbol"] = symbol
        frames.append(out)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, axis=0, ignore_index=True)
    data = data.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return data


def _split_indices(n_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = int(n_rows)
    if n < 30:
        idx = np.arange(n)
        return idx[: max(1, n // 3)], idx[max(1, n // 3) : max(2, 2 * n // 3)], idx[max(2, 2 * n // 3) :]
    n_train = int(n * 0.6)
    n_hold = int(n * 0.2)
    idx = np.arange(n)
    return idx[:n_train], idx[n_train : n_train + n_hold], idx[n_train + n_hold :]


def _fit_model(*, model: str, X: np.ndarray, y: np.ndarray, l2: float, seed: int) -> tuple[np.ndarray, float]:
    x = np.asarray(X, dtype=float)
    target = np.asarray(y, dtype=float)
    if x.ndim != 2:
        raise ValueError("X must be 2D")
    if target.ndim != 1:
        raise ValueError("y must be 1D")
    n, k = x.shape
    if n == 0:
        return np.zeros(k, dtype=float), 0.0
    x_center = x - x.mean(axis=0, keepdims=True)
    y_center = target - target.mean()
    if str(model) == "ridge":
        eye = np.eye(k, dtype=float)
        beta = np.linalg.pinv(x_center.T @ x_center + float(l2) * eye) @ (x_center.T @ y_center)
        intercept = float(target.mean() - x.mean(axis=0).dot(beta))
        return beta.astype(float), intercept
    # Deterministic logistic-like l2 fit with fixed iterations.
    beta = np.zeros(k, dtype=float)
    intercept = 0.0
    y01 = (target > 0).astype(float)
    lr = 0.05
    for _ in range(200):
        logits = x @ beta + intercept
        prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        err = prob - y01
        grad_b = (x.T @ err) / max(1, n) + float(l2) * beta
        grad_i = float(err.mean())
        beta = beta - lr * grad_b
        intercept = float(intercept - lr * grad_i)
    return beta.astype(float), float(intercept)


def _predict(*, model: str, X: np.ndarray, coef: np.ndarray, intercept: float) -> np.ndarray:
    score = np.asarray(X, dtype=float) @ np.asarray(coef, dtype=float) + float(intercept)
    if str(model) == "logreg_l2":
        prob = 1.0 / (1.0 + np.exp(-np.clip(score, -30.0, 30.0)))
        return (prob - 0.5) * 2.0
    return score


def _quick_backtest_from_score(
    frame: pd.DataFrame,
    score: np.ndarray,
    *,
    cfg: dict[str, Any],
    symbol: str,
    threshold: float = 0.25,
    dynamic_threshold: np.ndarray | None = None,
) -> Any:
    work = frame.copy()
    sc = np.asarray(score, dtype=float)
    if dynamic_threshold is None:
        thr = np.full(len(work), float(threshold), dtype=float)
    else:
        thr = np.asarray(dynamic_threshold, dtype=float)
    direction = np.where(sc >= thr, 1, np.where(sc <= -thr, -1, 0))
    work["signal"] = pd.Series(direction, index=work.index).shift(1).fillna(0).astype(int)
    eval_cfg = (((cfg.get("evaluation", {}) or {}).get("stage10", {}) or {}).get("evaluation", {}))
    return run_backtest(
        frame=work,
        strategy_name="Stage14",
        symbol=str(symbol),
        signal_col="signal",
        max_hold_bars=int(eval_cfg.get("max_hold_bars", 24)),
        stop_atr_multiple=float(eval_cfg.get("stop_atr_multiple", 1.5)),
        take_profit_atr_multiple=float(eval_cfg.get("take_profit_atr_multiple", 3.0)),
        round_trip_cost_pct=float(cfg.get("costs", {}).get("round_trip_cost_pct", 0.1)),
        slippage_pct=float(cfg.get("costs", {}).get("slippage_pct", 0.0005)),
        cost_model_cfg=cfg.get("cost_model", {}),
    )


def _metrics(result: Any, frame: pd.DataFrame) -> dict[str, float]:
    trade_count = float(result.metrics.get("trade_count", 0.0))
    expectancy = float(result.metrics.get("expectancy", 0.0))
    pf = float(result.metrics.get("profit_factor", 0.0))
    maxdd = float(result.metrics.get("max_drawdown", 0.0))
    pnl = pd.to_numeric(result.trades.get("pnl", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    exp_lcb = _exp_lcb(pnl)
    months = _months(frame)
    tpm = float(trade_count / months) if months > 0 else 0.0
    return {"trade_count": trade_count, "expectancy": expectancy, "PF": pf, "maxDD": maxdd, "exp_lcb": exp_lcb, "tpm": tpm}


def _exp_lcb(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    mean = float(np.mean(arr))
    if arr.size <= 1:
        return mean
    std = float(np.std(arr, ddof=0))
    return float(mean - std / math.sqrt(float(arr.size)))


def _months(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return 0.0
    days = float((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400.0)
    return max(days / 30.0, 1e-6)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        num = float(value)
        return num if np.isfinite(num) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value
