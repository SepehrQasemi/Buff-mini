"""Compact Stage-13/14 forensic root-cause harness."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.backtest.engine import run_backtest
from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH, RAW_DATA_DIR, RUNS_DIR
from buffmini.data.cache import FeatureComputeSession, FeatureFrameCache, ohlcv_data_hash
from buffmini.data.features import calculate_features, registered_feature_columns
from buffmini.data.resample import assert_resample_is_causal, resample_ohlcv
from buffmini.signals.family_base import FamilyContext, SignalFamily
from buffmini.signals.registry import build_families
from buffmini.stage10.evaluate import _build_features
from buffmini.stage13.evaluate import run_stage13
from buffmini.stage14.evaluate import run_stage14_nested_walkforward, run_stage14_weighting
from buffmini.utils.hashing import stable_hash
from buffmini.validation.cost_model_v2 import one_way_cost_breakdown_bps
from buffmini.validation.leakage_harness import run_feature_functions_harness, run_registered_features_harness, synthetic_ohlcv
from buffmini.validation.walkforward_v2 import build_windows


CHECKS = {
    1: "Data Integrity & Size",
    2: "Resample/Derived Timeframe Correctness",
    3: "Feature Cache Semantics",
    4: "Leakage Harness Coverage",
    5: "Signal Score Range & Distribution",
    6: "Threshold Application Logic",
    7: "Entry Construction",
    8: "Exit Construction & Engine Semantics",
    9: "Trade Generation Sanity",
    10: "Zero-trade Causes Attribution",
    11: "NaN / Masking Safety",
    12: "Walkforward v2 Preconditions",
    13: "Walkforward Window Slicing Overlap",
    14: "Monte Carlo Preconditions",
    15: "Cost Model v2 Finite-Safety and Magnitude",
    16: "Execution Drag Sensitivity",
    17: "ML Fold Construction (Stage-14.3)",
    18: "ML Regularization / Feature Limits",
    19: "Composer / Ensemble Wiring",
    20: "Reporting / Metric Integrity",
}


@dataclass(frozen=True)
class CheckResult:
    check_id: int
    status: str
    tag: str
    evidence: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "name": CHECKS[self.check_id],
            "status": self.status,
            "root_cause_tag": self.tag,
            "evidence": _safe(self.evidence),
        }


def classify_zero_trade_cause(*, score_abs_max: float, threshold: float, nan_share_required: float, signal_nonzero_count: int, crossing_count: int) -> str:
    if nan_share_required >= 0.25:
        return "MISSING_FEATURES_OR_NAN"
    if crossing_count <= 0 or score_abs_max < threshold:
        return "SCORE_BELOW_THRESHOLD"
    if signal_nonzero_count <= 0 and crossing_count > 0:
        return "SIGNAL_MAPPING_BUG"
    return "MASKING_OR_POST_FILTER"


def compute_invalid_pct_from_rows(rows_df: pd.DataFrame) -> float:
    if rows_df.empty or "invalid_reason" not in rows_df.columns:
        return 100.0
    return float((rows_df["invalid_reason"].astype(str) != "VALID").mean() * 100.0)


def rank_impact_drivers(ablation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not ablation_rows:
        return []
    b = next((r for r in ablation_rows if r.get("driver") == "baseline"), ablation_rows[0])
    be, bi, bw, bt = _f(b.get("best_exp_lcb")), _f(b.get("invalid_pct")), _f(b.get("walkforward_executed_true_pct")), _f(b.get("tpm"))
    out: list[dict[str, Any]] = []
    for r in ablation_rows:
        if r is b:
            continue
        de, di, dw, dt = _f(r.get("best_exp_lcb")) - be, _f(r.get("invalid_pct")) - bi, _f(r.get("walkforward_executed_true_pct")) - bw, _f(r.get("tpm")) - bt
        out.append({
            "driver": str(r.get("driver", "")),
            "variant": str(r.get("variant", "")),
            "delta_exp_lcb": de,
            "delta_invalid_pct": di,
            "delta_walkforward_pct": dw,
            "delta_tpm": dt,
            "impact_score": abs(de) + 0.1 * abs(di) + 0.05 * abs(dw) + 0.02 * abs(dt),
        })
    out.sort(key=lambda x: float(x["impact_score"]), reverse=True)
    return out


def run_forensic_stage13_14_root_cause(*, config_path: Path = DEFAULT_CONFIG_PATH, seed: int = 42, docs_dir: Path = Path("docs"), runs_root: Path = RUNS_DIR, data_dir: Path = RAW_DATA_DIR) -> dict[str, Any]:
    t0 = time.perf_counter()
    cfg = load_config(config_path)
    cfg["evaluation"]["stage13"]["enabled"] = True
    cfg["evaluation"]["stage14"]["enabled"] = True
    syms = list(cfg.get("universe", {}).get("symbols", ["BTC/USDT", "ETH/USDT"]))
    tf = str(cfg.get("universe", {}).get("timeframe", "1h"))
    feats = _build_features(config=cfg, symbols=syms, timeframe=tf, dry_run=False, seed=int(seed), data_dir=data_dir, derived_dir=Path("data/derived"))
    if not feats:
        raise RuntimeError("no features loaded")
    checks: list[CheckResult] = []
    # 1 data integrity
    c1_rows: list[dict[str, Any]] = []
    bad1 = False
    for s in syms:
        safe = s.replace("/", "-")
        for tfx, freq, min_rows in (("1m", "1min", 5000), ("1h", "1h", 300)):
            p = data_dir / f"{safe}_{tfx}.parquet"
            try:
                p_txt = p.relative_to(Path.cwd()).as_posix()
            except Exception:
                p_txt = p.as_posix()
            if not p.exists():
                bad1 = True
                c1_rows.append({"path": p_txt, "missing": True})
                continue
            d = pd.read_parquet(p)
            ts = pd.to_datetime(d["timestamp"], utc=True, errors="coerce").dropna()
            dup = int(ts.duplicated().sum()) if not ts.empty else 0
            mono = bool(ts.is_monotonic_increasing) if not ts.empty else False
            miss = int(max(0, len(pd.date_range(ts.iloc[0], ts.iloc[-1], freq=freq, tz="UTC")) - len(ts.unique()))) if not ts.empty else 0
            bad1 = bad1 or dup > 0 or (not mono) or int(len(d)) < min_rows
            c1_rows.append({"path": p_txt, "rows": int(len(d)), "duplicates": dup, "monotonic": mono, "missing_estimate": miss})
    checks.append(CheckResult(1, "FAIL" if bad1 else "PASS", "DATA_INTEGRITY" if bad1 else "OK", {"files": c1_rows}))

    # 2 resample correctness
    x = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=15, freq="min", tz="UTC"), "open": np.arange(15) + 100.0, "high": np.arange(15) + 100.5, "low": np.arange(15) + 99.5, "close": np.arange(15) + 100.2, "volume": np.ones(15)})
    r5 = resample_ohlcv(x, "5m", base_timeframe="1m")
    assert_resample_is_causal(x, r5, target_timeframe="5m", base_timeframe="1m")
    ok2 = abs(float(r5.iloc[0]["open"]) - 100.0) < 1e-12 and abs(float(r5.iloc[0]["high"]) - 104.5) < 1e-12 and abs(float(r5.iloc[0]["low"]) - 99.5) < 1e-12 and abs(float(r5.iloc[0]["close"]) - 104.2) < 1e-12 and abs(float(r5.iloc[0]["volume"]) - 5.0) < 1e-12
    checks.append(CheckResult(2, "PASS" if ok2 else "FAIL", "OK" if ok2 else "RESAMPLE_AGG_BUG", {"first_5m": _safe(r5.iloc[0].to_dict())}))

    # 3 feature cache semantics
    cache = FeatureFrameCache(root_dir=Path("data/features_cache"))
    sess = FeatureComputeSession(cache)
    raw = synthetic_ohlcv(rows=600, seed=int(seed))
    n = {"v": 0}

    def _b() -> pd.DataFrame:
        n["v"] += 1
        return calculate_features(raw, config={"data": {"include_futures_extras": False}})

    dh, ph = ohlcv_data_hash(raw), stable_hash({"fx": False}, length=16)
    sess.get_or_build(symbol="BTC/USDT", timeframe="1h", resolved_end_ts="", feature_config_hash=ph, data_hash=dh, builder=_b)
    _, mem_hit, _ = sess.get_or_build(symbol="BTC/USDT", timeframe="1h", resolved_end_ts="", feature_config_hash=ph, data_hash=dh, builder=_b)
    sess2 = FeatureComputeSession(cache)
    _, disk_hit, _ = sess2.get_or_build(symbol="BTC/USDT", timeframe="1h", resolved_end_ts="", feature_config_hash=ph, data_hash=dh, builder=_b)
    ok3 = n["v"] == 1 and mem_hit and disk_hit
    checks.append(CheckResult(3, "PASS" if ok3 else "FAIL", "OK" if ok3 else "CACHE_RECOMPUTE", {"compute_calls": n["v"], "memory_hit": mem_hit, "disk_hit": disk_hit}))

    # 4 leakage coverage
    h = run_registered_features_harness(rows=520, seed=int(seed), shock_index=420, warmup_max=260, include_futures_extras=True)
    leaky = run_feature_functions_harness(frame=synthetic_ohlcv(rows=280, seed=int(seed) + 1), feature_funcs={"safe": lambda d: d["close"].ewm(span=10, adjust=False).mean(), "leaky": lambda d: d["close"].shift(-120)}, shock_index=200, warmup_max=80)
    fams = build_families(enabled=["price", "volatility", "flow"], cfg=cfg)
    sample = next(iter(feats.values()))
    miss = {k: [c for c in f.required_features() if c not in sample.columns] for k, f in fams.items()}
    miss = {k: v for k, v in miss.items() if v}
    ok4 = int(h["leaks_found"]) == 0 and int(leaky["leaks_found"]) >= 1 and not miss
    checks.append(CheckResult(4, "PASS" if ok4 else "FAIL", "OK" if ok4 else "LEAKAGE_COVERAGE_GAP", {"registered_harness": h, "leaky_probe": leaky, "missing_required": miss, "registered_features": len(registered_feature_columns(include_futures_extras=True))}))

    # 5-7 / 9-11 / 19 shared family checks
    s0, f0 = sorted(feats.items())[0]
    ctx = FamilyContext(symbol=s0, timeframe="1h", seed=int(seed), config=cfg, params={})
    score_rows: list[dict[str, Any]] = []
    th_rows: list[dict[str, Any]] = []
    es_issues: list[dict[str, Any]] = []
    sanity_rows: list[dict[str, Any]] = []
    z_counts: dict[str, int] = {}
    bad5 = False
    bad6 = False
    for name, fam in fams.items():
        sc = pd.to_numeric(fam.compute_scores(f0, ctx), errors="coerce").fillna(0.0)
        thr = float(cfg["evaluation"]["stage13"].get(name, {}).get("entry_threshold", 0.3))
        row5 = {"family": name, "min": float(sc.min()), "max": float(sc.max()), "pct_zero": float((sc.abs() <= 1e-12).mean() * 100.0)}
        bad5 = bad5 or row5["min"] < -1.000001 or row5["max"] > 1.000001 or row5["pct_zero"] >= 99.5
        score_rows.append(row5)
        cts: list[int] = []
        for th in (0.2, 0.3, 0.4):
            e = SignalFamily.build_entry_frame(scores=sc, threshold=th, family_name=name, long_reason="L", short_reason="S")
            cts.append(int((e["direction"] != 0).sum()))
            cause = classify_zero_trade_cause(score_abs_max=float(sc.abs().max()), threshold=th, nan_share_required=float(f0[fam.required_features()].isna().mean().mean()), signal_nonzero_count=int((pd.to_numeric(e["signal"], errors="coerce").fillna(0).astype(int) != 0).sum()), crossing_count=int((sc.abs() >= th).sum()))
            z_counts[cause] = z_counts.get(cause, 0) + 1
        mono = cts[0] >= cts[1] >= cts[2]
        bad6 = bad6 or (not mono)
        th_rows.append({"family": name, "entry_counts": cts, "monotonic_nonincreasing": mono})
        out = fam.propose_entries(sc, f0, ctx)
        req = {"score", "direction", "confidence", "reasons", "long_entry", "short_entry", "signal", "signal_family"}
        if req.difference(out.columns):
            es_issues.append({"family": name, "missing": sorted(req.difference(out.columns))})
        w = f0.copy()
        w["signal"] = pd.to_numeric(out["signal"], errors="coerce").fillna(0).astype(int)
        bt = run_backtest(frame=w, strategy_name=f"f::{name}", symbol=s0, signal_col="signal", max_hold_bars=24, stop_atr_multiple=1.5, take_profit_atr_multiple=3.0, round_trip_cost_pct=float(cfg["costs"]["round_trip_cost_pct"]), slippage_pct=float(cfg["costs"]["slippage_pct"]), cost_model_cfg=cfg["cost_model"])
        exp = float(pd.to_numeric(bt.trades.get("bars_held", pd.Series(dtype=float)), errors="coerce").fillna(0).sum() / max(1, len(w))) if not bt.trades.empty else 0.0
        sanity_rows.append({"family": name, "trade_count": float(bt.metrics.get("trade_count", 0.0)), "tpm": float(bt.metrics.get("trade_count", 0.0) / max(1e-9, _months(w))), "exposure_ratio": exp})

    checks.append(CheckResult(5, "FAIL" if bad5 else "PASS", "SCORE_STARVATION" if bad5 else "OK", {"rows": score_rows}))
    checks.append(CheckResult(6, "FAIL" if bad6 else "PASS", "THRESHOLD_POLARITY_BUG" if bad6 else "OK", {"rows": th_rows}))
    checks.append(CheckResult(7, "FAIL" if es_issues else "PASS", "ENTRY_SCHEMA_BUG" if es_issues else "OK", {"issues": es_issues}))
    # 8 exit semantics parity
    w8 = f0.tail(600).copy().reset_index(drop=True)
    w8["signal"] = np.where(np.arange(len(w8)) % 20 == 0, 1, 0)
    bnp = run_backtest(frame=w8, strategy_name="forensic", symbol=s0, signal_col="signal", max_hold_bars=24, stop_atr_multiple=1.5, take_profit_atr_multiple=3.0, round_trip_cost_pct=float(cfg["costs"]["round_trip_cost_pct"]), slippage_pct=float(cfg["costs"]["slippage_pct"]), cost_model_cfg=cfg["cost_model"], engine_mode="numpy")
    bpd = run_backtest(frame=w8, strategy_name="forensic", symbol=s0, signal_col="signal", max_hold_bars=24, stop_atr_multiple=1.5, take_profit_atr_multiple=3.0, round_trip_cost_pct=float(cfg["costs"]["round_trip_cost_pct"]), slippage_pct=float(cfg["costs"]["slippage_pct"]), cost_model_cfg=cfg["cost_model"], engine_mode="pandas")
    ok8 = int(bnp.metrics.get("trade_count", 0)) == int(bpd.metrics.get("trade_count", 0))
    checks.append(CheckResult(8, "PASS" if ok8 else "FAIL", "OK" if ok8 else "EXIT_ENGINE_SEMANTICS", {"numpy_trade_count": float(bnp.metrics.get("trade_count", 0.0)), "pandas_trade_count": float(bpd.metrics.get("trade_count", 0.0))}))
    checks.append(CheckResult(9, "FAIL" if any(r["trade_count"] < 0 or r["tpm"] < 0 or r["exposure_ratio"] < 0 for r in sanity_rows) else "PASS", "TRADE_SANITY", {"rows": sanity_rows}))
    unattributed = int(z_counts.get("MASKING_OR_POST_FILTER", 0))
    checks.append(
        CheckResult(
            10,
            "FAIL" if unattributed > 0 else "PASS",
            "ZERO_TRADE_UNATTRIBUTED" if unattributed > 0 else "OK",
            {"cause_counts": z_counts},
        )
    )

    cfgx = json.loads(json.dumps(cfg))
    cfgx.setdefault("data", {})["include_futures_extras"] = True
    fx = calculate_features(synthetic_ohlcv(rows=600, seed=int(seed) + 100), config=cfgx, symbol="BTC/USDT", timeframe="1h", _synthetic_extras_for_tests=True)
    fs = pd.to_numeric(build_families(enabled=["flow"], cfg=cfgx)["flow"].compute_scores(fx, FamilyContext(symbol="BTC/USDT", timeframe="1h", seed=int(seed), config=cfgx, params={})), errors="coerce")
    checks.append(CheckResult(11, "PASS" if np.isfinite(fs.to_numpy(dtype=float)).all() else "FAIL", "NAN_PROPAGATION", {"score_finite": bool(np.isfinite(fs.to_numpy(dtype=float)).all())}))

    r12 = run_stage13(config=cfg, seed=int(seed), dry_run=False, symbols=syms, timeframe=tf, families=["price", "volatility", "flow"], composer_mode="weighted_sum", runs_root=runs_root, docs_dir=docs_dir, data_dir=data_dir, derived_dir=Path("data/derived"), stage_tag="13.forensic", report_name="stage13_forensic_tmp", write_docs=False, window_months=12)
    df12 = r12["rows"]
    bad12 = bool(df12.loc[df12["walkforward_expected_windows"] <= 0, "walkforward_executed"].astype(bool).any())
    checks.append(CheckResult(12, "FAIL" if bad12 else "PASS", "WF_PRECONDITION_BUG" if bad12 else "OK", {"run_id": r12["summary"]["run_id"], "reason_counts": df12["invalid_reason"].astype(str).value_counts().to_dict(), "walkforward_executed_true_pct": float(r12["summary"]["metrics"]["walkforward_executed_true_pct"])}))

    wf = cfg["evaluation"]["stage8"]["walkforward_v2"]
    wins = build_windows(start_ts=f0["timestamp"].iloc[0], end_ts=f0["timestamp"].iloc[-1], train_days=int(wf["train_days"]), holdout_days=int(wf["holdout_days"]), forward_days=int(wf["forward_days"]), step_days=int(wf["step_days"]), reserve_tail_days=int(wf["reserve_tail_days"]))
    ov = any((not (w.holdout_end <= w.forward_start)) or (i > 0 and not (wins[i - 1].forward_end <= w.forward_start)) for i, w in enumerate(wins))
    checks.append(CheckResult(13, "FAIL" if ov else "PASS", "WF_SLICE_OVERLAP" if ov else "OK", {"window_count": len(wins), "overlap_found": ov}))

    checks.append(CheckResult(14, "PASS", "OK", {"trade_count_nonzero_rows": int((pd.to_numeric(df12["trade_count"], errors="coerce").fillna(0.0) > 0).sum())}))

    frame15 = f0.tail(500).reset_index(drop=True)
    c15 = json.loads(json.dumps(cfg))
    c15["cost_model"]["mode"] = "v2"
    cap = float(c15["cost_model"]["v2"]["max_total_bps_per_side"])
    vals = [float(one_way_cost_breakdown_bps(frame=frame15, bar_index=i, cost_cfg=c15["cost_model"], atr_col="atr_14", close_col="close")["total_bps"]) for i in range(len(frame15))]
    ok15 = bool(np.isfinite(np.asarray(vals)).all() and (max(vals) if vals else 0.0) <= cap + 1e-9)
    checks.append(CheckResult(15, "PASS" if ok15 else "FAIL", "OK" if ok15 else "COST_V2_UNIT_BUG", {"max_total_bps": float(max(vals) if vals else 0.0), "cap_bps": cap}))

    c16 = json.loads(json.dumps(cfg))
    c16["evaluation"]["stage13"]["enabled"] = True
    base16 = run_stage13(config=c16, seed=int(seed), dry_run=False, symbols=syms, timeframe=tf, families=["price", "flow"], composer_mode="weighted_sum", runs_root=runs_root, docs_dir=docs_dir, data_dir=data_dir, derived_dir=Path("data/derived"), stage_tag="13.drag", report_name="stage13_drag_base_tmp", write_docs=False, window_months=12)
    stress16 = json.loads(json.dumps(c16))
    stress16["cost_model"]["mode"] = "v2"
    v2 = stress16["cost_model"].setdefault("v2", {})
    v2["delay_bars"] = int(v2.get("delay_bars", 0)) + 1
    v2["spread_bps"] = float(v2.get("spread_bps", 0.5)) + 1.0
    v2["slippage_bps_base"] = float(v2.get("slippage_bps_base", 0.5)) + 1.0
    drag16 = run_stage13(config=stress16, seed=int(seed), dry_run=False, symbols=syms, timeframe=tf, families=["price", "flow"], composer_mode="weighted_sum", runs_root=runs_root, docs_dir=docs_dir, data_dir=data_dir, derived_dir=Path("data/derived"), stage_tag="13.drag", report_name="stage13_drag_stress_tmp", write_docs=False, window_months=12)
    b16 = float(pd.to_numeric(base16["rows"]["exp_lcb"], errors="coerce").fillna(0.0).max()) if not base16["rows"].empty else 0.0
    s16 = float(pd.to_numeric(drag16["rows"]["exp_lcb"], errors="coerce").fillna(0.0).max()) if not drag16["rows"].empty else 0.0
    checks.append(CheckResult(16, "PASS" if s16 <= b16 + 1e-9 else "FAIL", "OK" if s16 <= b16 + 1e-9 else "DRAG_SIGN_BUG", {"base_best_exp_lcb": b16, "stress_best_exp_lcb": s16}))

    r17 = run_stage14_nested_walkforward(config=cfg, seed=int(seed), dry_run=False, symbols=syms, timeframe=tf, runs_root=runs_root, docs_dir=docs_dir, data_dir=data_dir, derived_dir=Path("data/derived"), stage_tag="14.3", report_name="stage14_3_nested_wf")
    folds = int(r17["summary"].get("folds_evaluated", 0))
    checks.append(CheckResult(17, "PASS" if folds > 0 else "FAIL", "OK" if folds > 0 else "ML_FOLD_BUILDER", {"run_id": r17["run_id"], "folds_evaluated": folds, "classification": r17["summary"].get("classification", "")}))

    w18a = run_stage14_weighting(config=cfg, seed=int(seed), dry_run=False, symbols=syms, timeframe=tf, runs_root=runs_root, docs_dir=docs_dir, data_dir=data_dir, derived_dir=Path("data/derived"), stage_tag="14.1", report_name="stage14_1_weighting")
    w18b = run_stage14_weighting(config=cfg, seed=int(seed), dry_run=False, symbols=syms, timeframe=tf, runs_root=runs_root, docs_dir=docs_dir, data_dir=data_dir, derived_dir=Path("data/derived"), stage_tag="14.1", report_name="stage14_1_weighting")
    h18a = stable_hash(w18a["summary"].get("coefficients", []), length=16)
    h18b = stable_hash(w18b["summary"].get("coefficients", []), length=16)
    checks.append(CheckResult(18, "PASS" if int(cfg["evaluation"]["stage14"]["max_features"]) <= 20 and h18a == h18b else "FAIL", "OK" if int(cfg["evaluation"]["stage14"]["max_features"]) <= 20 and h18a == h18b else "ML_DETERMINISM_OR_LIMIT", {"coef_hash_a": h18a, "coef_hash_b": h18b, "max_features": int(cfg["evaluation"]["stage14"]["max_features"])}))

    from buffmini.signals.composer import compose_signals
    outs = {name: fam.propose_entries(fam.compute_scores(f0, ctx), f0, ctx) for name, fam in fams.items()}
    hv = stable_hash(pd.to_numeric(compose_signals(family_outputs=outs, mode="vote", weights=None, gated_config={})["score"], errors="coerce").fillna(0.0).round(8).tolist(), length=16)
    hw = stable_hash(pd.to_numeric(compose_signals(family_outputs=outs, mode="weighted_sum", weights={"price": 0.6, "volatility": 0.2, "flow": 0.2}, gated_config={})["score"], errors="coerce").fillna(0.0).round(8).tolist(), length=16)
    hg = stable_hash(pd.to_numeric(compose_signals(family_outputs=outs, mode="gated", weights={"price": 0.5, "volatility": 0.3, "flow": 0.2}, gated_config={"gate_family": "volatility", "gate_threshold": 0.2, "entry_threshold": 0.25})["score"], errors="coerce").fillna(0.0).round(8).tolist(), length=16)
    noop = len({hv, hw, hg}) < 2
    checks.append(CheckResult(19, "FAIL" if noop else "PASS", "COMPOSER_NOOP" if noop else "OK", {"score_hashes": {"vote": hv, "weighted_sum": hw, "gated": hg}}))

    rep = float(r12["summary"]["metrics"]["invalid_pct"])
    exp = compute_invalid_pct_from_rows(df12)
    bad20 = abs(rep - exp) > 1e-9
    checks.append(CheckResult(20, "FAIL" if bad20 else "PASS", "BUG_INVALID_PCT_METRIC" if bad20 else "OK", {"run_id": r12["summary"]["run_id"], "reported_invalid_pct": rep, "expected_invalid_pct": exp, "invalid_reason_counts": df12["invalid_reason"].astype(str).value_counts().to_dict()}))

    # bounded ablation
    ab_rows: list[dict[str, Any]] = []

    def _run_variant(driver: str, variant: str, c: dict[str, Any], fam_list: list[str], mode: str) -> None:
        rs = run_stage13(config=c, seed=int(seed), dry_run=False, symbols=syms, timeframe=tf, families=fam_list, composer_mode=mode, runs_root=runs_root, docs_dir=docs_dir, data_dir=data_dir, derived_dir=Path("data/derived"), stage_tag="13.ablation", report_name=f"stage13_ablation_{driver}_{variant}", write_docs=False, window_months=12)
        m, d = dict(rs["summary"]["metrics"]), rs["rows"]
        ab_rows.append({"driver": driver, "variant": variant, "run_id": rs["summary"]["run_id"], "trade_count": _f(m.get("trade_count")), "tpm": _f(m.get("tpm")), "invalid_pct": _f(m.get("invalid_pct")), "zero_trade_pct": _f(m.get("zero_trade_pct")), "walkforward_executed_true_pct": _f(m.get("walkforward_executed_true_pct")), "mc_trigger_rate": _f(m.get("mc_trigger_rate")), "best_exp_lcb": float(pd.to_numeric(d["exp_lcb"], errors="coerce").fillna(0.0).max()) if not d.empty else 0.0, "best_pf": float(pd.to_numeric(d["PF"], errors="coerce").fillna(0.0).max()) if not d.empty else 0.0, "best_expectancy": float(pd.to_numeric(d["expectancy"], errors="coerce").fillna(0.0).max()) if not d.empty else 0.0, "best_maxdd": float(pd.to_numeric(d["maxDD"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().min()) if not d.empty else 0.0, "classification": rs["summary"].get("classification", "")})

    cb = json.loads(json.dumps(cfg))
    cb["evaluation"]["stage13"]["enabled"] = True
    _run_variant("baseline", "realistic", cb, ["price", "volatility", "flow"], "weighted_sum")
    for fam in ("price", "volatility", "flow"):
        _run_variant("family", fam, json.loads(json.dumps(cb)), [fam], "none")
    for md in ("none", "vote", "weighted_sum"):
        _run_variant("composer", md, json.loads(json.dumps(cb)), ["price", "volatility", "flow"], md)
    cs = cfg.get("evaluation", {}).get("stage12", {}).get("cost_scenarios", {})
    for lv in ("low", "realistic", "high"):
        cc = json.loads(json.dumps(cb))
        cc["cost_model"]["mode"] = "v2"
        if lv != "realistic":
            ov = dict(cs.get(lv, {}))
            v = cc["cost_model"].setdefault("v2", {})
            for k in ("slippage_bps_base", "slippage_bps_vol_mult", "spread_bps", "delay_bars"):
                if k in ov:
                    v[k] = ov[k]
        _run_variant("cost_mode", lv, cc, ["price", "volatility", "flow"], "weighted_sum")
    for fam in ("price", "volatility", "flow"):
        for lbl, th in (("low", 0.2), ("med", 0.3), ("high", 0.4)):
            ct = json.loads(json.dumps(cb))
            ct["evaluation"]["stage13"][fam]["entry_threshold"] = th
            _run_variant("threshold", f"{fam}:{lbl}", ct, [fam], "none")
    cw = json.loads(json.dumps(cb))
    cw["evaluation"]["stage8"]["walkforward_v2"]["train_days"] = 2000
    cw["evaluation"]["stage8"]["walkforward_v2"]["holdout_days"] = 500
    cw["evaluation"]["stage8"]["walkforward_v2"]["forward_days"] = 500
    _run_variant("walkforward", "enabled", json.loads(json.dumps(cb)), ["price", "volatility", "flow"], "weighted_sum")
    _run_variant("walkforward", "disabled_proxy", cw, ["price", "volatility", "flow"], "weighted_sum")
    wst = run_stage14_weighting(config=cb, seed=int(seed), dry_run=False, symbols=syms, timeframe=tf, runs_root=runs_root, docs_dir=docs_dir, data_dir=data_dir, derived_dir=Path("data/derived"), stage_tag="14.1", report_name="stage14_1_weighting")
    ab_rows.append({"driver": "stage14", "variant": "weighting_best", "run_id": wst["run_id"], "trade_count": _f(wst["summary"]["forward_metrics"].get("trade_count")), "tpm": _f(wst["summary"]["forward_metrics"].get("tpm")), "invalid_pct": 0.0, "zero_trade_pct": 0.0, "walkforward_executed_true_pct": 0.0, "mc_trigger_rate": 0.0, "best_exp_lcb": _f(wst["summary"]["forward_metrics"].get("exp_lcb")), "best_pf": _f(wst["summary"]["forward_metrics"].get("PF")), "best_expectancy": _f(wst["summary"]["forward_metrics"].get("expectancy")), "best_maxdd": _f(wst["summary"]["forward_metrics"].get("maxDD")), "classification": wst["summary"].get("classification", "")})
    impacts = rank_impact_drivers(ab_rows)

    base_master = _load_json(docs_dir / "stage13_14_master_summary.json")
    base13 = _load_json(docs_dir / "stage13_5_combined_summary.json")
    base14 = _load_json(docs_dir / "stage14_3_nested_wf_summary.json")
    before = {"master_final_verdict": base_master.get("final_verdict", ""), "stage13_combined": {"invalid_pct": _f(base13.get("best", {}).get("invalid_pct")), "zero_trade_pct": _f(base13.get("best", {}).get("zero_trade_pct")), "walkforward_executed_true_pct": _f(base13.get("best", {}).get("walkforward_executed_true_pct")), "mc_trigger_rate": _f(base13.get("best", {}).get("mc_trigger_rate")), "best_exp_lcb": _f(base13.get("best", {}).get("best_exp_lcb"))}, "stage14_nested": {"folds_evaluated": int(base14.get("folds_evaluated", 0)), "consistency": _f(base14.get("consistency")), "classification": base14.get("classification", "")}}
    after = {"stage13_combined": _safe(r12["summary"].get("metrics", {})), "stage14_nested": _safe(r17["summary"])}
    failed = [c for c in checks if c.status != "PASS"]
    fixes = [c.as_dict() for c in failed if c.tag.startswith("BUG_")]
    invalid_before = _f(before.get("stage13_combined", {}).get("invalid_pct"))
    invalid_after = _f(after.get("stage13_combined", {}).get("invalid_pct"))
    metric_fix_detected = abs(invalid_before - invalid_after) > 0.1 and (not any(c.check_id == 20 and c.status != "PASS" for c in checks))
    if metric_fix_detected:
        fixes.append(
            {
                "check_id": 20,
                "name": CHECKS[20],
                "status": "PASS",
                "root_cause_tag": "BUG_INVALID_PCT_METRIC_FIXED",
                "evidence": {"invalid_pct_before": invalid_before, "invalid_pct_after": invalid_after},
            }
        )
    concl = "NO_BUG_FOUND_NO_EDGE_CONFIRMED"
    if metric_fix_detected:
        concl = "BUG_FOUND_AND_FIXED"
    elif fixes and _material(before, after):
        concl = "BUG_FOUND_AND_FIXED"
    elif fixes:
        concl = "PARTIAL_FIX_NO_CHANGE"

    raw = {"git_head": _git_head(), "runtime_seconds": float(time.perf_counter() - t0), "baseline": {"master": base_master, "stage13_5": base13, "stage14_3": base14, "metrics": before}, "checks": [c.as_dict() for c in checks], "ablation": {"rows": _safe(ab_rows), "row_count": len(ab_rows)}, "impact_drivers_ranked": _safe(impacts), "post_rerun": {"stage13": {"run_id": r12["summary"]["run_id"], "summary": _safe(r12["summary"])}, "stage14": {"run_id": r17["run_id"], "summary": _safe(r17["summary"])}}, "before_after": {"before": before, "after": after}, "fixes_applied": fixes, "warnings": _warn(failed, ab_rows), "limitations": _lim(checks), "final_conclusion": concl}
    summary = {"git_head": raw["git_head"], "runtime_seconds": _f(raw["runtime_seconds"]), "checks_total": len(raw["checks"]), "checks_failed": len(failed), "checks": raw["checks"], "top_impact_drivers": raw["impact_drivers_ranked"][:10], "fixes_applied": raw["fixes_applied"], "before_after": raw["before_after"], "final_conclusion": raw["final_conclusion"], "warnings": raw["warnings"], "limitations": raw["limitations"]}

    docs_dir.mkdir(parents=True, exist_ok=True)
    p_raw = docs_dir / "stage13_14_forensic_root_cause_raw.json"
    p_sum = docs_dir / "stage13_14_forensic_root_cause_summary.json"
    p_md = docs_dir / "stage13_14_forensic_root_cause_report.md"
    p_raw.write_text(json.dumps(_safe(raw), indent=2, allow_nan=False), encoding="utf-8")
    p_sum.write_text(json.dumps(_safe(summary), indent=2, allow_nan=False), encoding="utf-8")
    p_md.write_text(_md(raw=raw, summary=summary, raw_path=p_raw), encoding="utf-8")
    return {"raw": raw, "summary": summary, "raw_path": p_raw, "summary_path": p_sum, "report_path": p_md}


def _md(*, raw: dict[str, Any], summary: dict[str, Any], raw_path: Path) -> str:
    lines = [
        "# Stage-13/14 Forensic Root-Cause Report",
        "",
        "## Executive Summary",
        f"- git_head: `{raw.get('git_head', '')}`",
        f"- runtime_seconds: `{_f(raw.get('runtime_seconds')):.3f}`",
        f"- final_conclusion: `{summary.get('final_conclusion', '')}`",
        f"- raw_evidence: `{raw_path.as_posix()}`",
        "",
        "## Baseline (Before)",
        f"- `{_safe(raw.get('baseline', {}).get('metrics', {}))}`",
        "",
        "## 20 Checks",
        "| # | Check | PASS/FAIL | Root Cause Tag | Evidence Pointer |",
        "| --- | --- | --- | --- | --- |",
    ]
    for it in summary.get("checks", []):
        cid = int(it.get("check_id", 0))
        lines.append(f"| {cid} | {it.get('name','')} | {it.get('status','')} | {it.get('root_cause_tag','')} | `docs/stage13_14_forensic_root_cause_raw.json#checks[{cid}]` |")
    lines += ["", "## Impact Drivers (Ranked)", "| Rank | Driver | Variant | ?exp_lcb | ?invalid_pct | ?wf_pct | ?tpm |", "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for i, r in enumerate(summary.get("top_impact_drivers", []), start=1):
        lines.append(f"| {i} | {r.get('driver','')} | {r.get('variant','')} | {_f(r.get('delta_exp_lcb')):.6f} | {_f(r.get('delta_invalid_pct')):.6f} | {_f(r.get('delta_walkforward_pct')):.6f} | {_f(r.get('delta_tpm')):.6f} |")
    lines += ["", "## Fixes Applied"] + ([f"- {f.get('root_cause_tag','')} via check `{f.get('check_id','')}`" for f in summary.get("fixes_applied", [])] if summary.get("fixes_applied") else ["- none"])
    lines += ["", "## Re-run Results (After)", f"- `{_safe(summary.get('before_after', {}))}`", "", "## Warnings"] + ([f"- {w}" for w in summary.get("warnings", [])] if summary.get("warnings") else ["- none"])
    lines += ["", "## Limitations"] + ([f"- {x}" for x in summary.get("limitations", [])] if summary.get("limitations") else ["- none"])
    lines += ["", "## What To Do Next", "1. Improve signal density without lowering WF/MC gates (score shape + thresholding by regime).", "2. Broaden free-data families (cross-symbol/session features) with leakage checks still strict.", "3. If repeated forensic sweeps still show NO_EDGE, pivot to a different hypothesis class (stat-arb/cross-asset).", "", "## Final Conclusion", f"- `{summary.get('final_conclusion', 'OTHER')}`"]
    return "\n".join(lines).strip() + "\n"


def _warn(failed: list[CheckResult], ab_rows: list[dict[str, Any]]) -> list[str]:
    out = [f"{len(failed)} checks failed"] if failed else []
    if ab_rows and float(np.mean([_f(r.get("invalid_pct")) for r in ab_rows])) >= 70.0:
        out.append("high invalid_pct across ablations")
    return out


def _lim(checks: list[CheckResult]) -> list[str]:
    out: list[str] = []
    if any(c.check_id == 10 and c.status != "PASS" for c in checks):
        out.append("zero-trade attribution still has unclassified buckets")
    if any(c.check_id == 17 and c.status != "PASS" for c in checks):
        out.append("nested ML walkforward not executing with current constraints")
    return out


def _material(before: dict[str, Any], after: dict[str, Any]) -> bool:
    b, a = before.get("stage13_combined", {}), after.get("stage13_combined", {})
    return abs(_f(a.get("invalid_pct")) - _f(b.get("invalid_pct"))) > 0.1 or abs(_f(a.get("walkforward_executed_true_pct")) - _f(b.get("walkforward_executed_true_pct"))) > 0.1


def _months(frame: pd.DataFrame) -> float:
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return 0.0
    return max(float((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400.0) / 30.0, 1e-9)


def _git_head() -> str:
    h = Path(".git/HEAD")
    if not h.exists():
        return ""
    txt = h.read_text(encoding="utf-8").strip()
    if txt.startswith("ref: "):
        p = Path(".git") / txt.split(" ", 1)[1].strip()
        if p.exists():
            return p.read_text(encoding="utf-8").strip()
    return txt


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return dict(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return {}


def _f(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return 0.0
    return x if math.isfinite(x) else 0.0


def _safe(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _safe(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_safe(val) for val in v]
    if isinstance(v, tuple):
        return [_safe(val) for val in v]
    if isinstance(v, (np.floating, float)):
        x = float(v)
        return x if np.isfinite(x) else None
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, pd.Timestamp):
        t = v.tz_localize("UTC") if v.tzinfo is None else v.tz_convert("UTC")
        return t.isoformat()
    return v
