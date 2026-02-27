"""Stage-5.6 Pine Script exporter."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from buffmini.constants import RUNS_DIR
from buffmini.discovery.generator import Candidate, candidate_to_strategy_spec


@dataclass(frozen=True)
class PineComponent:
    candidate_id: str
    strategy_family: str
    strategy_name: str
    gating_mode: str
    exit_mode: str
    parameters: dict[str, Any]
    weight: float


@dataclass(frozen=True)
class PineContext:
    run_id: str
    run_dir: Path
    stage1_run_id: str
    stage2_run_id: str
    stage3_3_run_id: str | None
    chosen_method: str
    chosen_leverage: float
    timeframe: str
    execution_mode: str
    config_hash: str
    data_hash: str
    seed: int
    round_trip_cost_pct: float
    slippage_pct: float
    summary_ui: dict[str, Any]
    components: list[PineComponent]


def export_pine_scripts(run_id: str, runs_dir: Path = RUNS_DIR) -> dict[str, Any]:
    """Export component Pine scripts and portfolio template for one run."""

    ctx = _resolve_context(run_id=run_id, runs_dir=Path(runs_dir))
    summary_ui = _ensure_summary_snapshot(ctx)
    out_dir = ctx.run_dir / "exports" / "pine"
    out_dir.mkdir(parents=True, exist_ok=True)

    files: list[dict[str, Any]] = []
    comp_hash: dict[str, str] = {}
    for component in ctx.components:
        text = render_component_pine(ctx, component)
        name = f"{_safe_name(component.candidate_id)}.pine.txt"
        (out_dir / name).write_text(text, encoding="utf-8")
        h = _sha256_text(text)
        comp_hash[component.candidate_id] = h
        files.append({"component_id": component.candidate_id, "path": name, "sha256": h})

    portfolio_text = render_portfolio_template_pine(ctx)
    portfolio_name = "portfolio_template.pine.txt"
    (out_dir / portfolio_name).write_text(portfolio_text, encoding="utf-8")
    portfolio_hash = _sha256_text(portfolio_text)

    validation = validate_pine_exports(out_dir=out_dir, run_id=ctx.run_id, summary_ui=summary_ui, components=ctx.components)
    deterministic = _deterministic_check(ctx=ctx, component_hashes=comp_hash, portfolio_hash=portfolio_hash)

    index = {
        "run_id": ctx.run_id,
        "stage1_run_id": ctx.stage1_run_id,
        "stage2_run_id": ctx.stage2_run_id,
        "stage3_3_run_id": ctx.stage3_3_run_id,
        "chosen_method": ctx.chosen_method,
        "chosen_leverage": ctx.chosen_leverage,
        "timeframe": ctx.timeframe,
        "execution_mode": ctx.execution_mode,
        "config_hash": ctx.config_hash,
        "data_hash": ctx.data_hash,
        "seed": ctx.seed,
        "round_trip_cost_pct": ctx.round_trip_cost_pct,
        "slippage_pct": ctx.slippage_pct,
        "files": files + [{"path": portfolio_name, "sha256": portfolio_hash, "type": "portfolio_template"}],
        "component_count": len(ctx.components),
        "validation": validation,
        "deterministic_export": deterministic,
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=True), encoding="utf-8")
    return index


def _ensure_summary_snapshot(ctx: PineContext) -> dict[str, Any]:
    summary_path = ctx.run_dir / "ui_bundle" / "summary_ui.json"
    snapshot = dict(ctx.summary_ui)
    if isinstance(snapshot.get("selected_components"), list):
        return snapshot
    components = []
    for comp in ctx.components:
        components.append(
            {
                "candidate_id": comp.candidate_id,
                "strategy_family": comp.strategy_family,
                "gating_mode": comp.gating_mode,
                "exit_mode": comp.exit_mode,
                "weight": comp.weight,
                "parameters": comp.parameters,
            }
        )
    snapshot["selected_components"] = components
    if summary_path.exists():
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=True), encoding="utf-8")
    return snapshot


def _resolve_context(run_id: str, runs_dir: Path) -> PineContext:
    run_dir = runs_dir / run_id
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    pipeline_summary = _safe_json(run_dir / "pipeline_summary.json")
    summary_ui = _safe_json(run_dir / "ui_bundle" / "summary_ui.json")
    selector_summary_local = _safe_json(run_dir / "selector_summary.json")
    stage2_summary_local = _safe_json(run_dir / "portfolio_summary.json")

    stage2_run_id = (
        str(stage2_summary_local.get("run_id", "")).strip()
        or str(pipeline_summary.get("stage2_run_id", "")).strip()
        or str(selector_summary_local.get("stage2_run_id", "")).strip()
    )
    if not stage2_run_id and run_id.endswith("_stage2"):
        stage2_run_id = run_id
    if not stage2_run_id:
        raise ValueError("Could not resolve stage2_run_id")
    stage2_summary = stage2_summary_local or _safe_json(runs_dir / stage2_run_id / "portfolio_summary.json")
    if not stage2_summary:
        raise FileNotFoundError(f"Missing Stage-2 summary for {stage2_run_id}")

    stage1_run_id = str(stage2_summary.get("stage1_run_id", "")).strip()
    if not stage1_run_id:
        raise ValueError("stage1_run_id missing in Stage-2 summary")

    stage3_3_run_id = (
        str(selector_summary_local.get("run_id", "")).strip()
        or str(pipeline_summary.get("stage3_3_run_id", "")).strip()
    ) or None
    if stage3_3_run_id and stage3_3_run_id != run_id:
        selector_summary = _safe_json(runs_dir / stage3_3_run_id / "selector_summary.json")
    else:
        selector_summary = selector_summary_local

    chosen_method = str(
        (selector_summary.get("overall_choice") or {}).get("method")
        or summary_ui.get("chosen_method")
        or pipeline_summary.get("chosen_method")
        or "equal"
    )
    chosen_leverage = float(
        (selector_summary.get("overall_choice") or {}).get("chosen_leverage")
        or summary_ui.get("chosen_leverage")
        or pipeline_summary.get("chosen_leverage")
        or 1.0
    )

    method_payload = (stage2_summary.get("portfolio_methods") or {}).get(chosen_method)
    if not isinstance(method_payload, dict):
        first_method = next(iter((stage2_summary.get("portfolio_methods") or {}).keys()), None)
        if first_method is None:
            raise ValueError("No portfolio_methods in Stage-2 summary")
        chosen_method = str(first_method)
        method_payload = (stage2_summary.get("portfolio_methods") or {}).get(chosen_method, {})

    weights = {str(k): float(v) for k, v in (method_payload.get("weights") or {}).items()}
    candidate_ids = [str(v) for v in (method_payload.get("selected_candidates") or list(weights.keys()))]
    if not candidate_ids:
        raise ValueError("No selected candidates in Stage-2 method payload")

    candidate_lookup = _load_candidate_lookup(runs_dir / stage1_run_id / "candidates")
    components: list[PineComponent] = []
    for cid in candidate_ids:
        payload = candidate_lookup.get(cid)
        if payload is None:
            continue
        components.append(_build_component(payload, weight=float(weights.get(cid, 0.0))))
    if not components:
        raise ValueError("No candidate payloads found for selected IDs")

    stage1_cfg = _safe_yaml(runs_dir / stage1_run_id / "config.yaml")
    timeframe = str(summary_ui.get("timeframe") or (stage1_cfg.get("universe") or {}).get("timeframe") or "1h")
    execution_mode = str(summary_ui.get("execution_mode") or "net")
    seed = int(summary_ui.get("seed") or (stage1_cfg.get("search") or {}).get("seed", 42))
    round_trip_cost_pct = float(stage2_summary.get("round_trip_cost_pct", (stage1_cfg.get("costs") or {}).get("round_trip_cost_pct", 0.1)))
    slippage_pct = float(stage2_summary.get("slippage_pct", (stage1_cfg.get("costs") or {}).get("slippage_pct", 0.0005)))

    return PineContext(
        run_id=run_id,
        run_dir=run_dir,
        stage1_run_id=stage1_run_id,
        stage2_run_id=stage2_run_id,
        stage3_3_run_id=stage3_3_run_id,
        chosen_method=chosen_method,
        chosen_leverage=chosen_leverage,
        timeframe=timeframe,
        execution_mode=execution_mode,
        config_hash=str(summary_ui.get("config_hash") or pipeline_summary.get("config_hash") or ""),
        data_hash=str(summary_ui.get("data_hash") or pipeline_summary.get("data_hash") or ""),
        seed=seed,
        round_trip_cost_pct=round_trip_cost_pct,
        slippage_pct=slippage_pct,
        summary_ui=summary_ui,
        components=components,
    )


def _build_component(payload: dict[str, Any], weight: float) -> PineComponent:
    candidate = Candidate(
        candidate_id=str(payload["candidate_id"]),
        family=str(payload["strategy_family"]),
        gating_mode=str(payload.get("gating", "none")),
        exit_mode=str(payload.get("exit_mode", "fixed_atr")),
        params=dict(payload.get("parameters", {})),
    )
    spec = candidate_to_strategy_spec(candidate)
    regime = spec.parameters.get("regime_gate", {"long": False, "short": False})
    p = candidate.params
    normalized = {
        "channel_period": int(p.get("channel_period", spec.parameters.get("channel_period", 20))),
        "ema_fast": int(p.get("ema_fast", spec.parameters.get("ema_fast", 50))),
        "ema_slow": int(p.get("ema_slow", spec.parameters.get("ema_slow", 200))),
        "signal_ema": int(spec.parameters.get("signal_ema", p.get("signal_ema", 20))),
        "rsi_period": int(spec.parameters.get("rsi_period", 14)),
        "rsi_long_entry": float(p.get("rsi_long_entry", spec.parameters.get("rsi_long_entry", 30))),
        "rsi_short_entry": float(p.get("rsi_short_entry", spec.parameters.get("rsi_short_entry", 70))),
        "bollinger_period": int(p.get("bollinger_period", spec.parameters.get("bollinger_period", 20))),
        "bollinger_std": float(p.get("bollinger_std", spec.parameters.get("bollinger_std", 2.0))),
        "atr_sl_multiplier": float(p.get("atr_sl_multiplier", 1.5)),
        "atr_tp_multiplier": float(p.get("atr_tp_multiplier", 3.0)),
        "trailing_atr_k": float(p.get("trailing_atr_k", 1.5)),
        "max_holding_bars": int(p.get("max_holding_bars", 24)),
        "regime_gate_long": bool(p.get("regime_gate_long", regime.get("long", False))),
        "regime_gate_short": bool(p.get("regime_gate_short", regime.get("short", False))),
    }
    return PineComponent(
        candidate_id=candidate.candidate_id,
        strategy_family=candidate.family,
        strategy_name=spec.name,
        gating_mode=candidate.gating_mode,
        exit_mode=candidate.exit_mode,
        parameters=normalized,
        weight=weight,
    )


def render_component_pine(ctx: PineContext, component: PineComponent) -> str:
    p = component.parameters
    use_vol = component.gating_mode in {"vol", "vol+regime"}
    use_regime = component.gating_mode == "vol+regime"
    one_way = max(0.0, ctx.round_trip_cost_pct / 2.0)
    lines = [
        "// AUTO-GENERATED by Buff-mini Stage-5.6 Pine export",
        f"// run_id: {ctx.run_id}",
        f"// component_id: {component.candidate_id}",
        f"// timeframe: {ctx.timeframe}",
        "// caveats: stop-first and fee/slippage semantics are approximated in TradingView.",
        "//@version=5",
        f"strategy(\"Buff-mini {component.strategy_name} [{component.candidate_id}]\", overlay=true, pyramiding=0, process_orders_on_close=true, initial_capital=10000, commission_type=strategy.commission.percent, commission_value={one_way:.10f})",
        f"runId = input.string(\"{ctx.run_id}\", \"Run ID\")",
        f"componentId = input.string(\"{component.candidate_id}\", \"Component ID\")",
        "channelPeriod = input.int(" + str(p["channel_period"]) + ", \"Donchian Period\", minval=2)",
        "emaFast = input.int(" + str(p["ema_fast"]) + ", \"EMA Fast\", minval=2)",
        "emaSlow = input.int(" + str(p["ema_slow"]) + ", \"EMA Slow\", minval=2)",
        "signalEma = input.int(" + str(p["signal_ema"]) + ", \"Signal EMA\", minval=2)",
        "rsiPeriod = input.int(" + str(p["rsi_period"]) + ", \"RSI Period\", minval=2)",
        f"rsiLongEntry = input.float({float(p['rsi_long_entry']):.10f}, \"RSI Long Entry\")",
        f"rsiShortEntry = input.float({float(p['rsi_short_entry']):.10f}, \"RSI Short Entry\")",
        "bbPeriod = input.int(" + str(p["bollinger_period"]) + ", \"Bollinger Period\", minval=2)",
        f"bbStdMult = input.float({float(p['bollinger_std']):.10f}, \"Bollinger Std\")",
        f"atrSlMult = input.float({float(p['atr_sl_multiplier']):.10f}, \"ATR Stop Multiplier\")",
        f"atrTpMult = input.float({float(p['atr_tp_multiplier']):.10f}, \"ATR TP Multiplier\")",
        f"trailingAtrK = input.float({float(p['trailing_atr_k']):.10f}, \"Trailing ATR K\")",
        f"maxHoldingBars = input.int({int(p['max_holding_bars'])}, \"Max Holding Bars\", minval=1)",
        f"exitMode = input.string(\"{component.exit_mode}\", \"Exit Mode\", options=[\"fixed_atr\",\"breakeven_1r\",\"trailing_atr\",\"partial_then_trail\"])",
        f"useVolGate = input.bool({'true' if use_vol else 'false'}, \"Use Volatility Gate\")",
        f"useRegimeGate = input.bool({'true' if use_regime else 'false'}, \"Use Regime Gate\")",
        f"regimeGateLong = input.bool({'true' if p['regime_gate_long'] else 'false'}, \"Regime Long\")",
        f"regimeGateShort = input.bool({'true' if p['regime_gate_short'] else 'false'}, \"Regime Short\")",
        "ema50 = ta.ema(close, 50)",
        "ema200 = ta.ema(close, 200)",
        "emaFastVal = ta.ema(close, emaFast)",
        "emaSlowVal = ta.ema(close, emaSlow)",
        "signalEmaVal = ta.ema(close, signalEma)",
        "rsiVal = ta.rsi(close, rsiPeriod)",
        "atr14 = ta.atr(14)",
        "atr14Sma50 = ta.sma(atr14, 50)",
        "bbBasis = ta.sma(close, bbPeriod)",
        "bbDev = ta.stdev(close, bbPeriod)",
        "bbUpper = bbBasis + bbStdMult * bbDev",
        "bbLower = bbBasis - bbStdMult * bbDev",
        "donchianHigh = ta.highest(high, channelPeriod)[1]",
        "donchianLow = ta.lowest(low, channelPeriod)[1]",
    ]
    lines.extend(_family_lines(component.strategy_family))
    lines.extend(
        [
            "volGateOk = not useVolGate or (atr14 > atr14Sma50)",
            "regimeLongOk = (not useRegimeGate) or (not regimeGateLong) or (ema50 > ema200)",
            "regimeShortOk = (not useRegimeGate) or (not regimeGateShort) or (ema50 < ema200)",
            "longSignal = (rawLong and volGateOk and regimeLongOk)[1]",
            "shortSignal = (rawShort and volGateOk and regimeShortOk)[1]",
            "if strategy.position_size == 0",
            "    if longSignal and not shortSignal",
            "        strategy.entry(\"L\", strategy.long)",
            "    else if shortSignal and not longSignal",
            "        strategy.entry(\"S\", strategy.short)",
            "if strategy.position_size > 0",
            "    stopPrice = strategy.position_avg_price - atrSlMult * atr14",
            "    takePrice = strategy.position_avg_price + atrTpMult * atr14",
            "    stopHit = low <= stopPrice",
            "    tpHit = high >= takePrice",
            "    if stopHit",
            "        strategy.exit(\"L_STOP\", from_entry=\"L\", stop=stopPrice, comment=\"stop_loss\")",
            "    else if (exitMode == \"fixed_atr\" or exitMode == \"breakeven_1r\") and tpHit",
            "        strategy.exit(\"L_TP\", from_entry=\"L\", limit=takePrice, comment=\"take_profit\")",
            "if strategy.position_size < 0",
            "    stopPrice = strategy.position_avg_price + atrSlMult * atr14",
            "    takePrice = strategy.position_avg_price - atrTpMult * atr14",
            "    stopHit = high >= stopPrice",
            "    tpHit = low <= takePrice",
            "    if stopHit",
            "        strategy.exit(\"S_STOP\", from_entry=\"S\", stop=stopPrice, comment=\"stop_loss\")",
            "    else if (exitMode == \"fixed_atr\" or exitMode == \"breakeven_1r\") and tpHit",
            "        strategy.exit(\"S_TP\", from_entry=\"S\", limit=takePrice, comment=\"take_profit\")",
        ]
    )
    return "\n".join(lines) + "\n"


def render_portfolio_template_pine(ctx: PineContext) -> str:
    lines = [
        "// AUTO-GENERATED by Buff-mini Stage-5.6 Pine export",
        f"// run_id: {ctx.run_id}",
        "// WARNING: portfolio mode is visual approximation; internal multi-component engine semantics differ.",
        "//@version=5",
        f"strategy(\"Buff-mini Portfolio Template [{ctx.run_id}]\", overlay=true, pyramiding=20, process_orders_on_close=true, initial_capital=10000)",
        f"runId = input.string(\"{ctx.run_id}\", \"Run ID\")",
        f"executionApprox = input.string(\"{ctx.execution_mode}\", \"Execution Approx\", options=[\"net\",\"hedge\"])",
        "entryThreshold = input.float(0.0, \"Entry Threshold\")",
        "longScore = 0.0",
        "shortScore = 0.0",
    ]
    for idx, comp in enumerate(ctx.components, start=1):
        pref = f"c{idx}"
        lines.extend(
            [
                f"{pref}Weight = input.float({float(comp.weight):.10f}, \"{pref} Weight\", minval=0.0)",
                f"{pref}Long = input.bool(false, \"{pref} long signal placeholder\")",
                f"{pref}Short = input.bool(false, \"{pref} short signal placeholder\")",
                f"longScore += {pref}Long ? {pref}Weight : 0.0",
                f"shortScore += {pref}Short ? {pref}Weight : 0.0",
            ]
        )
    lines.extend(
        [
            "netScore = longScore - shortScore",
            "if strategy.position_size == 0",
            "    if executionApprox == \"net\"",
            "        if netScore > entryThreshold",
            "            strategy.entry(\"NET_L\", strategy.long)",
            "        else if netScore < -entryThreshold",
            "            strategy.entry(\"NET_S\", strategy.short)",
            "    else",
            "        if longScore > shortScore and longScore > entryThreshold",
            "            strategy.entry(\"HEDGE_L\", strategy.long)",
            "        else if shortScore > longScore and shortScore > entryThreshold",
            "            strategy.entry(\"HEDGE_S\", strategy.short)",
            "plot(netScore, title=\"netScore\")",
        ]
    )
    return "\n".join(lines) + "\n"


def validate_pine_exports(out_dir: Path, run_id: str, summary_ui: dict[str, Any], components: list[PineComponent]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for p in sorted(out_dir.glob("*.pine.txt")):
        text = p.read_text(encoding="utf-8")
        checks.append(
            {
                "file": p.name,
                "has_version": "//@version=5" in text,
                "has_strategy": "strategy(" in text,
                "has_input": "input." in text,
                "has_run_id": f"run_id: {run_id}" in text or f"runId = input.string(\"{run_id}\"" in text,
                "lookahead_safe": "lookahead_on" not in text.lower() and re.search(r"\[-\d+\]", text) is None,
            }
        )
    snapshot = summary_ui.get("selected_components")
    by_id = {str(item.get("candidate_id")): item for item in snapshot if isinstance(item, dict)} if isinstance(snapshot, list) else {}
    param_rows: list[dict[str, Any]] = []
    for comp in components:
        item = by_id.get(comp.candidate_id)
        if not item or not isinstance(item.get("parameters"), dict):
            param_rows.append({"component_id": comp.candidate_id, "status": "skipped", "reason": "summary_ui has no selected_components snapshot"})
            continue
        ok = _param_subset_match(comp.parameters, dict(item["parameters"]))
        param_rows.append({"component_id": comp.candidate_id, "status": "pass" if ok else "fail"})
    return {"file_checks": checks, "parameter_validation": param_rows, "all_files_valid": all(all(v for k, v in r.items() if k not in {"file"}) for r in checks)}


def _deterministic_check(ctx: PineContext, component_hashes: dict[str, str], portfolio_hash: str) -> bool:
    for comp in ctx.components:
        if _sha256_text(render_component_pine(ctx, comp)) != component_hashes.get(comp.candidate_id):
            return False
    return _sha256_text(render_portfolio_template_pine(ctx)) == portfolio_hash


def _family_lines(family: str) -> list[str]:
    if family == "DonchianBreakout":
        return ["rawLong = close > donchianHigh", "rawShort = close < donchianLow"]
    if family == "RSIMeanReversion":
        return ["rawLong = rsiVal < rsiLongEntry", "rawShort = rsiVal > rsiShortEntry"]
    if family == "TrendPullback":
        return ["rawLong = emaFastVal > emaSlowVal and close > signalEmaVal and rsiVal < rsiLongEntry", "rawShort = emaFastVal < emaSlowVal and close < signalEmaVal and rsiVal > rsiShortEntry"]
    if family == "BollingerMeanReversion":
        return ["rawLong = close < bbLower and rsiVal < rsiLongEntry", "rawShort = close > bbUpper and rsiVal > rsiShortEntry"]
    if family == "RangeBreakoutTrendFilter":
        return ["rawLong = close > donchianHigh and emaFastVal > emaSlowVal", "rawShort = close < donchianLow and emaFastVal < emaSlowVal"]
    raise ValueError(f"Unsupported strategy family: {family}")


def _load_candidate_lookup(candidates_dir: Path) -> dict[str, dict[str, Any]]:
    if not candidates_dir.exists():
        raise FileNotFoundError(candidates_dir)
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(candidates_dir.glob("strategy_*.json")):
        payload = _safe_json(path)
        cid = str(payload.get("candidate_id", "")).strip()
        if cid:
            out[cid] = payload
    return out


def _param_subset_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
    keys = sorted(set(a.keys()).intersection(set(b.keys())))
    if not keys:
        return False
    for key in keys:
        va = a[key]
        vb = b[key]
        if isinstance(va, float) or isinstance(vb, float):
            if abs(float(va) - float(vb)) > 1e-9:
                return False
        elif va != vb:
            return False
    return True


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._")
    return cleaned or "component"


def _sha256_text(text: str) -> str:
    import hashlib

    return hashlib.sha256(text.encode("utf-8")).hexdigest()
