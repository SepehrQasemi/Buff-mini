"""Stage-15 A/B runner: Classic vs alpha-v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from buffmini.alpha_v2.contracts import AlphaRole, ClassicFamilyAdapter
from buffmini.alpha_v2.orchestrator import OrchestratorConfig, run_orchestrator
from buffmini.backtest.engine import run_backtest
from buffmini.baselines.stage0 import generate_signals, trend_pullback
from buffmini.config import compute_config_hash
from buffmini.constants import DERIVED_DATA_DIR, RAW_DATA_DIR, RUNS_DIR
from buffmini.signals.registry import build_families
from buffmini.stage10.evaluate import _build_features
from buffmini.utils.hashing import stable_hash
from buffmini.utils.time import utc_now_compact


def run_ab_compare(
    *,
    config: dict[str, Any],
    seed: int = 42,
    dry_run: bool = True,
    symbols: list[str] | None = None,
    timeframe: str = "1h",
    runs_root: Path = RUNS_DIR,
    data_dir: Path = RAW_DATA_DIR,
    derived_dir: Path = DERIVED_DATA_DIR,
    alpha_enabled: bool = True,
    max_rows: int = 1200,
) -> dict[str, Any]:
    cfg = json.loads(json.dumps(config))
    syms = list(symbols or cfg.get("universe", {}).get("symbols", ["BTC/USDT", "ETH/USDT"]))
    frames = _build_features(
        config=cfg,
        symbols=syms,
        timeframe=str(timeframe),
        dry_run=bool(dry_run),
        seed=int(seed),
        data_dir=data_dir,
        derived_dir=derived_dir,
    )
    if not frames:
        raise RuntimeError("ab_runner: no features")
    if max_rows > 0:
        frames = {symbol: frame.tail(int(max_rows)).reset_index(drop=True) for symbol, frame in frames.items()}

    data_hash = stable_hash(
        {
            symbol: stable_hash(
                frame.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]].to_dict(orient="list"),
                length=16,
            )
            for symbol, frame in sorted(frames.items())
        },
        length=16,
    )
    resolved_end_ts = max(pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").max() for frame in frames.values()).isoformat()

    classic_rows = [_run_classic_for_symbol(frame=frame, symbol=symbol, cfg=cfg) for symbol, frame in sorted(frames.items())]
    classic = _aggregate(rows=classic_rows)

    if not alpha_enabled:
        alpha = dict(classic)
        activation_stats = {
            "contracts_evaluated": 0,
            "pct_not_neutral_multiplier": 0.0,
            "pct_nonzero_confidence": 0.0,
        }
    else:
        alpha_rows: list[dict[str, Any]] = []
        act_rows: list[dict[str, float]] = []
        for symbol, frame in sorted(frames.items()):
            alpha_row, act = _run_alpha_for_symbol(frame=frame, symbol=symbol, cfg=cfg, seed=int(seed))
            alpha_rows.append(alpha_row)
            act_rows.append(act)
        alpha = _aggregate(rows=alpha_rows)
        activation_stats = {
            "contracts_evaluated": int(sum(item["contracts_evaluated"] for item in act_rows)),
            "pct_not_neutral_multiplier": float(np.mean([item["pct_not_neutral_multiplier"] for item in act_rows])),
            "pct_nonzero_confidence": float(np.mean([item["pct_nonzero_confidence"] for item in act_rows])),
        }

    payload = {
        "stage": "15",
        "run_id": "",
        "seed": int(seed),
        "config_hash": compute_config_hash(cfg),
        "data_hash": data_hash,
        "resolved_end_ts": resolved_end_ts,
        "alpha_enabled": bool(alpha_enabled),
        "classic": classic,
        "alpha_v2": alpha,
        "activation_stats": activation_stats,
        "delta": {
            "trade_count": float(alpha["trade_count"] - classic["trade_count"]),
            "tpm": float(alpha["tpm"] - classic["tpm"]),
            "exp_lcb": float(alpha["exp_lcb"] - classic["exp_lcb"]),
            "max_drawdown": float(alpha["max_drawdown"] - classic["max_drawdown"]),
        },
    }
    run_id = f"{utc_now_compact()}_{stable_hash(payload, length=12)}_stage15"
    payload["run_id"] = run_id
    run_dir = runs_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "ab_compare.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    (run_dir / "ab_compare.md").write_text(_ab_markdown(payload), encoding="utf-8")
    return {"summary": payload, "run_id": run_id, "run_dir": run_dir}


def _run_classic_for_symbol(*, frame: pd.DataFrame, symbol: str, cfg: dict[str, Any]) -> dict[str, Any]:
    strategy = trend_pullback()
    eval_cfg = (((cfg.get("evaluation", {}) or {}).get("stage10", {}) or {}).get("evaluation", {}))
    work = frame.copy()
    work["signal"] = generate_signals(work, strategy=strategy, gating_mode="none")
    bt = run_backtest(
        frame=work,
        strategy_name=strategy.name,
        symbol=symbol,
        signal_col="signal",
        max_hold_bars=int(eval_cfg.get("max_hold_bars", 24)),
        stop_atr_multiple=float(eval_cfg.get("stop_atr_multiple", 1.5)),
        take_profit_atr_multiple=float(eval_cfg.get("take_profit_atr_multiple", 3.0)),
        round_trip_cost_pct=float(cfg.get("costs", {}).get("round_trip_cost_pct", 0.1)),
        slippage_pct=float(cfg.get("costs", {}).get("slippage_pct", 0.0005)),
        cost_model_cfg=cfg.get("cost_model", {}),
    )
    return _metrics(bt=bt, frame=work)


def _run_alpha_for_symbol(*, frame: pd.DataFrame, symbol: str, cfg: dict[str, Any], seed: int) -> tuple[dict[str, Any], dict[str, float]]:
    stage13_cfg = (((cfg.get("evaluation", {}) or {}).get("stage13", {}) or {}))
    enabled = list(stage13_cfg.get("families", {}).get("enabled", ["price", "volatility", "flow"]))
    families = build_families(enabled=enabled, cfg=cfg)
    contracts = [
        ClassicFamilyAdapter(
            name=f"family::{name}",
            family=name,
            role=AlphaRole.ENTRY,
            wrapped=family,
            params={"symbol": symbol, "timeframe": "1h", "seed": int(seed), "config": cfg},
        )
        for name, family in families.items()
    ]
    intents, stats = run_orchestrator(
        frame=frame,
        contracts=contracts,
        seed=int(seed),
        config=cfg,
        orchestrator_cfg=OrchestratorConfig(entry_threshold=0.25, min_confidence=0.05),
    )
    work = frame.copy()
    work["signal"] = intents["side"].shift(1).fillna(0).astype(int)
    eval_cfg = (((cfg.get("evaluation", {}) or {}).get("stage10", {}) or {}).get("evaluation", {}))
    bt = run_backtest(
        frame=work,
        strategy_name="AlphaV2",
        symbol=symbol,
        signal_col="signal",
        max_hold_bars=int(eval_cfg.get("max_hold_bars", 24)),
        stop_atr_multiple=float(eval_cfg.get("stop_atr_multiple", 1.5)),
        take_profit_atr_multiple=float(eval_cfg.get("take_profit_atr_multiple", 3.0)),
        round_trip_cost_pct=float(cfg.get("costs", {}).get("round_trip_cost_pct", 0.1)),
        slippage_pct=float(cfg.get("costs", {}).get("slippage_pct", 0.0005)),
        cost_model_cfg=cfg.get("cost_model", {}),
    )
    metrics = _metrics(bt=bt, frame=work)
    act = {
        "contracts_evaluated": float(stats["contracts_evaluated"]),
        "pct_not_neutral_multiplier": float(stats["pct_not_neutral_multiplier"]),
        "pct_nonzero_confidence": float((pd.to_numeric(intents["confidence"], errors="coerce").fillna(0.0) > 0).mean() * 100.0),
    }
    return metrics, act


def _months(frame: pd.DataFrame) -> float:
    ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce").dropna()
    if ts.empty:
        return 1.0
    days = float((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400.0)
    return max(days / 30.0, 1e-9)


def _metrics(*, bt: Any, frame: pd.DataFrame) -> dict[str, Any]:
    trade_count = float(bt.metrics.get("trade_count", 0.0))
    tpm = float(trade_count / _months(frame))
    pnl = pd.to_numeric(bt.trades.get("pnl", pd.Series(dtype=float)), errors="coerce").dropna().to_numpy(dtype=float)
    if pnl.size == 0:
        exp_lcb = 0.0
    else:
        exp_lcb = float(np.mean(pnl) - np.std(pnl, ddof=0) / max(1.0, np.sqrt(float(pnl.size))))
    return {
        "trade_count": trade_count,
        "tpm": tpm,
        "exposure_ratio": float(
            pd.to_numeric(bt.trades.get("bars_held", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()
            / max(1.0, float(len(frame)))
        )
        if not bt.trades.empty
        else 0.0,
        "PF_raw": float(bt.metrics.get("profit_factor", 0.0)),
        "PF": float(np.clip(float(bt.metrics.get("profit_factor", 0.0)), 0.0, 10.0)),
        "expectancy": float(bt.metrics.get("expectancy", 0.0)),
        "exp_lcb": exp_lcb,
        "max_drawdown": float(bt.metrics.get("max_drawdown", 0.0)),
        "equity_hash": stable_hash(bt.equity_curve.to_dict(orient="records"), length=16),
        "trades_hash": stable_hash(bt.trades.to_dict(orient="records"), length=16),
    }


def _aggregate(*, rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "trade_count": 0.0,
            "tpm": 0.0,
            "exposure_ratio": 0.0,
            "PF_raw": 0.0,
            "PF": 0.0,
            "expectancy": 0.0,
            "exp_lcb": 0.0,
            "max_drawdown": 0.0,
            "equity_hash": "",
            "trades_hash": "",
        }
    keys = ["trade_count", "tpm", "exposure_ratio", "PF_raw", "PF", "expectancy", "exp_lcb", "max_drawdown"]
    out = {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}
    out["equity_hash"] = stable_hash([row["equity_hash"] for row in rows], length=16)
    out["trades_hash"] = stable_hash([row["trades_hash"] for row in rows], length=16)
    return out


def _ab_markdown(payload: dict[str, Any]) -> str:
    c = payload["classic"]
    a = payload["alpha_v2"]
    d = payload["delta"]
    return "\n".join(
        [
            "# Stage-15 A/B Compare",
            "",
            f"- run_id: `{payload['run_id']}`",
            f"- seed: `{payload['seed']}`",
            f"- config_hash: `{payload['config_hash']}`",
            f"- data_hash: `{payload['data_hash']}`",
            f"- resolved_end_ts: `{payload['resolved_end_ts']}`",
            "",
            "## Classic",
            f"- trade_count: `{c['trade_count']:.6f}`",
            f"- exp_lcb: `{c['exp_lcb']:.6f}`",
            "",
            "## Alpha v2",
            f"- trade_count: `{a['trade_count']:.6f}`",
            f"- exp_lcb: `{a['exp_lcb']:.6f}`",
            "",
            "## Delta",
            f"- trade_count: `{d['trade_count']:.6f}`",
            f"- exp_lcb: `{d['exp_lcb']:.6f}`",
            f"- max_drawdown: `{d['max_drawdown']:.6f}`",
        ]
    ) + "\n"
