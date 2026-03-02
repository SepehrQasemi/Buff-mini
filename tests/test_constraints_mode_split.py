from __future__ import annotations

import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.stage23.order_builder import build_adaptive_orders


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=6, freq="h", tz="UTC"),
            "open": [100.0, 100.5, 101.0, 100.8, 100.4, 100.2],
            "high": [100.8, 101.0, 101.4, 101.0, 100.8, 100.5],
            "low": [99.8, 100.1, 100.5, 100.2, 99.9, 99.7],
            "close": [100.0, 100.5, 101.0, 100.8, 100.4, 100.2],
            "volume": [5000.0, 5200.0, 5300.0, 5100.0, 4900.0, 5000.0],
            "atr_14": [1.2, 1.2, 1.1, 1.1, 1.0, 1.0],
        }
    )


def _cfg(mode: str) -> dict:
    cfg = load_config(DEFAULT_CONFIG_PATH)
    cfg.setdefault("evaluation", {}).setdefault("stage23", {})
    cfg["evaluation"]["stage23"]["enabled"] = True
    cfg["evaluation"]["stage23"].setdefault("order_builder", {})
    cfg["evaluation"]["stage23"]["order_builder"]["allow_size_bump_to_min_notional"] = False
    cfg.setdefault("evaluation", {}).setdefault("stage24", {})["enabled"] = False
    cfg.setdefault("evaluation", {}).setdefault("constraints", {})
    cfg["evaluation"]["constraints"]["mode"] = str(mode)
    cfg["evaluation"]["constraints"]["live"] = {
        "min_trade_notional": 10.0,
        "min_trade_qty": 0.0,
        "qty_step": 0.001,
    }
    cfg["evaluation"]["constraints"]["research"] = {
        "min_trade_notional": 0.5,
        "min_trade_qty": 0.0,
        "qty_step": 0.0,
    }
    cfg.setdefault("risk", {})["max_gross_exposure"] = 50.0
    return cfg


def test_research_mode_accepts_smaller_notional_and_tracks_shadow_live_rejects() -> None:
    frame = _frame()
    raw_side = pd.Series([1, 1, 1, 1, 1, 1], index=frame.index, dtype=int)
    score = pd.Series([0.01, 0.012, 0.011, 0.013, 0.01, 0.009], index=frame.index, dtype=float)
    out = build_adaptive_orders(
        frame=frame,
        raw_side=raw_side,
        score=score,
        cfg=_cfg("research"),
        symbol="BTC/USDT",
    )
    assert int((out["accepted_signal"] != 0).sum()) > 0
    shadow = dict(out.get("shadow_live_summary", {}))
    assert bool(shadow.get("enabled", False)) is True
    assert int(shadow.get("research_accepted_count", 0)) > 0
    assert int(shadow.get("research_accepted_but_live_rejected_count", 0)) > 0
    reason_counts = dict(shadow.get("live_reject_reason_counts", {}))
    assert int(reason_counts.get("SIZE_TOO_SMALL", 0)) > 0


def test_live_mode_enforces_live_constraints_and_no_shadow_counts() -> None:
    frame = _frame()
    raw_side = pd.Series([1, 1, 1, 1, 1, 1], index=frame.index, dtype=int)
    score = pd.Series([0.01, 0.012, 0.011, 0.013, 0.01, 0.009], index=frame.index, dtype=float)
    out = build_adaptive_orders(
        frame=frame,
        raw_side=raw_side,
        score=score,
        cfg=_cfg("live"),
        symbol="BTC/USDT",
    )
    assert int((out["accepted_signal"] != 0).sum()) == 0
    reasons = [str(event.get("reason", "")) for event in list(out.get("reject_events", []))]
    assert "SIZE_TOO_SMALL" in reasons
    shadow = dict(out.get("shadow_live_summary", {}))
    assert bool(shadow.get("enabled", False)) is False
    assert int(shadow.get("research_accepted_count", 0)) == 0

