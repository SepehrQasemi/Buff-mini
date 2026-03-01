from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd

from buffmini.config import load_config
from buffmini.constants import DEFAULT_CONFIG_PATH
from buffmini.signals.registry import build_families
from buffmini.stage13.evaluate import run_stage13, validate_stage13_summary_schema
from buffmini.validation.leakage_harness import synthetic_ohlcv
from buffmini.data.features import calculate_features


def _cfg() -> dict:
    cfg = deepcopy(load_config(DEFAULT_CONFIG_PATH))
    cfg["evaluation"]["stage13"]["enabled"] = True
    cfg["evaluation"]["stage13"]["families"]["enabled"] = ["price", "volatility", "flow"]
    cfg["evaluation"]["stage13"]["composer"]["mode"] = "weighted_sum"
    cfg["evaluation"]["stage10"]["evaluation"]["dry_run_rows"] = 600
    cfg["evaluation"]["stage12"]["monte_carlo"]["n_paths"] = 300
    cfg["evaluation"]["stage12"]["monte_carlo"]["top_pct"] = 1.0
    return cfg


def _feature_frame() -> pd.DataFrame:
    raw = synthetic_ohlcv(rows=640, seed=42)
    cfg = {
        "data": {
            "include_futures_extras": True,
            "futures_extras": {
                "timeframe": "1h",
                "max_fill_gap_bars": 8,
                "funding": {"z_windows": [30, 90], "trend_window": 24, "abs_pctl_window": 180, "extreme_pctl": 0.95},
                "open_interest": {
                    "chg_windows": [1, 24],
                    "z_window": 30,
                    "oi_to_volume_window": 24,
                    "overlay": {"enabled": False},
                },
            },
        }
    }
    return calculate_features(
        raw,
        config=cfg,
        symbol="BTC/USDT",
        timeframe="1h",
        _synthetic_extras_for_tests=True,
    )


def test_stage13_family_contract_deterministic_bounds() -> None:
    cfg = _cfg()
    families = build_families(enabled=["price", "volatility", "flow"], cfg=cfg)
    frame = _feature_frame()
    for name, family in families.items():
        ctx = type("Ctx", (), {"symbol": "BTC/USDT", "timeframe": "1h", "seed": 42, "config": cfg, "params": {}})()
        s1 = family.compute_scores(frame, ctx)
        s2 = family.compute_scores(frame, ctx)
        pd.testing.assert_series_equal(s1, s2, check_exact=False, atol=1e-12, rtol=0.0)
        assert (s1.abs() <= 1.0 + 1e-12).all()
        entries = family.propose_entries(s1, frame, ctx)
        required = {"direction", "confidence", "reasons", "signal", "score", "long_entry", "short_entry"}
        assert required.issubset(entries.columns)
        assert (entries["confidence"] >= 0.0).all()
        assert (entries["confidence"] <= 1.0 + 1e-12).all()


def test_stage13_toggle_and_schema(tmp_path: Path) -> None:
    cfg = _cfg()
    result = run_stage13(
        config=cfg,
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        timeframe="1h",
        runs_root=tmp_path / "runs",
        docs_dir=tmp_path / "docs",
        stage_tag="13.1",
        report_name="stage13_1_architecture",
    )
    summary = dict(result["summary"])
    validate_stage13_summary_schema(summary)
    assert (tmp_path / "docs" / "stage13_1_architecture_report.md").exists()
    assert (tmp_path / "docs" / "stage13_1_architecture_summary.json").exists()


def test_stage13_disabled_matches_baseline_hash(tmp_path: Path) -> None:
    cfg = _cfg()
    cfg["evaluation"]["stage13"]["enabled"] = False
    a = run_stage13(
        config=cfg,
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        timeframe="1h",
        runs_root=tmp_path / "runs_a",
        docs_dir=tmp_path / "docs_a",
        stage_tag="13.1",
        report_name="stage13_1_architecture_a",
    )
    b = run_stage13(
        config=cfg,
        seed=42,
        dry_run=True,
        symbols=["BTC/USDT"],
        timeframe="1h",
        runs_root=tmp_path / "runs_b",
        docs_dir=tmp_path / "docs_b",
        stage_tag="13.1",
        report_name="stage13_1_architecture_b",
    )
    assert a["summary"]["baseline_hash"] == b["summary"]["baseline_hash"]
    assert a["summary"]["classification"] == b["summary"]["classification"]
