from __future__ import annotations

import pandas as pd

from buffmini.stage26.context import classify_context
from buffmini.stage26.rulelets import RuleletContract, build_rulelet_library
from buffmini.validation.leakage_harness import synthetic_ohlcv


def _frame() -> pd.DataFrame:
    base = synthetic_ohlcv(rows=900, seed=99)
    return classify_context(base)


def test_stage26_rulelet_library_minimum_and_contract() -> None:
    library = build_rulelet_library()
    assert len(library) >= 12
    for name, rulelet in library.items():
        assert isinstance(rulelet, RuleletContract)
        assert str(name) == str(rulelet.name)
        assert isinstance(rulelet.required_features(), list)
        assert len(rulelet.required_features()) > 0
        assert isinstance(rulelet.contexts_allowed, tuple)
        assert len(rulelet.contexts_allowed) > 0
        assert -1.0 <= float(rulelet.threshold) <= 1.0


def test_stage26_rulelet_scores_bounded_and_deterministic() -> None:
    frame = _frame()
    library = build_rulelet_library()
    for rulelet in library.values():
        a = rulelet.compute_score(frame)
        b = rulelet.compute_score(frame)
        assert len(a) == len(frame)
        assert a.index.equals(frame.index)
        assert bool(((a >= -1.0) & (a <= 1.0)).all())
        pd.testing.assert_series_equal(a, b, check_names=False)


def test_stage26_rulelet_required_features_present() -> None:
    frame = _frame()
    library = build_rulelet_library()
    for rulelet in library.values():
        required = set(rulelet.required_features())
        # Rulelets must at minimum declare canonical OHLCV fields.
        for col in ("open", "high", "low", "close", "volume"):
            assert col in required, f"{rulelet.name} missing required feature declaration for {col}"
        # Contract-level guarantee: compute_score still returns aligned output.
        score = rulelet.compute_score(frame)
        assert len(score) == len(frame)
