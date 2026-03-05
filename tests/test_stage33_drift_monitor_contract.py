from __future__ import annotations

import numpy as np

from buffmini.stage33.drift import build_drift_summary, performance_drift, representation_drift


def test_stage33_drift_monitor_contract() -> None:
    rng = np.random.default_rng(1)
    base = rng.normal(size=(200, 8)).astype(float)
    recent = base * 1.05 + 0.01
    rep = representation_drift(base, recent)
    perf = performance_drift(
        baseline_metrics={"exp_lcb": 0.02, "PF_clipped": 1.2, "maxDD": 0.15},
        recent_metrics={"exp_lcb": 0.01, "PF_clipped": 1.1, "maxDD": 0.16},
    )
    out = build_drift_summary(rep_drift=rep, perf_drift=perf)
    assert "representation_drift" in out
    assert "performance_drift" in out
    assert "warnings" in out
    assert isinstance(out["warnings"], list)

