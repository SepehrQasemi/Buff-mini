from __future__ import annotations

from buffmini.stage41.contribution import oi_short_only_runtime_guard


def test_stage41_oi_short_only_runtime_guard_blocks_1h() -> None:
    guard = oi_short_only_runtime_guard(timeframe="1h", short_only_enabled=True, short_horizon_max="30m")
    assert guard["timeframe_allowed"] is False
    assert guard["oi_allowed"] is False


def test_stage41_oi_short_only_runtime_guard_allows_15m() -> None:
    guard = oi_short_only_runtime_guard(timeframe="15m", short_only_enabled=True, short_horizon_max="30m")
    assert guard["timeframe_allowed"] is True
    assert guard["oi_allowed"] is True

