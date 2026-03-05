from __future__ import annotations

from typing import Any

from buffmini.data.coinapi.planner import build_backfill_plan
from scripts.update_coinapi_extras import execute_plan_items


def test_coinapi_planner_builds_deterministic_slices() -> None:
    plan = build_backfill_plan(
        symbols=["BTC/USDT", "ETH/USDT"],
        endpoints=["funding_rates", "open_interest"],
        start_ts="2026-01-01T00:00:00Z",
        end_ts="2026-01-20T00:00:00Z",
        increment_days=7,
        max_requests=999,
    )
    # 20-day span split into 3 slices x 2 symbols x 2 endpoints.
    assert plan["planned_count"] == 12
    assert plan["selected_count"] == 12
    assert plan["truncated"] is False
    # deterministic plan id
    plan2 = build_backfill_plan(
        symbols=["BTC/USDT", "ETH/USDT"],
        endpoints=["funding_rates", "open_interest"],
        start_ts="2026-01-01T00:00:00Z",
        end_ts="2026-01-20T00:00:00Z",
        increment_days=7,
        max_requests=999,
    )
    assert plan["plan_id"] == plan2["plan_id"]


def test_coinapi_planner_respects_max_requests_and_stops_execution() -> None:
    plan = build_backfill_plan(
        symbols=["BTC/USDT"],
        endpoints=["funding_rates", "open_interest"],
        start_ts="2026-01-01T00:00:00Z",
        end_ts="2026-01-15T00:00:00Z",
        increment_days=2,
        max_requests=3,
    )
    assert plan["selected_count"] == 3
    assert plan["truncated"] is True

    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        def request_json(self, path: str, *, params: dict[str, Any], endpoint_name: str, symbol: str, time_start: str, time_end: str, plan_id: str):  # noqa: ANN001, D401, E501
            self.calls += 1
            return [{"time_exchange": time_start, "value": 1.0, "open_interest": 2.0, "funding_rate": 0.1}], type(
                "Meta",
                (),
                {"response_bytes": 12},
            )()

    fake = FakeClient()
    result = execute_plan_items(plan=plan, client=fake, store_raw=False)
    assert fake.calls == 3
    assert result["selected_count"] == 3
    assert result["success_count"] == 3

