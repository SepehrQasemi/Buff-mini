from __future__ import annotations

import pytest

from buffmini.data.derived_tf import resolve_divisible_base


def test_stage26_9_resolve_divisible_base_examples() -> None:
    available = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d", "1w"]
    assert resolve_divisible_base("2h", available) == "2h"  # canonical load path
    assert resolve_divisible_base("45m", available) == "15m"
    # If 30m is canonical this resolves directly; derived case covered below.
    assert resolve_divisible_base("30m", ["1m", "5m", "15m", "1h"]) == "5m"


def test_stage26_9_resolve_divisible_base_hourly_preferences() -> None:
    assert resolve_divisible_base("12h", ["1m", "5m", "15m", "4h"]) == "4h"
    assert resolve_divisible_base("12h", ["1m", "5m", "15m", "1h", "4h"]) == "1h"


def test_stage26_9_resolve_divisible_base_errors() -> None:
    with pytest.raises(ValueError, match="No divisible canonical base"):
        resolve_divisible_base("7m", ["5m", "1h"])
    with pytest.raises(ValueError, match="monthly"):
        resolve_divisible_base("1M", ["1m", "5m", "1h"])
