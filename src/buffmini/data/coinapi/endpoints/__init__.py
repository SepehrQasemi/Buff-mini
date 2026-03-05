"""CoinAPI endpoint adapters for canonical normalized datasets."""

from .funding_rates import (
    FUNDING_RATES_ENDPOINT_NAME,
    FUNDING_RATES_PATH,
    funding_coverage_summary,
    normalize_funding_rates,
    write_funding_canonical,
)
from .liquidations import (
    LIQUIDATIONS_ENDPOINT_NAME,
    LIQUIDATIONS_PATH,
    liquidations_coverage_summary,
    normalize_liquidations,
    write_liquidations_canonical,
)
from .open_interest import (
    OPEN_INTEREST_ENDPOINT_NAME,
    OPEN_INTEREST_PATH,
    normalize_open_interest,
    open_interest_coverage_summary,
    write_open_interest_canonical,
)

__all__ = [
    "FUNDING_RATES_ENDPOINT_NAME",
    "FUNDING_RATES_PATH",
    "funding_coverage_summary",
    "normalize_funding_rates",
    "write_funding_canonical",
    "OPEN_INTEREST_ENDPOINT_NAME",
    "OPEN_INTEREST_PATH",
    "normalize_open_interest",
    "open_interest_coverage_summary",
    "write_open_interest_canonical",
    "LIQUIDATIONS_ENDPOINT_NAME",
    "LIQUIDATIONS_PATH",
    "normalize_liquidations",
    "liquidations_coverage_summary",
    "write_liquidations_canonical",
]

