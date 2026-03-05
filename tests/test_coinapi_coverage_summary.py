from __future__ import annotations

import json
from pathlib import Path

from buffmini.data.coinapi.endpoints.funding_rates import normalize_funding_rates, write_funding_canonical
from buffmini.data.coinapi.endpoints.open_interest import normalize_open_interest, write_open_interest_canonical


def _fixture(name: str) -> list[dict]:
    path = Path(__file__).resolve().parent / "data" / "coinapi" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_write_canonical_and_coverage_files(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical"
    meta_root = tmp_path / "meta"

    funding = normalize_funding_rates(_fixture("funding_fixture.json"), symbol="BTC/USDT")
    fp, fm = write_funding_canonical(
        funding,
        symbol="BTC/USDT",
        canonical_root=canonical_root,
        meta_root=meta_root,
    )
    assert fp.exists()
    assert fm.exists()

    oi = normalize_open_interest(_fixture("open_interest_fixture.json"), symbol="ETH/USDT")
    op, om = write_open_interest_canonical(
        oi,
        symbol="ETH/USDT",
        canonical_root=canonical_root,
        meta_root=meta_root,
    )
    assert op.exists()
    assert om.exists()
    summary = json.loads(om.read_text(encoding="utf-8"))
    assert summary["sample_count"] == 3
    assert summary["endpoint"] == "open_interest"

