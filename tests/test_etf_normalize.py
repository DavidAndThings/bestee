"""Tests for bestee.etf.normalize — canonical-schema projection."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from bestee.etf.ishares import _parse_holdings_csv as _parse_ishares_csv
from bestee.etf.normalize import (
    _CANONICAL_COLS,
    _COLUMN_MAP,
    get_holdings_normalized,
    normalize_holdings,
    supported_issuers_for_normalization,
)
from bestee.etf.spdr import _parse_holdings_xlsx as _parse_spdr_xlsx
from bestee.etf.vaneck import _parse_holdings_xlsx as _parse_vaneck_xlsx

_FIXTURES = Path(__file__).parent / "fixtures"

_CANONICAL_OUTPUT_COLS = (*_CANONICAL_COLS, "issuer")


# ── Registry ─────────────────────────────────────────────────────────


class TestRegistry:
    def test_supported_issuers_covers_all_providers(self) -> None:
        names = supported_issuers_for_normalization()
        assert {"VanEck", "SPDR", "iShares", "Roundhill"} <= set(names)

    def test_unknown_issuer_raises(self) -> None:
        df = pl.DataFrame({"Ticker": ["NVDA"]})
        with pytest.raises(KeyError, match="Unknown issuer"):
            normalize_holdings(df, "Vanguard")


# ── Per-provider projection ──────────────────────────────────────────


class TestVanEck:
    def test_smh_fixture_projects(self) -> None:
        raw = _parse_vaneck_xlsx((_FIXTURES / "smh_holdings.xlsx").read_bytes())
        out = normalize_holdings(raw, "VanEck")

        assert list(out.columns) == list(_CANONICAL_OUTPUT_COLS)
        assert out.height == raw.height
        # Top SMH holding is NVDA per the captured fixture.
        top = out.row(0, named=True)
        assert top["ticker"] == "NVDA"
        assert top["name"] == "Nvidia Corp"
        assert top["weight"] == pytest.approx(0.167, abs=1e-3)
        assert top["market_value"] > 0
        assert top["issuer"] == "VanEck"


class TestSpdr:
    def test_spy_fixture_projects(self) -> None:
        raw = _parse_spdr_xlsx((_FIXTURES / "spy_holdings.xlsx").read_bytes())
        out = normalize_holdings(raw, "SPDR")

        assert list(out.columns) == list(_CANONICAL_OUTPUT_COLS)
        top = out.row(0, named=True)
        assert top["ticker"] == "NVDA"
        assert top["name"] == "NVIDIA CORP"
        # SPDR weight is already decimal post-parse (0.0834 for 8.34%).
        assert top["weight"] == pytest.approx(0.0834, abs=1e-3)
        # SPDR doesn't publish Market Value — column is all-null.
        assert out["market_value"].null_count() == out.height
        assert top["issuer"] == "SPDR"


class TestIshares:
    def test_ivv_fixture_projects(self) -> None:
        raw = _parse_ishares_csv((_FIXTURES / "ivv_holdings.csv").read_bytes())
        out = normalize_holdings(raw, "iShares")

        assert list(out.columns) == list(_CANONICAL_OUTPUT_COLS)
        top = out.row(0, named=True)
        assert top["ticker"] == "NVDA"
        assert top["name"] == "NVIDIA CORP"
        assert top["weight"] == pytest.approx(0.0834, abs=1e-3)
        assert top["market_value"] > 0
        assert top["issuer"] == "iShares"

    def test_bond_layout_yields_null_ticker(self) -> None:
        """An iShares bond CSV has no Ticker column — ticker should be null."""
        bond_csv = (
            b"iShares Core US Aggregate Bond ETF\n"
            b'Fund Holdings as of,"x"\n'
            b"\n"
            b"Name,Sector,Asset Class,Market Value,Weight (%),Notional Value\n"
            b'"TREASURY NOTE","Treasury","Fixed Income",'
            b'"1,000,000","0.50","1,000,000"\n'
        )
        raw = _parse_ishares_csv(bond_csv)
        out = normalize_holdings(raw, "iShares")

        assert out["ticker"].null_count() == out.height
        assert out.row(0, named=True)["name"] == "TREASURY NOTE"
        assert out.row(0, named=True)["weight"] == pytest.approx(0.005)


class TestRoundhill:
    def test_roundhill_dataframe_projects(self) -> None:
        """Roundhill uses StockTicker/SecurityName/Weightings/MarketValue."""
        raw = pl.DataFrame(
            {
                "Date": ["05/26/2026", "05/26/2026"],
                "Account": ["CHAT", "CHAT"],
                "StockTicker": ["000660 KS", "005930 KS"],
                "CUSIP": ["6450267", "6771720"],
                "SecurityName": ["SK hynix Inc", "Samsung Electronics Co Ltd"],
                "Weightings": [0.0553, 0.0403],
                "MarketValue": [103_113_465.42, 75_130_284.50],
            }
        )
        out = normalize_holdings(raw, "Roundhill")

        assert list(out.columns) == list(_CANONICAL_OUTPUT_COLS)
        assert out["ticker"].to_list() == ["000660 KS", "005930 KS"]
        assert out["name"].to_list() == ["SK hynix Inc", "Samsung Electronics Co Ltd"]
        assert out["weight"].to_list() == [0.0553, 0.0403]
        assert out["market_value"].to_list() == [103_113_465.42, 75_130_284.50]
        assert out["issuer"].to_list() == ["Roundhill", "Roundhill"]


# ── Cross-issuer concatenation ───────────────────────────────────────


class TestConcatenation:
    """Demonstrate the canonical schema is friendly to cross-issuer concat."""

    def test_can_concat_three_providers(self) -> None:
        smh = normalize_holdings(
            _parse_vaneck_xlsx((_FIXTURES / "smh_holdings.xlsx").read_bytes()),
            "VanEck",
        )
        spy = normalize_holdings(
            _parse_spdr_xlsx((_FIXTURES / "spy_holdings.xlsx").read_bytes()),
            "SPDR",
        )
        ivv = normalize_holdings(
            _parse_ishares_csv((_FIXTURES / "ivv_holdings.csv").read_bytes()),
            "iShares",
        )
        combined = pl.concat([smh, spy, ivv])

        assert combined.height == smh.height + spy.height + ivv.height
        assert set(combined["issuer"].unique().to_list()) == {
            "VanEck",
            "SPDR",
            "iShares",
        }


# ── Dispatcher + normalize one-shot ──────────────────────────────────


@patch("bestee.etf.normalize._resolve_provider")
def test_get_holdings_normalized_dispatches_and_normalizes(
    mock_resolve: MagicMock,
) -> None:
    raw = pl.DataFrame(
        {
            "Ticker": ["NVDA"],
            "Holding Name": ["Nvidia Corp"],
            "% of Net Assets": [0.167],
            "Market Value (US$)": [1e10],
            # Plus other columns we should drop.
            "Identifier (FIGI)": ["BBG000BBJQV0"],
            "Shares": [50_000_000],
        }
    )
    mock_provider = MagicMock()
    mock_provider.get_holdings.return_value = raw
    mock_resolve.return_value = ("VanEck", mock_provider)

    out = get_holdings_normalized("SMH")

    mock_provider.get_holdings.assert_called_once_with("SMH")
    assert list(out.columns) == list(_CANONICAL_OUTPUT_COLS)
    assert out.row(0, named=True) == {
        "ticker": "NVDA",
        "name": "Nvidia Corp",
        "weight": 0.167,
        "market_value": 1e10,
        "issuer": "VanEck",
    }


# ── Sanity on every mapping entry ────────────────────────────────────


def test_column_map_covers_canonical_keys() -> None:
    """Every mapping entry uses only known canonical keys."""
    canonical = set(_CANONICAL_COLS)
    for issuer, m in _COLUMN_MAP.items():
        unknown = set(m) - canonical
        assert not unknown, f"{issuer}: unknown canonical keys {unknown}"
