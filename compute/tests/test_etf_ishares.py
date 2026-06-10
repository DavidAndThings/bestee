"""Tests for bestee_compute.etf.ishares — iShares (BlackRock) ETF holdings scraper."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import polars as pl
import pytest
from great_tables import GT

from bestee_compute.etf.ishares import (
    _PRODUCT_IDS,
    _parse_holdings_csv,
    get_holdings,
    get_holdings_table,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "ivv_holdings.csv"
_HTTPX_PATCH = "bestee_compute.etf.ishares.httpx.Client"


@pytest.fixture
def csv_bytes() -> bytes:
    """Real IVV holdings CSV captured from iShares."""
    return _FIXTURE.read_bytes()


# ── CSV parser ───────────────────────────────────────────────────────


class TestParseHoldingsCsv:
    def test_columns_match_ishares_schema(self, csv_bytes: bytes) -> None:
        df = _parse_holdings_csv(csv_bytes)
        assert df.columns == [
            "Ticker",
            "Name",
            "Sector",
            "Asset Class",
            "Market Value",
            "Weight (%)",
            "Notional Value",
            "Quantity",
            "Price",
            "Location",
            "Exchange",
            "Currency",
            "FX Rate",
            "Market Currency",
            "Accrual Date",
        ]

    def test_numeric_columns_are_floats(self, csv_bytes: bytes) -> None:
        df = _parse_holdings_csv(csv_bytes)
        for col in (
            "Market Value",
            "Weight (%)",
            "Notional Value",
            "Quantity",
            "Price",
            "FX Rate",
        ):
            assert df.schema[col] == pl.Float64, col

    def test_top_holding_is_nvidia(self, csv_bytes: bytes) -> None:
        df = _parse_holdings_csv(csv_bytes)
        top = df.row(0, named=True)
        assert top["Ticker"] == "NVDA"
        assert top["Name"] == "NVIDIA CORP"
        # Stored as decimal (0.0834 for 8.34%).
        assert top["Weight (%)"] == pytest.approx(0.0834, abs=1e-3)
        assert top["Market Value"] == pytest.approx(6.9162e10, rel=1e-3)

    def test_weights_sum_to_approximately_one(self, csv_bytes: bytes) -> None:
        df = _parse_holdings_csv(csv_bytes)
        # Sum of rounded weights — allow 1% slack.
        assert df["Weight (%)"].sum() == pytest.approx(1.0, abs=0.01)

    def test_row_count_is_reasonable(self, csv_bytes: bytes) -> None:
        # IVV tracks the S&P 500 (~500 holdings).
        df = _parse_holdings_csv(csv_bytes)
        assert 480 < df.height < 520

    def test_missing_header_raises(self) -> None:
        with pytest.raises(ValueError, match="header row"):
            _parse_holdings_csv(b"Fund: foo\nSomething else\n")

    def test_bond_etf_header_without_ticker(self) -> None:
        """Bond ETFs use ``Name,`` as the first column — must still parse."""
        csv = (
            b"iShares Core US Aggregate Bond ETF\n"
            b'Fund Holdings as of,"x"\n'
            b"\n"
            b"Name,Sector,Asset Class,Market Value,Weight (%),Notional Value\n"
            b'"TREASURY NOTE","Treasury","Fixed Income",'
            b'"1,000,000","0.50","1,000,000"\n'
            b'"BLACKROCK CASH","Money Market","Cash",'
            b'"500,000","0.25","500,000"\n'
        )
        df = _parse_holdings_csv(csv)
        assert df.height == 2
        assert df["Name"].to_list() == ["TREASURY NOTE", "BLACKROCK CASH"]
        assert df["Weight (%)"].to_list() == [0.005, 0.0025]

    def test_drops_blank_ticker_rows(self) -> None:
        csv = (
            b"Some metadata\n"
            b'Fund Holdings as of,"foo"\n'
            b"\n"
            b"Ticker,Name,Weight (%)\n"
            b'"AAPL","Apple Inc","5.00"\n'
            b',"","",\n'
            b'"NVDA","Nvidia Corp","3.00"\n'
        )
        df = _parse_holdings_csv(csv)
        assert df["Ticker"].to_list() == ["AAPL", "NVDA"]


# ── Product-ID lookup ────────────────────────────────────────────────


class TestProductIdLookup:
    def test_known_tickers_table_includes_ivv(self) -> None:
        assert _PRODUCT_IDS["IVV"] == 239726

    def test_known_tickers_cover_top_etfs(self) -> None:
        # Spot-check a few that we definitely want supported.
        for ticker in ("IVV", "AGG", "TLT", "EFA", "EEM", "USMV"):
            assert ticker in _PRODUCT_IDS, ticker


# ── Full-pipeline HTTP-mocked tests ──────────────────────────────────


@patch(_HTTPX_PATCH)
def test_get_holdings_resolves_product_id_from_table(
    mock_client_cls: MagicMock,
    csv_bytes: bytes,
) -> None:
    response = MagicMock(spec=httpx.Response)
    response.content = csv_bytes
    response.raise_for_status = MagicMock()
    client = MagicMock()
    client.get.return_value = response
    mock_client_cls.return_value.__enter__.return_value = client

    df = get_holdings("IVV")

    assert isinstance(df, pl.DataFrame)
    assert df.height > 0

    # Verify the URL parameters carry the expected portfolioId.
    _, kwargs = client.get.call_args
    params = kwargs["params"]
    assert params["portfolioId"] == "239726"
    assert params["component"] == "holdings"
    assert params["appSubType"] == "ISHARES"


@patch(_HTTPX_PATCH)
def test_get_holdings_accepts_product_id_override(
    mock_client_cls: MagicMock,
    csv_bytes: bytes,
) -> None:
    """Passing product_id directly bypasses the ticker lookup."""
    response = MagicMock(spec=httpx.Response)
    response.content = csv_bytes
    response.raise_for_status = MagicMock()
    client = MagicMock()
    client.get.return_value = response
    mock_client_cls.return_value.__enter__.return_value = client

    get_holdings("XXXX", product_id=999999)

    _, kwargs = client.get.call_args
    assert kwargs["params"]["portfolioId"] == "999999"


def test_get_holdings_unknown_ticker_raises() -> None:
    """Unknown ticker without product_id should fail with a helpful message."""
    with pytest.raises(ValueError, match="No iShares product_id known"):
        get_holdings("UNKNOWN_TICKER")


@patch(_HTTPX_PATCH)
def test_get_holdings_ticker_case_insensitive(
    mock_client_cls: MagicMock,
    csv_bytes: bytes,
) -> None:
    response = MagicMock(spec=httpx.Response)
    response.content = csv_bytes
    response.raise_for_status = MagicMock()
    client = MagicMock()
    client.get.return_value = response
    mock_client_cls.return_value.__enter__.return_value = client

    get_holdings("ivv")  # lowercase
    assert client.get.call_args.kwargs["params"]["portfolioId"] == "239726"


@patch(_HTTPX_PATCH)
def test_get_holdings_default_ticker_is_ivv(
    mock_client_cls: MagicMock,
    csv_bytes: bytes,
) -> None:
    response = MagicMock(spec=httpx.Response)
    response.content = csv_bytes
    response.raise_for_status = MagicMock()
    client = MagicMock()
    client.get.return_value = response
    mock_client_cls.return_value.__enter__.return_value = client

    get_holdings()
    assert client.get.call_args.kwargs["params"]["portfolioId"] == "239726"


@patch(_HTTPX_PATCH)
def test_get_holdings_table_returns_gt(
    mock_client_cls: MagicMock,
    csv_bytes: bytes,
) -> None:
    response = MagicMock(spec=httpx.Response)
    response.content = csv_bytes
    response.raise_for_status = MagicMock()
    client = MagicMock()
    client.get.return_value = response
    mock_client_cls.return_value.__enter__.return_value = client

    gt = get_holdings_table("IVV")
    assert isinstance(gt, GT)
    html = gt.as_raw_html()
    assert "NVIDIA" in html
    assert "IVV Holdings" in html
