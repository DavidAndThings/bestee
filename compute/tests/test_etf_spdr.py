"""Tests for bestee_compute.etf.spdr — State Street SPDR ETF holdings scraper."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import polars as pl
import pytest
from great_tables import GT

from bestee_compute.etf.spdr import (
    _parse_holdings_xlsx,
    get_holdings,
    get_holdings_table,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "spy_holdings.xlsx"
_HTTPX_PATCH = "bestee_compute.etf.spdr.httpx.Client"


@pytest.fixture
def xlsx_bytes() -> bytes:
    """Real SPY holdings xlsx captured from SSGA."""
    return _FIXTURE.read_bytes()


# ── xlsx parser ──────────────────────────────────────────────────────


class TestParseHoldingsXlsx:
    def test_columns_match_spdr_schema(self, xlsx_bytes: bytes) -> None:
        df = _parse_holdings_xlsx(xlsx_bytes)
        assert df.columns == [
            "Name",
            "Ticker",
            "Identifier",
            "SEDOL",
            "Weight",
            "Sector",
            "Shares Held",
            "Local Currency",
        ]

    def test_numeric_coercion(self, xlsx_bytes: bytes) -> None:
        df = _parse_holdings_xlsx(xlsx_bytes)
        assert df.schema["Weight"] == pl.Float64
        assert df.schema["Shares Held"] == pl.Int64
        # Other columns remain strings.
        assert df.schema["Ticker"] == pl.Utf8
        assert df.schema["Name"] == pl.Utf8
        assert df.schema["Identifier"] == pl.Utf8

    def test_top_holding_is_nvidia(self, xlsx_bytes: bytes) -> None:
        df = _parse_holdings_xlsx(xlsx_bytes)
        top = df.row(0, named=True)
        assert top["Ticker"] == "NVDA"
        assert top["Name"] == "NVIDIA CORP"
        # The fixture sample's NVDA weight is ~8.35% — stored as decimal.
        assert top["Weight"] == pytest.approx(0.0835, abs=1e-3)
        # Shares Held are converted from scientific notation to int.
        assert top["Shares Held"] == 291_237_232

    def test_weights_sum_to_approximately_one(self, xlsx_bytes: bytes) -> None:
        df = _parse_holdings_xlsx(xlsx_bytes)
        total = df["Weight"].sum()
        # SPDR weights are rounded to 6 decimals each — sum lands very
        # close to 1.0 (allow 0.5% slack for any cash/other-residual row).
        assert total == pytest.approx(1.0, abs=0.005)

    def test_row_count_is_reasonable(self, xlsx_bytes: bytes) -> None:
        # SPY tracks the S&P 500 (~500 holdings + a few class-share splits).
        df = _parse_holdings_xlsx(xlsx_bytes)
        assert 480 < df.height < 520

    def test_disclaimer_footer_excluded(self, xlsx_bytes: bytes) -> None:
        df = _parse_holdings_xlsx(xlsx_bytes)
        for name in df["Name"].to_list():
            assert "recommendation" not in (name or "").lower()
            assert "past performance" not in (name or "").lower()

    def test_missing_header_raises(self) -> None:
        """A workbook missing the Ticker/Weight header should error clearly."""
        import io
        import zipfile

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(
                "xl/sharedStrings.xml",
                '<?xml version="1.0"?>'
                '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                "<si><t>Something</t></si>"
                "</sst>",
            )
            zf.writestr(
                "xl/worksheets/sheet1.xml",
                '<?xml version="1.0"?>'
                '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                "<sheetData>"
                '<row r="1"><c r="A1" t="s"><v>0</v></c></row>'
                "</sheetData>"
                "</worksheet>",
            )
        with pytest.raises(ValueError, match="header row"):
            _parse_holdings_xlsx(buf.getvalue())


# ── Full-pipeline HTTP-mocked tests ──────────────────────────────────


@patch(_HTTPX_PATCH)
def test_get_holdings_calls_correct_url(
    mock_client_cls: MagicMock,
    xlsx_bytes: bytes,
) -> None:
    response = MagicMock(spec=httpx.Response)
    response.content = xlsx_bytes
    response.raise_for_status = MagicMock()
    client = MagicMock()
    client.get.return_value = response
    mock_client_cls.return_value.__enter__.return_value = client

    df = get_holdings("SPY")
    assert isinstance(df, pl.DataFrame)
    assert df.height > 0

    called_url = client.get.call_args.args[0]
    assert called_url == (
        "https://www.ssga.com/library-content/products/fund-data/etfs/us/"
        "holdings-daily-us-en-spy.xlsx"
    )


@patch(_HTTPX_PATCH)
def test_get_holdings_ticker_case_insensitive(
    mock_client_cls: MagicMock,
    xlsx_bytes: bytes,
) -> None:
    response = MagicMock(spec=httpx.Response)
    response.content = xlsx_bytes
    response.raise_for_status = MagicMock()
    client = MagicMock()
    client.get.return_value = response
    mock_client_cls.return_value.__enter__.return_value = client

    get_holdings("XLK")
    called_url = client.get.call_args.args[0]
    assert "/holdings-daily-us-en-xlk.xlsx" in called_url


@patch(_HTTPX_PATCH)
def test_get_holdings_default_ticker_is_spy(
    mock_client_cls: MagicMock,
    xlsx_bytes: bytes,
) -> None:
    response = MagicMock(spec=httpx.Response)
    response.content = xlsx_bytes
    response.raise_for_status = MagicMock()
    client = MagicMock()
    client.get.return_value = response
    mock_client_cls.return_value.__enter__.return_value = client

    get_holdings()
    assert "/holdings-daily-us-en-spy.xlsx" in client.get.call_args.args[0]


@patch(_HTTPX_PATCH)
def test_get_holdings_table_returns_gt(
    mock_client_cls: MagicMock,
    xlsx_bytes: bytes,
) -> None:
    response = MagicMock(spec=httpx.Response)
    response.content = xlsx_bytes
    response.raise_for_status = MagicMock()
    client = MagicMock()
    client.get.return_value = response
    mock_client_cls.return_value.__enter__.return_value = client

    gt = get_holdings_table("SPY")
    assert isinstance(gt, GT)
    html = gt.as_raw_html()
    assert "NVIDIA" in html
    assert "SPY Holdings" in html
