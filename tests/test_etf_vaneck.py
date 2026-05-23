"""Tests for bestee.etf.vaneck — VanEck ETF holdings scraper."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import polars as pl
import pytest
from great_tables import GT

from bestee.etf._xlsx import col_letter_to_index
from bestee.etf.vaneck import (
    _parse_holdings_xlsx,
    get_holdings,
    get_holdings_table,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "smh_holdings.xlsx"
_HTTPX_PATCH = "bestee.etf.vaneck.httpx.Client"


@pytest.fixture
def xlsx_bytes() -> bytes:
    """Real SMH holdings xlsx captured from VanEck."""
    return _FIXTURE.read_bytes()


# ── Cell letter helper ───────────────────────────────────────────────


class TestColLetterToIndex:
    def test_single_letter(self) -> None:
        assert col_letter_to_index("A") == 0
        assert col_letter_to_index("B") == 1
        assert col_letter_to_index("I") == 8
        assert col_letter_to_index("Z") == 25

    def test_double_letter(self) -> None:
        assert col_letter_to_index("AA") == 26
        assert col_letter_to_index("AB") == 27
        assert col_letter_to_index("AZ") == 51
        assert col_letter_to_index("BA") == 52


# ── xlsx parser ──────────────────────────────────────────────────────


class TestParseHoldingsXlsx:
    def test_shape_and_schema(self, xlsx_bytes: bytes) -> None:
        df = _parse_holdings_xlsx(xlsx_bytes)
        # 27 actual holdings (25 stocks + 2 cash rows), 9 columns.
        assert df.shape == (27, 9)
        assert df.columns == [
            "Number",
            "Ticker",
            "Holding Name",
            "Identifier (FIGI)",
            "Shares",
            "Asset Class",
            "Market Value (US$)",
            "Notional Value",
            "% of Net Assets",
        ]

    def test_numeric_coercion(self, xlsx_bytes: bytes) -> None:
        df = _parse_holdings_xlsx(xlsx_bytes)
        assert df.schema["Shares"] == pl.Int64
        assert df.schema["Market Value (US$)"] == pl.Float64
        assert df.schema["% of Net Assets"] == pl.Float64
        # Other columns remain string.
        assert df.schema["Ticker"] == pl.Utf8
        assert df.schema["Holding Name"] == pl.Utf8

    def test_top_holding_is_nvda(self, xlsx_bytes: bytes) -> None:
        df = _parse_holdings_xlsx(xlsx_bytes)
        top = df.row(0, named=True)
        assert top["Ticker"] == "NVDA"
        assert top["Holding Name"] == "Nvidia Corp"
        assert top["Shares"] == 50_096_031
        assert top["Market Value (US$)"] == pytest.approx(10_996_579_764.81)
        # 16.70% → 0.167
        assert top["% of Net Assets"] == pytest.approx(0.167, abs=1e-4)

    def test_weights_sum_to_one(self, xlsx_bytes: bytes) -> None:
        df = _parse_holdings_xlsx(xlsx_bytes)
        total = df["% of Net Assets"].sum()
        # Allow small rounding (each weight is reported to 2 decimals).
        assert total == pytest.approx(1.0, abs=0.01)

    def test_cash_rows_preserved(self, xlsx_bytes: bytes) -> None:
        """The fixture has two cash entries — they should be in the output."""
        df = _parse_holdings_xlsx(xlsx_bytes)
        asset_classes = set(df["Asset Class"].to_list())
        assert "Stock" in asset_classes
        # Cash entries appear with either "Cash" or "Cash Balance" asset
        # class depending on row layout — assert at least one.
        assert any("Cash" in ac for ac in asset_classes)

    def test_disclaimer_footer_excluded(self, xlsx_bytes: bytes) -> None:
        """Number of rows should be exactly the holding count — no footer text."""
        df = _parse_holdings_xlsx(xlsx_bytes)
        # No row should contain disclaimer phrases.
        for name in df["Holding Name"].to_list():
            assert "recommendation" not in (name or "").lower()
            assert "not " not in (name or "")[:10].lower()

    def test_missing_ticker_column_raises(self) -> None:
        """A workbook without a Ticker header column should error clearly."""
        import io
        import zipfile

        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            # Minimal xlsx with no Ticker column.
            zf.writestr(
                "xl/sharedStrings.xml",
                '<?xml version="1.0"?>'
                '<x:sst xmlns:x="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                "<x:si><x:t>Some Other Header</x:t></x:si>"
                "</x:sst>",
            )
            zf.writestr(
                "xl/worksheets/sheet1.xml",
                '<?xml version="1.0"?>'
                '<x:worksheet xmlns:x="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                "<x:sheetData>"
                '<x:row r="1"><x:c r="A1" t="s"><x:v>0</x:v></x:c></x:row>'
                "</x:sheetData>"
                "</x:worksheet>",
            )
        with pytest.raises(ValueError, match="Ticker"):
            _parse_holdings_xlsx(buf.getvalue())


# ── Full-pipeline HTTP-mocked tests ──────────────────────────────────


@patch(_HTTPX_PATCH)
def test_get_holdings_calls_correct_url(
    mock_client_cls: MagicMock,
    xlsx_bytes: bytes,
) -> None:
    """Ticker is lowercased into the URL; the xlsx is fetched + parsed."""
    response = MagicMock(spec=httpx.Response)
    response.content = xlsx_bytes
    response.raise_for_status = MagicMock()

    client = MagicMock()
    client.get.return_value = response
    mock_client_cls.return_value.__enter__.return_value = client

    df = get_holdings("SMH")

    assert isinstance(df, pl.DataFrame)
    assert df.height == 27

    called_url = client.get.call_args.args[0]
    assert called_url == (
        "https://www.vaneck.com/us/en/etf/equity/smh/holdings/download/xlsx/"
    )


@patch(_HTTPX_PATCH)
def test_get_holdings_ticker_case_insensitive(
    mock_client_cls: MagicMock,
    xlsx_bytes: bytes,
) -> None:
    """Mixed-case ticker is normalized to lowercase in the URL."""
    response = MagicMock(spec=httpx.Response)
    response.content = xlsx_bytes
    response.raise_for_status = MagicMock()
    client = MagicMock()
    client.get.return_value = response
    mock_client_cls.return_value.__enter__.return_value = client

    get_holdings("MoAt")

    called_url = client.get.call_args.args[0]
    assert "/moat/" in called_url
    assert "/MoAt/" not in called_url


@patch(_HTTPX_PATCH)
def test_get_holdings_default_ticker_is_smh(
    mock_client_cls: MagicMock,
    xlsx_bytes: bytes,
) -> None:
    """Calling without arguments hits the SMH endpoint."""
    response = MagicMock(spec=httpx.Response)
    response.content = xlsx_bytes
    response.raise_for_status = MagicMock()
    client = MagicMock()
    client.get.return_value = response
    mock_client_cls.return_value.__enter__.return_value = client

    df = get_holdings()
    assert isinstance(df, pl.DataFrame)
    assert df.height == 27
    assert "/smh/" in client.get.call_args.args[0]


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

    gt = get_holdings_table("SMH")
    assert isinstance(gt, GT)
    html = gt.as_raw_html()
    # Top holding should appear in the rendered table.
    assert "NVDA" in html
    assert "SMH Holdings" in html
