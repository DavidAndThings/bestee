"""Tests for bestee.etf.invesco — Invesco QQQ holdings scraper."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import polars as pl
import pytest
from great_tables import GT

from bestee.etf.invesco import (
    _parse_holdings_json,
    get_holdings,
    get_holdings_table,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "qqq_holdings.json"
_HTTPX_PATCH = "bestee.etf.invesco.httpx.Client"


@pytest.fixture
def json_bytes() -> bytes:
    """Real QQQ holdings JSON captured from dng-api.invesco.com."""
    return _FIXTURE.read_bytes()


# ── JSON parser ──────────────────────────────────────────────────────


class TestParseHoldingsJson:
    def test_columns_match_invesco_schema(self, json_bytes: bytes) -> None:
        df = _parse_holdings_json(json_bytes)
        assert df.columns == [
            "Ticker",
            "Issuer Name",
            "Units",
            "% of Net Assets",
            "CUSIP",
            "Security Type",
            "Currency",
        ]

    def test_numeric_coercion(self, json_bytes: bytes) -> None:
        df = _parse_holdings_json(json_bytes)
        assert df.schema["Units"] == pl.Float64
        assert df.schema["% of Net Assets"] == pl.Float64
        # Strings stay strings.
        assert df.schema["Ticker"] == pl.Utf8
        assert df.schema["Issuer Name"] == pl.Utf8
        assert df.schema["CUSIP"] == pl.Utf8

    def test_top_holding_is_nvda(self, json_bytes: bytes) -> None:
        df = _parse_holdings_json(json_bytes)
        top = df.row(0, named=True)
        assert top["Ticker"] == "NVDA"
        assert top["Issuer Name"] == "NVIDIA Corp"
        # JSON reports 8.757223 in percent units → 0.08757... decimal.
        assert top["% of Net Assets"] == pytest.approx(0.08757, abs=1e-4)

    def test_weight_converted_to_decimal(self, json_bytes: bytes) -> None:
        """Largest weight should be well under 1.0 (decimal form)."""
        df = _parse_holdings_json(json_bytes)
        weights = df["% of Net Assets"].to_list()
        assert max(w for w in weights if w is not None) < 0.5

    def test_missing_holdings_key_raises(self) -> None:
        with pytest.raises(ValueError, match="holdings"):
            _parse_holdings_json(b'{"effectiveDate": "2026-05-21"}')


# ── HTTP-mocked full-pipeline tests ──────────────────────────────────


def _ok_response(content: bytes) -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.content = content
    response.raise_for_status = MagicMock()
    return response


@patch(_HTTPX_PATCH)
def test_get_holdings_calls_correct_url(
    mock_client_cls: MagicMock, json_bytes: bytes
) -> None:
    client = MagicMock()
    client.get.return_value = _ok_response(json_bytes)
    mock_client_cls.return_value.__enter__.return_value = client

    df = get_holdings("QQQ")
    assert isinstance(df, pl.DataFrame)
    assert df.height > 0

    called_url = client.get.call_args.args[0]
    assert called_url == (
        "https://dng-api.invesco.com/cache/v1/accounts/en_US/"
        "shareclasses/QQQ/holdings/fund"
        "?idType=ticker&interval=monthly&productType=ETF"
    )


@patch(_HTTPX_PATCH)
def test_get_holdings_ticker_case_insensitive(
    mock_client_cls: MagicMock, json_bytes: bytes
) -> None:
    client = MagicMock()
    client.get.return_value = _ok_response(json_bytes)
    mock_client_cls.return_value.__enter__.return_value = client

    get_holdings("qqq")
    assert "/QQQ/" in client.get.call_args.args[0]


@patch(_HTTPX_PATCH)
def test_get_holdings_default_ticker_is_qqq(
    mock_client_cls: MagicMock, json_bytes: bytes
) -> None:
    client = MagicMock()
    client.get.return_value = _ok_response(json_bytes)
    mock_client_cls.return_value.__enter__.return_value = client

    df = get_holdings()
    assert "/QQQ/" in client.get.call_args.args[0]
    assert df.height > 0


def test_get_holdings_rejects_other_invesco_tickers() -> None:
    """Other Invesco ETFs sit behind a bot-protected backend — should
    error early with a clear message rather than hit the network."""
    with pytest.raises(ValueError, match="QQQ"):
        get_holdings("RSP")
    with pytest.raises(ValueError, match="QQQ"):
        get_holdings("SPLV")


@patch(_HTTPX_PATCH)
def test_get_holdings_table_returns_gt(
    mock_client_cls: MagicMock, json_bytes: bytes
) -> None:
    client = MagicMock()
    client.get.return_value = _ok_response(json_bytes)
    mock_client_cls.return_value.__enter__.return_value = client

    gt = get_holdings_table("QQQ")
    assert isinstance(gt, GT)
    html = gt.as_raw_html()
    assert "QQQ Holdings" in html
    assert "NVDA" in html
