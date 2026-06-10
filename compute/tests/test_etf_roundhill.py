"""Tests for bestee_compute.etf.roundhill — Roundhill ETF holdings scraper."""

from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import polars as pl
import pytest
from great_tables import GT

from bestee_compute.etf.roundhill import (
    _candidate_dates,
    _parse_holdings_csv,
    get_all_holdings,
    get_holdings,
    get_holdings_table,
    list_etfs,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "roundhill_holdings.csv"
_HTTPX_PATCH = "bestee_compute.etf.roundhill.httpx.Client"


@pytest.fixture
def csv_bytes() -> bytes:
    """Real Roundhill master holdings CSV sliced to three ETFs."""
    return _FIXTURE.read_bytes()


# ── CSV parser ───────────────────────────────────────────────────────


class TestParseHoldingsCsv:
    def test_columns_match_filepoint_schema(self, csv_bytes: bytes) -> None:
        df = _parse_holdings_csv(csv_bytes)
        assert df.columns == [
            "Date",
            "Account",
            "StockTicker",
            "CUSIP",
            "SecurityName",
            "Shares",
            "Price",
            "MarketValue",
            "Weightings",
            "NetAssets",
            "SharesOutstanding",
            "CreationUnits",
            "MoneyMarketFlag",
        ]

    def test_numeric_coercion(self, csv_bytes: bytes) -> None:
        df = _parse_holdings_csv(csv_bytes)
        assert df.schema["Shares"] == pl.Float64
        assert df.schema["Price"] == pl.Float64
        assert df.schema["MarketValue"] == pl.Float64
        assert df.schema["Weightings"] == pl.Float64
        assert df.schema["NetAssets"] == pl.Float64
        # Strings stay strings.
        assert df.schema["Account"] == pl.Utf8
        assert df.schema["StockTicker"] == pl.Utf8
        assert df.schema["SecurityName"] == pl.Utf8

    def test_weight_converted_to_decimal(self, csv_bytes: bytes) -> None:
        """``"5.53%"`` becomes ``0.0553``."""
        df = _parse_holdings_csv(csv_bytes)
        chat = df.filter(pl.col("Account") == "CHAT")
        # SK hynix is the first row in the CHAT slice (5.53%).
        sk = chat.filter(pl.col("SecurityName").str.contains("SK hynix"))
        assert sk.height == 1
        assert sk["Weightings"][0] == pytest.approx(0.0553, abs=1e-4)

    def test_covers_multiple_etfs(self, csv_bytes: bytes) -> None:
        df = _parse_holdings_csv(csv_bytes)
        assert set(df["Account"].unique().to_list()) == {"CHAT", "MAGX", "QDTE"}

    def test_missing_account_column_raises(self) -> None:
        with pytest.raises(ValueError, match="Account"):
            _parse_holdings_csv(b"Date,StockTicker\n05/26/2026,NVDA\n")


# ── Date walk-back ───────────────────────────────────────────────────


class TestCandidateDates:
    def test_starts_with_today_then_walks_back(self) -> None:
        stamps = _candidate_dates(date(2026, 5, 22))
        assert stamps[0] == "05222026"
        assert stamps[1] == "05212026"
        # Match the JS retry budget: today + 15 lookback days.
        assert len(stamps) == 16

    def test_handles_month_boundary(self) -> None:
        stamps = _candidate_dates(date(2026, 3, 2))
        assert stamps[0] == "03022026"
        assert stamps[2] == "02282026"


# ── Full-pipeline HTTP-mocked tests ──────────────────────────────────


def _mock_client(
    mock_client_cls: MagicMock,
    *,
    responses: list[MagicMock],
) -> MagicMock:
    """Wire ``mock_client_cls`` so ``client.get`` returns *responses* in order."""
    client = MagicMock()
    client.get.side_effect = responses
    mock_client_cls.return_value.__enter__.return_value = client
    return client


def _ok_response(content: bytes) -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.content = content
    response.raise_for_status = MagicMock()
    return response


def _error_response() -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.raise_for_status = MagicMock(
        side_effect=httpx.HTTPStatusError(
            "404", request=MagicMock(), response=MagicMock(status_code=404)
        )
    )
    return response


@patch(_HTTPX_PATCH)
def test_get_all_holdings_hits_dated_master_url(
    mock_client_cls: MagicMock, csv_bytes: bytes
) -> None:
    client = _mock_client(mock_client_cls, responses=[_ok_response(csv_bytes)])
    df = get_all_holdings(as_of=date(2026, 5, 22))

    assert isinstance(df, pl.DataFrame)
    assert df.height == 58  # fixture has 58 holdings rows across 3 ETFs

    called_url = client.get.call_args.args[0]
    assert called_url == (
        "https://www.roundhillinvestments.com/assets/data/"
        "FilepointRoundhill.40RU.RU_Holdings_05222026.csv"
    )


@patch(_HTTPX_PATCH)
def test_get_all_holdings_walks_back_on_404(
    mock_client_cls: MagicMock, csv_bytes: bytes
) -> None:
    """If today's file is missing, the function should retry the prior day."""
    client = _mock_client(
        mock_client_cls,
        responses=[_error_response(), _ok_response(csv_bytes)],
    )

    df = get_all_holdings(as_of=date(2026, 5, 22))
    assert df.height == 58
    assert client.get.call_count == 2
    # Second call should be the day before.
    second_url = client.get.call_args_list[1].args[0]
    assert "05212026" in second_url


@patch(_HTTPX_PATCH)
def test_get_all_holdings_walks_back_on_html_404(
    mock_client_cls: MagicMock, csv_bytes: bytes
) -> None:
    """The server returns 200 + HTML when a dated file is missing — the
    walk-back has to recognize that and keep going."""
    html_404 = _ok_response(b"<!doctype html><html>404</html>")
    client = _mock_client(
        mock_client_cls, responses=[html_404, _ok_response(csv_bytes)]
    )

    df = get_all_holdings(as_of=date(2026, 5, 22))
    assert df.height == 58
    assert client.get.call_count == 2


@patch(_HTTPX_PATCH)
def test_get_all_holdings_gives_up_after_lookback_window(
    mock_client_cls: MagicMock,
) -> None:
    """16 consecutive failures (today + 15 lookback days) should re-raise."""
    client = _mock_client(mock_client_cls, responses=[_error_response()] * 16)

    with pytest.raises(httpx.HTTPStatusError):
        get_all_holdings(as_of=date(2026, 5, 22))
    assert client.get.call_count == 16


@patch(_HTTPX_PATCH)
def test_get_all_holdings_gives_up_on_html_only_window(
    mock_client_cls: MagicMock,
) -> None:
    """All-200-but-HTML responses should raise a clear ValueError."""
    html_404 = _ok_response(b"<!doctype html><html>404</html>")
    _mock_client(mock_client_cls, responses=[html_404] * 16)

    with pytest.raises(ValueError, match="lookback window"):
        get_all_holdings(as_of=date(2026, 5, 22))


@patch(_HTTPX_PATCH)
def test_list_etfs_returns_sorted_unique_tickers(
    mock_client_cls: MagicMock, csv_bytes: bytes
) -> None:
    _mock_client(mock_client_cls, responses=[_ok_response(csv_bytes)])
    assert list_etfs(as_of=date(2026, 5, 22)) == ["CHAT", "MAGX", "QDTE"]


@patch(_HTTPX_PATCH)
def test_get_holdings_filters_to_one_etf(
    mock_client_cls: MagicMock, csv_bytes: bytes
) -> None:
    _mock_client(mock_client_cls, responses=[_ok_response(csv_bytes)])
    df = get_holdings("CHAT", as_of=date(2026, 5, 22))
    assert set(df["Account"].unique().to_list()) == {"CHAT"}
    assert df.height > 0


@patch(_HTTPX_PATCH)
def test_get_holdings_ticker_case_insensitive(
    mock_client_cls: MagicMock, csv_bytes: bytes
) -> None:
    _mock_client(mock_client_cls, responses=[_ok_response(csv_bytes)])
    df = get_holdings("chat", as_of=date(2026, 5, 22))
    assert df["Account"].unique().to_list() == ["CHAT"]


@patch(_HTTPX_PATCH)
def test_get_holdings_unknown_ticker_raises(
    mock_client_cls: MagicMock, csv_bytes: bytes
) -> None:
    _mock_client(mock_client_cls, responses=[_ok_response(csv_bytes)])
    with pytest.raises(ValueError, match="ZZZZ"):
        get_holdings("ZZZZ", as_of=date(2026, 5, 22))


@patch(_HTTPX_PATCH)
def test_get_holdings_table_returns_gt(
    mock_client_cls: MagicMock, csv_bytes: bytes
) -> None:
    _mock_client(mock_client_cls, responses=[_ok_response(csv_bytes)])
    gt = get_holdings_table("CHAT", as_of=date(2026, 5, 22))
    assert isinstance(gt, GT)
    html = gt.as_raw_html()
    assert "CHAT Holdings" in html
    assert "SK hynix" in html
