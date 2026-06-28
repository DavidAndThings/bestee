"""Tests for bestee_compute.stocks.tickers -- SIC index + related companies.

The Massive client, ticker listing, and per-ticker detail fetch are mocked, so
these run offline and assert the cache filtering, index building, and the
related-company SIC comparison.
"""

from unittest.mock import MagicMock, patch

from massive.rest.models import TickerDetails

from bestee_compute.stocks.tickers import (
    _sic_key,
    build_ticker_sic_index,
    get_related_tickers,
    get_tickers_by_sic_code,
    related_tickers_sharing_sic,
)

_CLIENT_PATCH = "bestee_compute.stocks.tickers.get_client"
_DETAIL_PATCH = "bestee_compute.stocks.tickers._fetch_one_ticker_detail"
_FETCH_PATCH = "bestee_compute.stocks.tickers._fetch_tickers"


def _detail(symbol: str, sic_code: object) -> TickerDetails:
    detail = MagicMock(spec=TickerDetails)
    detail.ticker = symbol
    detail.sic_code = sic_code
    return detail


# ── get_tickers_by_sic_code (cache filter) ──────────────────────────


def test_filters_index_by_sic_code() -> None:
    index = {"AAA": "7372", "BBB": "3826", "CCC": "7372"}
    assert get_tickers_by_sic_code("7372", index=index) == ["AAA", "CCC"]


def test_accepts_int_and_normalizes_zero_padding() -> None:
    index = {"FARM": "100"}  # API may store "0100"; both normalize to "100"
    assert get_tickers_by_sic_code(100, index=index) == ["FARM"]
    assert _sic_key("0100") == _sic_key(100) == "100"


# ── build_ticker_sic_index ──────────────────────────────────────────


@patch(_CLIENT_PATCH)
@patch(_DETAIL_PATCH)
@patch(_FETCH_PATCH)
def test_build_index_omits_tickers_without_sic(
    mock_fetch: MagicMock, mock_detail: MagicMock, mock_get_client: MagicMock
) -> None:
    mock_fetch.return_value = [
        MagicMock(ticker="AAA"),
        MagicMock(ticker="BBB"),
        MagicMock(ticker="ETF"),  # funds carry no SIC code
    ]
    sic = {"AAA": 7372, "BBB": 3826, "ETF": None}
    mock_detail.side_effect = lambda _client, symbol: _detail(symbol, sic[symbol])
    assert build_ticker_sic_index() == {"AAA": "7372", "BBB": "3826"}


# ── related companies ───────────────────────────────────────────────


@patch(_CLIENT_PATCH)
def test_get_related_tickers(mock_get_client: MagicMock) -> None:
    client = mock_get_client.return_value
    client.get_related_companies.return_value = [
        MagicMock(ticker="MSFT"),
        MagicMock(ticker="GOOGL"),
    ]
    assert get_related_tickers("AAPL") == ["MSFT", "GOOGL"]


@patch(_CLIENT_PATCH)
def test_related_tickers_sharing_sic_from_index(mock_get_client: MagicMock) -> None:
    client = mock_get_client.return_value
    client.get_related_companies.return_value = [
        MagicMock(ticker="ADBE"),
        MagicMock(ticker="ORCL"),
        MagicMock(ticker="JPM"),
    ]
    index = {"MSFT": "7372", "ADBE": "7372", "ORCL": "7372", "JPM": "6021"}
    assert related_tickers_sharing_sic("MSFT", index=index) == ["ADBE", "ORCL"]


@patch(_DETAIL_PATCH)
@patch(_CLIENT_PATCH)
def test_related_sharing_sic_falls_back_to_live_details(
    mock_get_client: MagicMock, mock_detail: MagicMock
) -> None:
    client = mock_get_client.return_value
    client.get_related_companies.return_value = [MagicMock(ticker="NEW")]
    # "NEW" is absent from the index, so its SIC is fetched live.
    mock_detail.side_effect = lambda _client, symbol: _detail(symbol, 7372)
    assert related_tickers_sharing_sic("MSFT", index={"MSFT": "7372"}) == ["NEW"]


@patch(_CLIENT_PATCH)
def test_related_sharing_sic_empty_when_seed_sic_unknown(
    mock_get_client: MagicMock,
) -> None:
    client = mock_get_client.return_value
    client.get_related_companies.return_value = [MagicMock(ticker="MSFT")]
    # Seed not in the index and (mock) details return nothing useful.
    with patch(_DETAIL_PATCH, return_value=None):
        assert related_tickers_sharing_sic("UNKNOWN", index={}) == []
