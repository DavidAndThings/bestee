"""Tests for bestee_compute.stocks.tickers -- SIC-code filtering.

The Massive client and per-ticker detail fetch are mocked, so these run offline
and assert the filtering/normalization logic of ``get_tickers_by_sic_code``.
"""

from unittest.mock import MagicMock, patch

from massive.rest.models import TickerDetails

from bestee_compute.stocks.tickers import get_tickers_by_sic_code

_CLIENT_PATCH = "bestee_compute.stocks.tickers.get_client"
_DETAIL_PATCH = "bestee_compute.stocks.tickers._fetch_one_ticker_detail"


def _detail(symbol: str, sic_code: object) -> TickerDetails:
    detail = MagicMock(spec=TickerDetails)
    detail.ticker = symbol
    detail.sic_code = sic_code
    return detail


@patch(_CLIENT_PATCH)
@patch(_DETAIL_PATCH)
def test_filters_candidates_by_sic_code(
    mock_detail: MagicMock, mock_get_client: MagicMock
) -> None:
    sic_by_symbol = {"AAA": 7372, "BBB": 3826, "CCC": 7372, "DDD": None}
    mock_detail.side_effect = lambda _client, symbol: _detail(
        symbol, sic_by_symbol[symbol]
    )
    result = get_tickers_by_sic_code("7372", tickers=list(sic_by_symbol))
    assert result == ["AAA", "CCC"]


@patch(_CLIENT_PATCH)
@patch(_DETAIL_PATCH)
def test_accepts_int_and_normalizes_zero_padding(
    mock_detail: MagicMock, mock_get_client: MagicMock
) -> None:
    mock_detail.side_effect = lambda _client, symbol: _detail(symbol, "0100")
    # ``int`` input vs zero-padded API value still matches.
    assert get_tickers_by_sic_code(100, tickers=["FARM"]) == ["FARM"]


@patch(_CLIENT_PATCH)
@patch(_DETAIL_PATCH)
def test_skips_failed_detail_lookups(
    mock_detail: MagicMock, mock_get_client: MagicMock
) -> None:
    def _lookup(_client: object, symbol: str) -> TickerDetails | None:
        return None if symbol == "BAD" else _detail(symbol, 7372)

    mock_detail.side_effect = _lookup
    assert get_tickers_by_sic_code(7372, tickers=["AAA", "BAD"]) == ["AAA"]


@patch(_CLIENT_PATCH)
@patch(_DETAIL_PATCH)
def test_fetches_universe_when_no_candidates_given(
    mock_detail: MagicMock, mock_get_client: MagicMock
) -> None:
    with patch("bestee_compute.stocks.tickers._fetch_tickers") as mock_fetch:
        mock_fetch.return_value = [
            MagicMock(ticker="AAA"),
            MagicMock(ticker="BBB"),
        ]
        mock_detail.side_effect = lambda _client, symbol: _detail(
            symbol, 7372 if symbol == "AAA" else 1234
        )
        result = get_tickers_by_sic_code(7372)
    assert result == ["AAA"]
    mock_fetch.assert_called_once()
