"""Tests for bestee modules."""

from unittest.mock import MagicMock, patch

import pytest
from great_tables import GT
from massive.rest.models import Ticker, TickerDetails

from bestee_compute.stocks.tickers import (
    _fetch_tickers,
    get_all_tickers,
    get_ticker_detail,
    get_ticker_details,
)

# Patch target for the client factory used by tickers.py
_CLIENT_PATCH = "bestee_compute.stocks.tickers.get_client"
_DOTENV_PATCH = "bestee_compute.client.load_dotenv"


def _make_fake_ticker(**kwargs: str | bool) -> MagicMock:
    """Create a mock Ticker with sensible defaults."""
    defaults: dict[str, str | bool] = {
        "active": True,
        "ticker": "AAPL",
        "name": "Apple Inc.",
        "market": "stocks",
        "locale": "us",
        "currency_name": "usd",
        "primary_exchange": "XNAS",
        "type": "CS",
    }
    defaults.update(kwargs)
    mock = MagicMock(spec=Ticker)
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


class TestFetchTickers:
    """Tests for the _fetch_tickers helper."""

    @patch(_DOTENV_PATCH)
    def test_raises_without_api_key(
        self,
        _mock_dotenv: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Should raise RuntimeError when no API key is available."""
        monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="No API key provided"):
            _fetch_tickers()

    @patch(_CLIENT_PATCH)
    def test_returns_all_tickers(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """Should collect all pages into a single list."""
        fake_tickers = [
            _make_fake_ticker(ticker="AAPL", name="Apple Inc."),
            _make_fake_ticker(ticker="MSFT", name="Microsoft Corporation"),
            _make_fake_ticker(ticker="GOOG", name="Alphabet Inc."),
        ]
        mock_get_client.return_value.list_tickers.return_value = iter(fake_tickers)

        result = _fetch_tickers()

        assert len(result) == 3
        assert result[0].ticker == "AAPL"
        assert result[1].ticker == "MSFT"
        assert result[2].ticker == "GOOG"

    @patch(_CLIENT_PATCH)
    def test_passes_filters_to_client(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """Should forward market/type/active filters to the SDK."""
        mock_get_client.return_value.list_tickers.return_value = iter([])

        _fetch_tickers(market="crypto", ticker_type="CRYPTO", active=False)

        mock_get_client.return_value.list_tickers.assert_called_once_with(
            market="crypto",
            type="CRYPTO",
            active=False,
            limit=1000,
        )

    @patch(_CLIENT_PATCH)
    def test_forwards_api_key(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """Should forward the api_key to get_client."""
        mock_get_client.return_value.list_tickers.return_value = iter([])

        _fetch_tickers(api_key="explicit-key")

        mock_get_client.assert_called_once_with("explicit-key")


class TestGetAllTickers:
    """Tests for the get_all_tickers GT wrapper."""

    @patch(_CLIENT_PATCH)
    def test_returns_gt_object(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """Should return a GT display table."""
        fake_tickers = [
            _make_fake_ticker(ticker="AAPL", name="Apple Inc."),
            _make_fake_ticker(ticker="MSFT", name="Microsoft Corporation"),
        ]
        mock_get_client.return_value.list_tickers.return_value = iter(fake_tickers)

        result = get_all_tickers()

        assert isinstance(result, GT)

    @patch(_CLIENT_PATCH)
    def test_gt_contains_ticker_data(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """The rendered HTML should contain the ticker symbols."""
        fake_tickers = [
            _make_fake_ticker(ticker="AAPL", name="Apple Inc."),
            _make_fake_ticker(ticker="GOOG", name="Alphabet Inc."),
        ]
        mock_get_client.return_value.list_tickers.return_value = iter(fake_tickers)

        result = get_all_tickers()
        html = result.as_raw_html()

        assert "AAPL" in html
        assert "GOOG" in html
        assert "Apple Inc." in html
        assert "Alphabet Inc." in html


# ── Helpers for get_ticker_details ───────────────────────────────────


def _make_fake_details(**kwargs: str | float | int | bool) -> MagicMock:
    """Create a mock TickerDetails with sensible defaults."""
    defaults: dict[str, str | float | int | bool] = {
        "ticker": "AAPL",
        "name": "Apple Inc.",
        "description": "Apple designs consumer electronics.",
        "type": "CS",
        "market": "stocks",
        "locale": "us",
        "primary_exchange": "XNAS",
        "currency_name": "usd",
        "cik": "0000320193",
        "composite_figi": "BBG000B9XRY4",
        "share_class_figi": "BBG001S5N8V8",
        "sic_code": "3571",
        "sic_description": "ELECTRONIC COMPUTERS",
        "market_cap": 4_400_000_000_000.0,
        "share_class_shares_outstanding": 14_687_356_000,
        "weighted_shares_outstanding": 14_687_356_000,
        "total_employees": 166_000,
        "list_date": "1980-12-12",
        "homepage_url": "https://www.apple.com",
        "phone_number": "(408) 996-1010",
        "ticker_root": "AAPL",
    }
    defaults.update(kwargs)
    mock = MagicMock(spec=TickerDetails)
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


class TestGetTickerDetails:
    """Tests for the get_ticker_details function."""

    @patch(_CLIENT_PATCH)
    def test_returns_gt_object(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """Should return a GT display table."""
        mock_get_client.return_value.get_ticker_details.return_value = (
            _make_fake_details(ticker="AAPL")
        )

        result = get_ticker_details(["AAPL"])

        assert isinstance(result, GT)

    @patch(_CLIENT_PATCH)
    def test_one_call_per_ticker(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """Should call get_ticker_details once per ticker."""
        mock_get_client.return_value.get_ticker_details.return_value = (
            _make_fake_details()
        )

        get_ticker_details(["AAPL", "MSFT", "GOOG"])

        assert mock_get_client.return_value.get_ticker_details.call_count == 3

    @patch(_CLIENT_PATCH)
    def test_contains_expected_fields(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """The HTML should contain key field values."""
        mock_get_client.return_value.get_ticker_details.return_value = (
            _make_fake_details(
                ticker="AAPL",
                name="Apple Inc.",
                cik="0000320193",
                sic_description="ELECTRONIC COMPUTERS",
            )
        )

        result = get_ticker_details(["AAPL"])
        html = result.as_raw_html()

        assert "Apple Inc." in html
        assert "0000320193" in html
        assert "ELECTRONIC COMPUTERS" in html

    @patch(_CLIENT_PATCH)
    def test_multiple_tickers_as_columns(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """Each ticker should appear as a column header."""
        mock_get_client.return_value.get_ticker_details.side_effect = [
            _make_fake_details(ticker="AAPL", name="Apple Inc."),
            _make_fake_details(ticker="MSFT", name="Microsoft Corporation"),
        ]

        result = get_ticker_details(["AAPL", "MSFT"])
        html = result.as_raw_html()

        assert "AAPL" in html
        assert "MSFT" in html
        assert "Apple Inc." in html
        assert "Microsoft Corporation" in html


class TestGetTickerDetail:
    """Single-ticker public wrapper shared with other modules (e.g. etf.issuer)."""

    @patch(_CLIENT_PATCH)
    def test_returns_ticker_details_on_success(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        details = _make_fake_details(ticker="AAPL", name="Apple Inc.")
        mock_get_client.return_value.get_ticker_details.return_value = details

        result = get_ticker_detail("AAPL")
        assert result is details
        mock_get_client.return_value.get_ticker_details.assert_called_once_with("AAPL")

    @patch(_CLIENT_PATCH)
    def test_returns_none_on_sdk_exception(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        mock_get_client.return_value.get_ticker_details.side_effect = RuntimeError(
            "timeout"
        )
        assert get_ticker_detail("AAPL") is None

    @patch(_CLIENT_PATCH)
    def test_returns_none_for_unexpected_response_type(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        mock_get_client.return_value.get_ticker_details.return_value = "not details"
        assert get_ticker_detail("AAPL") is None

    @patch(_CLIENT_PATCH)
    def test_threads_api_key_through(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        mock_get_client.return_value.get_ticker_details.return_value = (
            _make_fake_details()
        )
        get_ticker_detail("AAPL", api_key="xyz")
        mock_get_client.assert_called_once_with("xyz")
