"""Tests for bestee modules."""

from unittest.mock import MagicMock, patch

import pytest
from great_tables import GT
from massive.rest.models import Ticker

from bestee.tickers import _fetch_tickers, get_all_tickers

# Patch target for the client factory used by tickers.py
_CLIENT_PATCH = "bestee.tickers.get_client"
_DOTENV_PATCH = "bestee.client.load_dotenv"


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
