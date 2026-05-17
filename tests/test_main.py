"""Tests for bestee modules."""

from unittest.mock import MagicMock, patch

import pytest
from massive.rest.models import Ticker

from bestee.tickers import get_all_tickers


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


class TestGetAllTickers:
    """Tests for the get_all_tickers function."""

    @patch("bestee.tickers.load_dotenv")
    def test_raises_without_api_key(
        self,
        _mock_dotenv: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Should raise RuntimeError when no API key is available."""
        monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="No API key provided"):
            get_all_tickers()

    @patch("bestee.tickers.RESTClient")
    def test_returns_all_tickers(
        self,
        mock_client_cls: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Should collect all pages into a single list."""
        monkeypatch.setenv("MASSIVE_API_KEY", "test-key")

        fake_tickers = [
            _make_fake_ticker(ticker="AAPL", name="Apple Inc."),
            _make_fake_ticker(ticker="MSFT", name="Microsoft Corporation"),
            _make_fake_ticker(ticker="GOOG", name="Alphabet Inc."),
        ]
        mock_client_cls.return_value.list_tickers.return_value = iter(fake_tickers)

        result = get_all_tickers()

        assert len(result) == 3
        assert result[0].ticker == "AAPL"
        assert result[1].ticker == "MSFT"
        assert result[2].ticker == "GOOG"

    @patch("bestee.tickers.RESTClient")
    def test_passes_filters_to_client(
        self,
        mock_client_cls: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Should forward market/type/active filters to the SDK."""
        monkeypatch.setenv("MASSIVE_API_KEY", "test-key")
        mock_client_cls.return_value.list_tickers.return_value = iter([])

        get_all_tickers(market="crypto", ticker_type="CRYPTO", active=False)

        mock_client_cls.return_value.list_tickers.assert_called_once_with(
            market="crypto",
            type="CRYPTO",
            active=False,
            limit=1000,
        )

    @patch("bestee.tickers.RESTClient")
    def test_uses_explicit_api_key(
        self,
        mock_client_cls: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Should prefer an explicit api_key over the env var."""
        monkeypatch.setenv("MASSIVE_API_KEY", "env-key")
        mock_client_cls.return_value.list_tickers.return_value = iter([])

        get_all_tickers(api_key="explicit-key")

        mock_client_cls.assert_called_once_with(api_key="explicit-key")
