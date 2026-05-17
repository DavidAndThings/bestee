"""Tests for bestee.market."""

from unittest.mock import MagicMock, patch

from great_tables import GT
from massive.rest.models import GroupedDailyAgg

from bestee.market import get_market_snapshot

_CLIENT_PATCH = "bestee.market.get_client"


def _make_bar(**kwargs: str | float | int | None) -> MagicMock:
    """Create a mock GroupedDailyAgg."""
    defaults: dict[str, str | float | int | None] = {
        "ticker": "AAPL",
        "open": 190.0,
        "high": 195.0,
        "low": 189.0,
        "close": 194.0,
        "volume": 50_000_000,
        "vwap": 192.5,
        "transactions": 500_000,
        "timestamp": None,
        "otc": None,
    }
    defaults.update(kwargs)
    mock = MagicMock(spec=GroupedDailyAgg)
    for k, v in defaults.items():
        setattr(mock, k, v)
    return mock


class TestGetMarketSnapshot:
    @patch(_CLIENT_PATCH)
    def test_returns_gt_object(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        mock_get_client.return_value.get_grouped_daily_aggs.return_value = [
            _make_bar(ticker="AAPL"),
        ]

        result = get_market_snapshot("2025-07-11")

        assert isinstance(result, GT)

    @patch(_CLIENT_PATCH)
    def test_single_api_call(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """Should make exactly one API call regardless of ticker count."""
        mock_get_client.return_value.get_grouped_daily_aggs.return_value = [
            _make_bar(ticker="AAPL"),
            _make_bar(ticker="MSFT"),
            _make_bar(ticker="GOOG"),
        ]

        get_market_snapshot("2025-07-11")

        mock_get_client.return_value.get_grouped_daily_aggs.assert_called_once_with(
            "2025-07-11",
            adjusted=True,
            include_otc=False,
        )

    @patch(_CLIENT_PATCH)
    def test_html_contains_ticker_and_ohlcv(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        mock_get_client.return_value.get_grouped_daily_aggs.return_value = [
            _make_bar(ticker="AAPL", open=190.0, close=194.0, volume=50_000_000),
        ]

        html = get_market_snapshot("2025-07-11").as_raw_html()

        assert "AAPL" in html
        assert "190" in html
        assert "194" in html

    @patch(_CLIENT_PATCH)
    def test_columns_include_date(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """Column names should be suffixed with the date."""
        mock_get_client.return_value.get_grouped_daily_aggs.return_value = [
            _make_bar(ticker="AAPL"),
        ]

        html = get_market_snapshot("2025-07-11").as_raw_html()

        assert "Open (2025-07-11)" in html
        assert "Close (2025-07-11)" in html
        assert "Volume (2025-07-11)" in html

    @patch(_CLIENT_PATCH)
    def test_rows_sorted_by_ticker(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        mock_get_client.return_value.get_grouped_daily_aggs.return_value = [
            _make_bar(ticker="MSFT"),
            _make_bar(ticker="AAPL"),
            _make_bar(ticker="GOOG"),
        ]

        html = get_market_snapshot("2025-07-11").as_raw_html()

        # AAPL should appear before GOOG, GOOG before MSFT.
        assert html.index("AAPL") < html.index("GOOG") < html.index("MSFT")

    @patch(_CLIENT_PATCH)
    def test_empty_trading_day(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """Weekends/holidays return an empty list — should still produce a GT."""
        mock_get_client.return_value.get_grouped_daily_aggs.return_value = []

        result = get_market_snapshot("2025-07-12")

        assert isinstance(result, GT)

    @patch(_CLIENT_PATCH)
    def test_forwards_options(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        mock_get_client.return_value.get_grouped_daily_aggs.return_value = []

        get_market_snapshot("2025-07-11", adjusted=False, include_otc=True)

        mock_get_client.return_value.get_grouped_daily_aggs.assert_called_once_with(
            "2025-07-11",
            adjusted=False,
            include_otc=True,
        )
