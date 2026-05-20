"""Tests for bestee.market."""

from unittest.mock import MagicMock, patch

import pytest
from great_tables import GT
from massive.rest.models import Agg, GroupedDailyAgg, TickerSnapshot

from bestee.market import get_latest_market_snapshot, get_market_snapshot

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


# ── Helpers for get_latest_market_snapshot ─────────────────────────


# A fixed nanosecond timestamp for 2025-07-18 14:00 UTC (10:00 ET).
_FAKE_UPDATED_NS = 1_752_847_200_000_000_000


def _make_snapshot(
    ticker: str = "AAPL",
    *,
    day_open: float = 190.0,
    day_high: float = 195.0,
    day_low: float = 189.0,
    day_close: float = 194.0,
    day_volume: float = 50_000_000.0,
    day_vwap: float = 192.5,
    updated: int = _FAKE_UPDATED_NS,
) -> MagicMock:
    """Create a mock TickerSnapshot with a day Agg."""
    day = MagicMock(spec=Agg)
    day.open = day_open
    day.high = day_high
    day.low = day_low
    day.close = day_close
    day.volume = day_volume
    day.vwap = day_vwap

    snap = MagicMock(spec=TickerSnapshot)
    snap.ticker = ticker
    snap.day = day
    snap.updated = updated
    return snap


class TestGetLatestMarketSnapshot:
    @patch(_CLIENT_PATCH)
    def test_returns_gt_object(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        mock_get_client.return_value.get_snapshot_all.return_value = [
            _make_snapshot("AAPL"),
        ]

        result = get_latest_market_snapshot()

        assert isinstance(result, GT)

    @patch(_CLIENT_PATCH)
    def test_single_api_call(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        mock_get_client.return_value.get_snapshot_all.return_value = [
            _make_snapshot("AAPL"),
            _make_snapshot("MSFT"),
        ]

        get_latest_market_snapshot()

        mock_get_client.return_value.get_snapshot_all.assert_called_once_with(
            "stocks",
            include_otc=False,
        )

    @patch(_CLIENT_PATCH)
    def test_contains_day_ohlcv(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        mock_get_client.return_value.get_snapshot_all.return_value = [
            _make_snapshot("AAPL", day_open=190.0, day_close=194.0),
        ]

        html = get_latest_market_snapshot().as_raw_html()

        assert "AAPL" in html
        assert "190" in html
        assert "194" in html

    @patch(_CLIENT_PATCH)
    def test_columns_include_trading_date(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """Column names should include the trading date from updated."""
        mock_get_client.return_value.get_snapshot_all.return_value = [
            _make_snapshot("AAPL"),
        ]

        html = get_latest_market_snapshot().as_raw_html()

        assert "Open (2025-07-18)" in html
        assert "Close (2025-07-18)" in html
        assert "Volume (2025-07-18)" in html

    @patch(_CLIENT_PATCH)
    def test_sorted_by_ticker(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        mock_get_client.return_value.get_snapshot_all.return_value = [
            _make_snapshot("MSFT"),
            _make_snapshot("AAPL"),
            _make_snapshot("GOOG"),
        ]

        html = get_latest_market_snapshot().as_raw_html()

        assert html.index("AAPL") < html.index("GOOG") < html.index("MSFT")

    @patch(_CLIENT_PATCH)
    def test_skips_snapshots_without_day(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        no_day = MagicMock(spec=TickerSnapshot)
        no_day.ticker = "BAD"
        no_day.day = None
        no_day.updated = _FAKE_UPDATED_NS

        mock_get_client.return_value.get_snapshot_all.return_value = [
            _make_snapshot("AAPL"),
            no_day,
        ]

        html = get_latest_market_snapshot().as_raw_html()

        assert "AAPL" in html
        assert "BAD" not in html

    @patch(_CLIENT_PATCH)
    def test_empty_result_raises(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """Empty snapshots leave the trading date unknowable — raise."""
        mock_get_client.return_value.get_snapshot_all.return_value = []

        with pytest.raises(RuntimeError, match="trading date"):
            get_latest_market_snapshot()
