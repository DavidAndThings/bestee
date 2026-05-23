"""Tests for bestee.stocks.market."""

import datetime as dt
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
from great_tables import GT
from massive.rest.models import Agg, GroupedDailyAgg, TickerSnapshot
from massive.rest.models.common import Sort

from bestee.stocks.market import (
    get_latest_market_snapshot,
    get_market_snapshot,
    get_ohlc,
    get_ohlc_table,
)

_CLIENT_PATCH = "bestee.stocks.market.get_client"


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


# ── Helpers for get_ohlc ───────────────────────────────────────────


def _make_agg(
    *,
    timestamp: int | None = 1_736_172_000_000,  # 2025-01-06 14:00 UTC
    open_: float | None = 100.0,
    high: float | None = 102.0,
    low: float | None = 99.5,
    close: float | None = 101.0,
    volume: float | None = 1_000_000.0,
    vwap: float | None = 100.5,
    transactions: int | None = 5000,
) -> MagicMock:
    mock = MagicMock(spec=Agg)
    mock.timestamp = timestamp
    mock.open = open_
    mock.high = high
    mock.low = low
    mock.close = close
    mock.volume = volume
    mock.vwap = vwap
    mock.transactions = transactions
    mock.otc = None
    return mock


class TestGetOhlc:
    @patch(_CLIENT_PATCH)
    def test_returns_polars_dataframe_with_expected_schema(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        mock_get_client.return_value.get_aggs.return_value = [_make_agg()]

        df = get_ohlc("AAPL", "2025-01-01", "2025-01-31")

        assert isinstance(df, pl.DataFrame)
        assert df.columns == [
            "Timestamp",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "VWAP",
            "Transactions",
        ]
        assert df.schema["Timestamp"] == pl.Datetime("ms", time_zone="UTC")
        assert df.schema["Open"] == pl.Float64
        assert df.schema["Volume"] == pl.Float64
        assert df.schema["Transactions"] == pl.Int64

    @patch(_CLIENT_PATCH)
    def test_timestamp_converted_to_utc_datetime(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """Massive returns ms-epoch ints; we should expose a tz-aware datetime."""
        # 2025-01-06 14:00:00 UTC = 1736172000000 ms
        mock_get_client.return_value.get_aggs.return_value = [
            _make_agg(timestamp=1_736_172_000_000)
        ]

        df = get_ohlc("AAPL", "2025-01-01", "2025-01-31")

        ts = df["Timestamp"][0]
        assert ts is not None
        assert ts == dt.datetime(2025, 1, 6, 14, 0, tzinfo=dt.UTC)

    @patch(_CLIENT_PATCH)
    def test_defaults_send_day_bars_asc(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """Default call should request 1-day bars in ascending order."""
        mock_get_client.return_value.get_aggs.return_value = []

        get_ohlc("aapl", "2025-01-01", "2025-01-31")

        mock_get_client.return_value.get_aggs.assert_called_once_with(
            "AAPL",  # ticker uppercased
            1,
            "day",
            "2025-01-01",
            "2025-01-31",
            adjusted=True,
            sort=Sort.ASC,
        )

    @patch(_CLIENT_PATCH)
    def test_custom_timespan_and_multiplier(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """timespan and multiplier should be forwarded as-is."""
        mock_get_client.return_value.get_aggs.return_value = []

        get_ohlc(
            "TSLA",
            "2025-01-06",
            "2025-01-06",
            timespan="minute",
            multiplier=15,
        )

        call = mock_get_client.return_value.get_aggs.call_args
        assert call.args[1:4] == (15, "minute", "2025-01-06")

    @patch(_CLIENT_PATCH)
    def test_sort_desc_forwarded(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        mock_get_client.return_value.get_aggs.return_value = []

        get_ohlc("AAPL", "2025-01-01", "2025-01-31", sort="desc")

        assert mock_get_client.return_value.get_aggs.call_args.kwargs["sort"] == (
            Sort.DESC
        )

    @patch(_CLIENT_PATCH)
    def test_limit_only_included_when_set(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """``limit=None`` should not appear in the call kwargs — let Massive
        use its own default."""
        mock_get_client.return_value.get_aggs.return_value = []
        get_ohlc("AAPL", "2025-01-01", "2025-01-31")
        kwargs = mock_get_client.return_value.get_aggs.call_args.kwargs
        assert "limit" not in kwargs

        mock_get_client.reset_mock()
        mock_get_client.return_value.get_aggs.return_value = []
        get_ohlc("AAPL", "2025-01-01", "2025-01-31", limit=10_000)
        kwargs = mock_get_client.return_value.get_aggs.call_args.kwargs
        assert kwargs["limit"] == 10_000

    @patch(_CLIENT_PATCH)
    def test_adjusted_forwarded(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        mock_get_client.return_value.get_aggs.return_value = []
        get_ohlc("AAPL", "2025-01-01", "2025-01-31", adjusted=False)
        assert (
            mock_get_client.return_value.get_aggs.call_args.kwargs["adjusted"] is False
        )

    @patch(_CLIENT_PATCH)
    def test_accepts_date_objects(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """``from_``/``to`` can be date or datetime; just forward them."""
        mock_get_client.return_value.get_aggs.return_value = []

        start = dt.date(2025, 1, 1)
        end = dt.date(2025, 1, 31)
        get_ohlc("AAPL", start, end)

        args = mock_get_client.return_value.get_aggs.call_args.args
        assert args[3] is start and args[4] is end

    @patch(_CLIENT_PATCH)
    def test_empty_range_returns_empty_frame_with_schema(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """A range with no bars still returns a typed empty DataFrame."""
        mock_get_client.return_value.get_aggs.return_value = []

        df = get_ohlc("AAPL", "2025-12-25", "2025-12-25")

        assert df.height == 0
        assert "Timestamp" in df.columns
        assert df.schema["Open"] == pl.Float64

    @patch(_CLIENT_PATCH)
    def test_skips_non_agg_items_in_response(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        """Defensive: anything that isn't an Agg should be ignored."""
        mock_get_client.return_value.get_aggs.return_value = [
            _make_agg(),
            "garbage",  # type: ignore[list-item]
            _make_agg(timestamp=1_736_258_400_000),
        ]

        df = get_ohlc("AAPL", "2025-01-01", "2025-01-31")
        assert df.height == 2

    @patch(_CLIENT_PATCH)
    def test_null_timestamp_preserved_as_none(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        mock_get_client.return_value.get_aggs.return_value = [_make_agg(timestamp=None)]

        df = get_ohlc("AAPL", "2025-01-01", "2025-01-31")
        assert df.height == 1
        assert df["Timestamp"][0] is None


class TestGetOhlcTable:
    @patch(_CLIENT_PATCH)
    def test_returns_gt(self, mock_get_client: MagicMock) -> None:
        mock_get_client.return_value.get_aggs.return_value = [_make_agg()]
        gt = get_ohlc_table("AAPL", "2025-01-01", "2025-01-31")
        assert isinstance(gt, GT)

    @patch(_CLIENT_PATCH)
    def test_subtitle_reflects_timespan(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        mock_get_client.return_value.get_aggs.return_value = [_make_agg()]
        html = get_ohlc_table(
            "AAPL",
            "2025-01-01",
            "2025-01-31",
            timespan="hour",
            multiplier=4,
        ).as_raw_html()

        assert "AAPL OHLC" in html
        assert "4 hours" in html

    @patch(_CLIENT_PATCH)
    def test_subtitle_singular_for_multiplier_1(
        self,
        mock_get_client: MagicMock,
    ) -> None:
        mock_get_client.return_value.get_aggs.return_value = [_make_agg()]
        html = get_ohlc_table("AAPL", "2025-01-01", "2025-01-31").as_raw_html()
        assert "1 day" in html
        assert "1 days" not in html
