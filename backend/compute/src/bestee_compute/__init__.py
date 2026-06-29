"""Bestee - Financial data toolkit powered by the Massive API."""

__version__ = "0.1.0"

from bestee_compute.stocks.financials import build_financials_table
from bestee_compute.stocks.market import (
    get_latest_market_snapshot,
    get_market_snapshot,
    get_ohlc,
    get_ohlc_table,
    get_time_series,
)
from bestee_compute.stocks.models import (
    FinancialMetric,
    Metric,
    StatementType,
    TimeSeriesDef,
    TimeSeriesName,
    TimeSeriesSpan,
)
from bestee_compute.stocks.sic import get_sic_codes
from bestee_compute.stocks.tickers import get_all_tickers, get_ticker_details

__all__ = [
    "FinancialMetric",
    "Metric",
    "StatementType",
    "TimeSeriesDef",
    "TimeSeriesName",
    "TimeSeriesSpan",
    "build_financials_table",
    "get_all_tickers",
    "get_latest_market_snapshot",
    "get_market_snapshot",
    "get_ohlc",
    "get_ohlc_table",
    "get_sic_codes",
    "get_ticker_details",
    "get_time_series",
]
