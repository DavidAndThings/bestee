"""Fetch all tickers from the Massive (formerly Polygon.io) API."""

import logging

import polars as pl
from great_tables import GT
from massive.rest.models import Ticker, TickerDetails

from bestee.client import get_client

logger = logging.getLogger(__name__)

# Fields to extract from TickerDetails, in display order.
# Each tuple is (attribute_name, column_label).
_DETAIL_FIELDS: list[tuple[str, str]] = [
    ("ticker", "Ticker"),
    ("name", "Name"),
    ("description", "Description"),
    ("type", "Type"),
    ("market", "Market"),
    ("locale", "Locale"),
    ("primary_exchange", "Primary Exchange"),
    ("currency_name", "Currency"),
    ("cik", "CIK"),
    ("composite_figi", "Composite FIGI"),
    ("share_class_figi", "Share Class FIGI"),
    ("sic_code", "SIC Code"),
    ("sic_description", "SIC Description"),
    ("market_cap", "Market Cap"),
    ("share_class_shares_outstanding", "Shares Outstanding"),
    ("weighted_shares_outstanding", "Weighted Shares Outstanding"),
    ("total_employees", "Total Employees"),
    ("list_date", "List Date"),
    ("homepage_url", "Homepage URL"),
    ("phone_number", "Phone Number"),
    ("ticker_root", "Ticker Root"),
]


def _fetch_tickers(
    *,
    api_key: str | None = None,
    market: str | None = None,
    ticker_type: str | None = None,
    active: bool | None = True,
    limit: int = 1000,
) -> list[Ticker]:
    """Fetch every ticker from the Massive API and return raw objects.

    Args:
        api_key: Massive API key.  Falls back to the ``MASSIVE_API_KEY``
            environment variable when *None*.
        market: Optional market filter (e.g. ``"stocks"``, ``"crypto"``,
            ``"fx"``, ``"otc"``).
        ticker_type: Optional ticker type filter (e.g. ``"CS"`` for common
            stock, ``"ETF"``, ``"ADRC"``).  See the Ticker Types API for
            the full list.
        active: If *True* only actively-traded tickers are returned.
            *None* returns both active and delisted tickers.
        limit: Page size per request (max 1000).  A higher value means
            fewer HTTP round-trips.

    Returns:
        A list of :class:`massive.rest.models.Ticker` objects.

    Raises:
        RuntimeError: If no API key is available.
    """
    client = get_client(api_key)
    logger.info(
        "Fetching tickers (market=%s, type=%s, active=%s)", market, ticker_type, active
    )

    tickers: list[Ticker] = []
    for ticker in client.list_tickers(
        market=market,
        type=ticker_type,
        active=active,
        limit=limit,
    ):
        if isinstance(ticker, Ticker):
            tickers.append(ticker)

    logger.info("Fetched %d tickers", len(tickers))
    return tickers


def get_all_tickers(
    *,
    api_key: str | None = None,
    market: str | None = None,
    ticker_type: str | None = None,
    active: bool | None = True,
    limit: int = 1000,
) -> GT:
    """Return every ticker offered by the Massive API as a Great Tables object.

    Fetches all tickers (auto-paginating through every page) and returns
    the result as a :class:`great_tables.GT` display table.

    Args:
        api_key: Massive API key.  Falls back to the ``MASSIVE_API_KEY``
            environment variable when *None*.
        market: Optional market filter (e.g. ``"stocks"``, ``"crypto"``,
            ``"fx"``, ``"otc"``).
        ticker_type: Optional ticker type filter (e.g. ``"CS"`` for common
            stock, ``"ETF"``, ``"ADRC"``).  See the Ticker Types API for
            the full list.
        active: If *True* only actively-traded tickers are returned.
            *None* returns both active and delisted tickers.
        limit: Page size per request (max 1000).  A higher value means
            fewer HTTP round-trips.

    Returns:
        A :class:`great_tables.GT` display table.

    Raises:
        RuntimeError: If no API key is available.
    """
    logger.info(
        "Building all-tickers table (market=%s, type=%s, active=%s)",
        market,
        ticker_type,
        active,
    )
    tickers = _fetch_tickers(
        api_key=api_key,
        market=market,
        ticker_type=ticker_type,
        active=active,
        limit=limit,
    )

    df = pl.DataFrame(
        [
            {
                "Ticker": t.ticker,
                "Name": t.name,
                "Market": t.market,
                "Type": t.type,
                "Currency": t.currency_name,
                "Exchange": t.primary_exchange,
                "Active": t.active,
            }
            for t in tickers
        ]
    )

    logger.info("Built GT table with %d tickers", len(tickers))
    return (
        GT(df)
        .tab_header(
            title="All Tickers",
            subtitle=f"{len(tickers)} tickers from the Massive API",
        )
        .cols_align(align="center", columns=["Market", "Type", "Active"])
        .cols_align(align="left", columns=["Ticker", "Name", "Currency", "Exchange"])
    )


def get_ticker_details(
    tickers: list[str],
    *,
    api_key: str | None = None,
) -> GT:
    """Return detailed company information for each ticker.

    Calls the Massive ``get_ticker_details`` endpoint once per ticker and
    compiles the results into a :class:`great_tables.GT` table with
    tickers as rows and detail fields as columns.

    Args:
        tickers: Ticker symbols to look up (e.g. ``["AAPL", "MSFT"]``).
        api_key: Massive API key.  Falls back to the ``MASSIVE_API_KEY``
            environment variable when *None*.

    Returns:
        A :class:`great_tables.GT` display table.

    Raises:
        RuntimeError: If no API key is available.
    """
    logger.info("Fetching details for %d tickers: %s", len(tickers), tickers)
    client = get_client(api_key)

    details: list[TickerDetails] = []
    for symbol in tickers:
        result = client.get_ticker_details(symbol)
        logger.debug("Fetched details for %s", symbol)
        if isinstance(result, TickerDetails):
            details.append(result)

    logger.info("Retrieved details for %d/%d tickers", len(details), len(tickers))

    # One row per ticker, one column per detail field.
    rows: list[dict[str, str | None]] = []
    for det in details:
        row: dict[str, str | None] = {}
        for attr, label in _DETAIL_FIELDS:
            value = getattr(det, attr, None)
            if isinstance(value, float):
                row[label] = f"{value:,.0f}"
            elif isinstance(value, int):
                row[label] = f"{value:,}"
            elif value is not None:
                row[label] = str(value)
            else:
                row[label] = None
        rows.append(row)

    df = pl.DataFrame(rows)

    detail_cols = [label for _, label in _DETAIL_FIELDS]

    logger.info("Built ticker details GT table")
    return (
        GT(df)
        .tab_header(
            title="Ticker Details",
            subtitle=f"{len(details)} tickers from the Massive API",
        )
        .cols_align(align="left", columns=detail_cols)
        .sub_missing(missing_text="\u2014")
    )
