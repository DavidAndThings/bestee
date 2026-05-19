"""Fetch all tickers from the Massive (formerly Polygon.io) API."""

import logging
from concurrent.futures import ThreadPoolExecutor

import polars as pl
from great_tables import GT
from massive import RESTClient
from massive.rest.models import Ticker, TickerDetails

from bestee import columns as cols
from bestee.client import get_client

logger = logging.getLogger(__name__)

# Fields to extract from TickerDetails, in display order.
# Each tuple is (attribute_name_on_SDK_model, column_label_in_table).
_DETAIL_FIELDS: list[tuple[str, str]] = [
    ("ticker", cols.TICKER),
    ("name", cols.NAME),
    ("description", cols.DESCRIPTION),
    ("type", cols.TYPE),
    ("market", cols.MARKET),
    ("locale", cols.LOCALE),
    ("primary_exchange", cols.PRIMARY_EXCHANGE),
    ("currency_name", cols.CURRENCY),
    ("cik", cols.CIK),
    ("composite_figi", cols.COMPOSITE_FIGI),
    ("share_class_figi", cols.SHARE_CLASS_FIGI),
    ("sic_code", cols.SIC_CODE),
    ("sic_description", cols.SIC_DESCRIPTION),
    ("market_cap", cols.MARKET_CAP),
    ("share_class_shares_outstanding", cols.SHARES_OUTSTANDING),
    ("weighted_shares_outstanding", cols.WEIGHTED_SHARES_OUTSTANDING),
    ("total_employees", cols.TOTAL_EMPLOYEES),
    ("list_date", cols.LIST_DATE),
    ("homepage_url", cols.HOMEPAGE_URL),
    ("phone_number", cols.PHONE_NUMBER),
    ("ticker_root", cols.TICKER_ROOT),
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


def _fetch_one_ticker_detail(
    client: RESTClient,
    symbol: str,
) -> TickerDetails | None:
    """Fetch details for a single ticker, returning *None* on failure."""
    try:
        result = client.get_ticker_details(symbol)
    except Exception:
        logger.warning("Failed to fetch details for %s", symbol, exc_info=True)
        return None
    if isinstance(result, TickerDetails):
        logger.debug("Fetched details for %s", symbol)
        return result
    return None


def get_ticker_details(
    tickers: list[str],
    *,
    api_key: str | None = None,
    max_workers: int = 10,
) -> GT:
    """Return detailed company information for each ticker.

    The Massive ``get_ticker_details`` endpoint is per-ticker, so this
    function dispatches the requests **concurrently** using a thread
    pool.  The Massive SDK is synchronous but each call is I/O-bound,
    so threading gives a near-linear speedup until the API rate limit
    or the network is saturated.

    Args:
        tickers: Ticker symbols to look up (e.g. ``["AAPL", "MSFT"]``).
        api_key: Massive API key.  Falls back to the ``MASSIVE_API_KEY``
            environment variable when *None*.
        max_workers: Number of concurrent HTTP requests.  Defaults to
            ``10``.  Tune up for higher API tiers, down to respect rate
            limits.

    Returns:
        A :class:`great_tables.GT` display table with tickers as rows
        and detail fields as columns.  Tickers for which the API
        returned no data are silently dropped.

    Raises:
        RuntimeError: If no API key is available.
    """
    logger.info(
        "Fetching details for %d tickers (max_workers=%d)",
        len(tickers),
        max_workers,
    )
    client = get_client(api_key)

    # Preserve input order by mapping in a thread pool.
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(lambda t: _fetch_one_ticker_detail(client, t), tickers))

    details: list[TickerDetails] = [r for r in results if r is not None]

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
