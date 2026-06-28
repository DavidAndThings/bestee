"""Fetch all tickers from the Massive (formerly Polygon.io) API."""

import logging
from concurrent.futures import ThreadPoolExecutor

import polars as pl
from great_tables import GT
from massive import RESTClient
from massive.rest.models import Ticker, TickerDetails

from bestee_compute.client import get_client
from bestee_compute.stocks import columns as cols

logger = logging.getLogger(__name__)

# Fields to extract from TickerDetails, in display order.  Each tuple is
# (attribute_name_on_SDK_model, column_label_in_table).  The SDK attribute
# names match the Stock dataclass fields, so the decorator pipeline reuses
# this to rebuild a Stock from a details row (label -> attribute).
DETAIL_FIELDS: list[tuple[str, str]] = [
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


def get_all_tickers_df(
    *,
    api_key: str | None = None,
    market: str | None = None,
    ticker_type: str | None = None,
    active: bool | None = True,
    limit: int = 1000,
) -> pl.DataFrame:
    """Return all tickers as a Polars DataFrame (unstyled).

    Used by the decorator pipeline; see :func:`get_all_tickers` for the
    Great Tables-wrapped public variant and full argument documentation.
    """
    logger.info(
        "Building all-tickers DataFrame (market=%s, type=%s, active=%s)",
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
    return pl.DataFrame(
        [
            {
                cols.TICKER: t.ticker,
                cols.NAME: t.name,
                cols.MARKET: t.market,
                cols.TYPE: t.type,
                cols.CURRENCY: t.currency_name,
                cols.EXCHANGE: t.primary_exchange,
                cols.ACTIVE: t.active,
            }
            for t in tickers
        ]
    )


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
    df = get_all_tickers_df(
        api_key=api_key,
        market=market,
        ticker_type=ticker_type,
        active=active,
        limit=limit,
    )
    logger.info("Built GT table with %d tickers", df.height)
    return (
        GT(df)
        .tab_header(
            title="All Tickers",
            subtitle=f"{df.height} tickers from the Massive API",
        )
        .cols_align(align="center", columns=[cols.MARKET, cols.TYPE, cols.ACTIVE])
        .cols_align(
            align="left",
            columns=[cols.TICKER, cols.NAME, cols.CURRENCY, cols.EXCHANGE],
        )
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


def get_ticker_detail(
    symbol: str,
    *,
    api_key: str | None = None,
) -> TickerDetails | None:
    """Return raw :class:`TickerDetails` for a single ticker, or *None*.

    Useful when callers only need a small piece of the detail payload
    (e.g. the long-form ``name``) and don't want the DataFrame wrapping
    that :func:`get_ticker_details_df` provides.  Network/SDK errors are
    caught and reported via the logger.

    Args:
        symbol: The ticker symbol to look up.
        api_key: Massive API key.  Falls back to ``MASSIVE_API_KEY`` when
            *None*.

    Returns:
        The :class:`TickerDetails` model on success, or *None* if the
        request failed or the SDK returned an unexpected type.

    Raises:
        RuntimeError: If no API key is available.
    """
    client = get_client(api_key)
    return _fetch_one_ticker_detail(client, symbol)


def get_ticker_details_df(
    tickers: list[str],
    *,
    api_key: str | None = None,
    max_workers: int = 10,
) -> pl.DataFrame:
    """Return ticker-details as a Polars DataFrame (unstyled).

    Used by the decorator pipeline; see :func:`get_ticker_details` for
    the Great Tables-wrapped public variant and full argument
    documentation.
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
        for attr, label in DETAIL_FIELDS:
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
    # Declare the schema explicitly: every cell is str-or-None.  Without
    # this, polars infers from the first few rows — if those rows are
    # null for a column it guesses the wrong dtype and chokes when a
    # later row finally has a string value.
    schema = {label: pl.Utf8 for _, label in DETAIL_FIELDS}
    return pl.DataFrame(rows, schema=schema)


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
    df = get_ticker_details_df(tickers, api_key=api_key, max_workers=max_workers)
    detail_cols = [label for _, label in DETAIL_FIELDS]

    logger.info("Built ticker details GT table")
    return (
        GT(df)
        .tab_header(
            title="Ticker Details",
            subtitle=f"{df.height} tickers from the Massive API",
        )
        .cols_align(align="left", columns=detail_cols)
        .sub_missing(missing_text="\u2014")
    )


def _sic_key(value: object) -> str:
    """Normalize a SIC code to a comparable string.

    Trims whitespace and drops leading zeros for numeric codes so equivalent
    forms match (``7372`` == ``"7372"``, and ``100`` == ``"0100"``).
    """
    text = str(value if value is not None else "").strip()
    return str(int(text)) if text.isdigit() else text


def get_tickers_by_sic_code(
    sic_code: str | int,
    *,
    tickers: list[str] | None = None,
    api_key: str | None = None,
    market: str | None = "stocks",
    ticker_type: str | None = "CS",
    active: bool | None = True,
    limit: int = 1000,
    max_workers: int = 10,
) -> list[str]:
    """Return every ticker whose company SIC code equals *sic_code*.

    The Massive ``list_tickers`` endpoint exposes no SIC filter, so the code
    must be read from each company's :class:`TickerDetails`.  This function
    therefore (1) takes a candidate universe -- *tickers* if given, otherwise
    every symbol matching *market*/*ticker_type*/*active* -- and (2) fetches
    their details concurrently and keeps those whose ``sic_code`` matches.

    Step 2 issues one details request per candidate, so the default universe
    (US common stocks) is several thousand calls.  Pass *tickers* to scope the
    search when you already have a candidate set.

    Args:
        sic_code: The SIC code to match (``"7372"`` or ``7372``).
        tickers: Candidate symbols to filter.  When *None*, the universe is
            fetched from the Massive API.
        api_key: Massive API key.  Falls back to the ``MASSIVE_API_KEY``
            environment variable when *None*.
        market: Market filter for the fetched universe (default ``"stocks"``).
        ticker_type: Ticker-type filter for the fetched universe (default
            ``"CS"``, common stock).  Pass *None* for every type.
        active: If *True* only actively-traded tickers form the universe.
        limit: Page size when fetching the universe (max 1000).
        max_workers: Number of concurrent details requests.

    Returns:
        A sorted list of the matching ticker symbols.

    Raises:
        RuntimeError: If no API key is available.
    """
    target = _sic_key(sic_code)
    if tickers is None:
        candidates = [
            t.ticker
            for t in _fetch_tickers(
                api_key=api_key,
                market=market,
                ticker_type=ticker_type,
                active=active,
                limit=limit,
            )
        ]
    else:
        candidates = list(tickers)

    client = get_client(api_key)
    logger.info(
        "Filtering %d candidate tickers for SIC code %s", len(candidates), target
    )
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        details = pool.map(lambda s: _fetch_one_ticker_detail(client, s), candidates)

    matches = sorted(
        detail.ticker
        for detail in details
        if detail is not None and _sic_key(getattr(detail, "sic_code", None)) == target
    )
    logger.info("Found %d tickers with SIC code %s", len(matches), target)
    return matches
