"""Fetch all tickers from the Massive (formerly Polygon.io) API."""

import polars as pl
from great_tables import GT
from massive.rest.models import Ticker

from bestee.client import get_client


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

    tickers: list[Ticker] = []
    for ticker in client.list_tickers(
        market=market,
        type=ticker_type,
        active=active,
        limit=limit,
    ):
        if isinstance(ticker, Ticker):
            tickers.append(ticker)

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

    return (
        GT(df)
        .tab_header(
            title="All Tickers",
            subtitle=f"{len(tickers)} tickers from the Massive API",
        )
        .cols_align(align="center", columns=["Market", "Type", "Active"])
        .cols_align(align="left", columns=["Ticker", "Name", "Currency", "Exchange"])
    )
