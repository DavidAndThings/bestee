"""Fetch all tickers from the Massive (formerly Polygon.io) API."""

import os

from dotenv import load_dotenv
from massive import RESTClient
from massive.rest.models import Ticker


def get_all_tickers(
    *,
    api_key: str | None = None,
    market: str | None = None,
    ticker_type: str | None = None,
    active: bool | None = True,
    limit: int = 1000,
) -> list[Ticker]:
    """Return every ticker offered by the Massive API.

    The SDK's ``list_tickers`` method returns a paginated iterator that
    automatically follows ``next_url`` links, so this function simply
    exhausts the iterator and collects all results into a list.

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
    load_dotenv()
    key = api_key or os.environ.get("MASSIVE_API_KEY")
    if key is None:
        msg = (
            "No API key provided. Pass one via the 'api_key' parameter "
            "or set the MASSIVE_API_KEY environment variable."
        )
        raise RuntimeError(msg)

    client = RESTClient(api_key=key)

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
