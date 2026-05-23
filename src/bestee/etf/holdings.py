"""Dispatch ETF-holdings requests to the correct issuer scraper.

Use :func:`get_holdings` (or :func:`get_holdings_table`) when you have a
ticker but don't know — or don't want to hard-code — which issuer
publishes it.  The dispatcher calls :func:`bestee.etf.issuer.guess_etf_issuer`
to identify the brand, then forwards to the matching provider module.

Adding a new provider:

1. Implement ``get_holdings(ticker)`` (and optionally ``get_holdings_table``)
   in ``bestee/etf/<provider>.py``.
2. Register it in :data:`_PROVIDERS` below.
3. Make sure :data:`bestee.etf.issuer._ISSUER_PATTERNS` returns the
   canonical brand name that matches a :data:`_PROVIDERS` key.
"""

import logging
from collections.abc import Callable
from typing import Any

import polars as pl
from great_tables import GT

from bestee.etf import ishares, spdr, vaneck  # noqa: F401 — resolved via globals()
from bestee.etf.issuer import guess_etf_issuer

logger = logging.getLogger(__name__)


class IssuerNotSupportedError(ValueError):
    """Raised when no scraper is registered for the ETF's issuer."""


# Maps the canonical issuer name (as returned by ``guess_etf_issuer``) to
# the *module-level attribute name* of the provider in this module.
# We resolve the module via ``globals()`` at call time so tests can patch
# ``bestee.etf.holdings.<provider>`` and have the dispatch honour it.
_PROVIDER_MODULES: dict[str, str] = {
    "VanEck": "vaneck",
    "SPDR": "spdr",
    "iShares": "ishares",
}


def supported_issuers() -> list[str]:
    """Return the canonical issuer names this module can dispatch to."""
    return sorted(_PROVIDER_MODULES)


def _resolve_provider(ticker: str) -> tuple[str, Any]:
    """Identify the issuer and look up its provider module.

    Raises:
        IssuerNotSupportedError: If the issuer is unknown to
            ``guess_etf_issuer`` or no scraper is registered for it.
    """
    issuer = guess_etf_issuer(ticker)
    if issuer is None:
        msg = (
            f"Could not identify the issuer for {ticker!r}. "
            f"Supported issuers: {supported_issuers()}."
        )
        raise IssuerNotSupportedError(msg)
    mod_name = _PROVIDER_MODULES.get(issuer)
    if mod_name is None:
        msg = (
            f"No holdings scraper registered for issuer {issuer!r} "
            f"(ticker {ticker!r}). Supported issuers: {supported_issuers()}."
        )
        raise IssuerNotSupportedError(msg)
    return issuer, globals()[mod_name]


def get_holdings(ticker: str) -> pl.DataFrame:
    """Return the current holdings for *ticker*, picking the right scraper.

    The schemas vary by issuer (different column names, sector vs no
    sector, etc.) — see each provider's :func:`get_holdings` docstring.

    Raises:
        IssuerNotSupportedError: If no scraper is available for the
            ticker's issuer.
        httpx.HTTPError: If the provider's download request fails.
        ValueError: If the downloaded payload has an unexpected layout.
    """
    issuer, provider = _resolve_provider(ticker)
    logger.info("Dispatching %s holdings request to %s", ticker, issuer)
    get_holdings_fn: Callable[..., pl.DataFrame] = provider.get_holdings
    return get_holdings_fn(ticker)


def get_holdings_table(ticker: str) -> GT:
    """Same as :func:`get_holdings` but returns a styled GT table."""
    issuer, provider = _resolve_provider(ticker)
    logger.info("Dispatching %s holdings-table request to %s", ticker, issuer)
    get_table_fn: Callable[..., GT] = provider.get_holdings_table
    return get_table_fn(ticker)
