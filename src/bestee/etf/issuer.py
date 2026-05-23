"""Identify the issuer of an ETF from its Massive ticker-details name."""

import logging
import re

from bestee.stocks.tickers import get_ticker_detail

logger = logging.getLogger(__name__)


# (canonical brand name, word-boundary regex)
# Order matters when one brand's name contains another's — list the
# more specific brand first.
_ISSUER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (canonical, re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE))
    for canonical, keyword in (
        # SPDR is the brand; Massive prefixes these funds with "State
        # Street SPDR ..." so the substring still wins via word match.
        ("SPDR", "SPDR"),
        ("iShares", "iShares"),
        ("Vanguard", "Vanguard"),
        ("Invesco", "Invesco"),
        ("VanEck", "VanEck"),
        ("Schwab", "Schwab"),
        ("First Trust", "First Trust"),
        ("JPMorgan", "JPMorgan"),
        ("Goldman Sachs", "Goldman Sachs"),
        ("Fidelity", "Fidelity"),
        ("Franklin", "Franklin"),
        ("T. Rowe Price", "Rowe Price"),
        ("Nuveen", "Nuveen"),
        ("Hartford", "Hartford"),
        ("Janus Henderson", "Janus Henderson"),
        ("PIMCO", "PIMCO"),
        ("Pacer", "Pacer"),
        ("ProShares", "ProShares"),
        ("Direxion", "Direxion"),
        ("WisdomTree", "WisdomTree"),
        ("Global X", "Global X"),
        ("ARK", "ARK"),
        ("KraneShares", "KraneShares"),
        ("Roundhill", "Roundhill"),
        ("Innovator", "Innovator"),
        ("Amplify", "Amplify"),
        ("AdvisorShares", "AdvisorShares"),
        ("USCF", "USCF"),
        ("abrdn", "abrdn"),
        ("Dimensional", "Dimensional"),
        ("DWS", "DWS"),
        # Falls through to BlackRock if "iShares" wasn't matched first.
        ("BlackRock", "BlackRock"),
        ("State Street", "State Street"),
    )
)


def _match_issuer(name: str) -> str | None:
    """Return the canonical issuer brand found in *name*, or ``None``."""
    if not name:
        return None
    for issuer, pattern in _ISSUER_PATTERNS:
        if pattern.search(name):
            return issuer
    return None


def guess_etf_issuer(
    ticker: str,
    *,
    api_key: str | None = None,
) -> str | None:
    """Guess the issuer of an ETF by looking up its Massive ticker name.

    Calls the Massive ``get_ticker_details`` endpoint to fetch the
    long-form name (e.g. ``"VanEck Semiconductor ETF"``) and matches it
    against a curated list of major US ETF brands.

    Args:
        ticker: The ETF ticker symbol (case-insensitive on Massive's
            side).
        api_key: Massive API key.  Falls back to ``MASSIVE_API_KEY`` when
            *None*.

    Returns:
        The issuer brand (e.g. ``"VanEck"``, ``"iShares"``, ``"SPDR"``)
        or *None* if the ticker has no details, the lookup failed, or no
        known issuer matched.

    Raises:
        RuntimeError: If no API key is available.
    """
    details = get_ticker_detail(ticker, api_key=api_key)
    if details is None:
        return None
    name = details.name or ""
    if not name:
        logger.warning("Ticker details for %s have no name", ticker)
        return None
    issuer = _match_issuer(name)
    logger.info("Ticker %s -> %r -> issuer=%r", ticker, name, issuer)
    return issuer
