"""Scrape ETF holdings from Invesco.

Invesco's main product catalog sits behind an Akamai-managed bot wall,
but the QQQ marketing site (``invesco.com/qqq-etf/``) is backed by a
public JSON API on a separate domain that's reachable without browser
automation::

    https://dng-api.invesco.com/cache/v1/accounts/en_US/
        shareclasses/QQQ/holdings/fund?idType=ticker
        &interval=monthly&productType=ETF

This module wires up that endpoint.  Only **QQQ** is supported — other
Invesco ETFs return the same JSON schema but their backends are
gated.  Calling :func:`get_holdings` with another ticker raises a
clear :class:`ValueError`.

The response schema is:

* ``cusip`` (str) — the fund's CUSIP (not constituent's)
* ``effectiveDate`` / ``effectiveBusinessDate`` (str) — "YYYY-MM-DD"
* ``totalNumberOfHoldings`` (int)
* ``holdings`` (list of dicts), each with ``ticker``, ``issuerName``,
  ``units``, ``percentageOfTotalNetAssets``, ``securityTypeName``,
  ``cusip``, ``currency``, ``localCurrencyName``, ``securityTypeCode``.
"""

import json
import logging

import httpx
import polars as pl
from great_tables import GT

logger = logging.getLogger(__name__)

_URL_TEMPLATE = (
    "https://dng-api.invesco.com/cache/v1/accounts/en_US/"
    "shareclasses/{ticker}/holdings/fund"
    "?idType=ticker&interval=monthly&productType=ETF"
)
_REFERER = "https://www.invesco.com/qqq-etf/en/about.html"
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
)
_SUPPORTED_TICKERS: frozenset[str] = frozenset({"QQQ"})


# ── JSON parsing ─────────────────────────────────────────────────────


def _parse_holdings_json(content: bytes) -> pl.DataFrame:
    """Parse the dng-api ``/holdings/fund`` JSON into a typed DataFrame.

    Columns: ``Ticker``, ``Issuer Name``, ``Units``, ``% of Net Assets``
    (decimal form — ``0.0876`` for 8.76%), ``CUSIP``, ``Security Type``,
    ``Currency``.

    Raises:
        ValueError: If the payload has no ``holdings`` list.
    """
    payload = json.loads(content)
    holdings = payload.get("holdings")
    if not isinstance(holdings, list):
        msg = (
            "Invesco holdings JSON has no 'holdings' list — "
            f"got top-level keys: {sorted(payload)}"
        )
        raise ValueError(msg)

    rows: list[dict[str, object]] = [
        {
            "Ticker": h.get("ticker"),
            "Issuer Name": h.get("issuerName"),
            "Units": h.get("units"),
            "% of Net Assets": h.get("percentageOfTotalNetAssets"),
            "CUSIP": h.get("cusip"),
            "Security Type": h.get("securityTypeName"),
            "Currency": h.get("currency"),
        }
        for h in holdings
    ]
    df = pl.DataFrame(
        rows,
        schema={
            "Ticker": pl.Utf8,
            "Issuer Name": pl.Utf8,
            "Units": pl.Float64,
            "% of Net Assets": pl.Float64,
            "CUSIP": pl.Utf8,
            "Security Type": pl.Utf8,
            "Currency": pl.Utf8,
        },
    ).with_columns(
        # Invesco reports weights in percent units (8.757223 = 8.76%);
        # convert to decimal to match the other providers.
        (pl.col("% of Net Assets") / 100.0).alias("% of Net Assets"),
    )

    logger.info("Parsed %d Invesco holdings from JSON", df.height)
    return df


# ── Public API ───────────────────────────────────────────────────────


def get_holdings(
    ticker: str = "QQQ",
    *,
    timeout: float = 30.0,
) -> pl.DataFrame:
    """Download and parse the current holdings of an Invesco ETF.

    Only **QQQ** is supported — see the module docstring for why.

    Args:
        ticker: Invesco ETF ticker.  Case-insensitive.  Defaults to
            ``"QQQ"``.
        timeout: HTTP timeout in seconds.

    Returns:
        A :class:`polars.DataFrame` with one row per holding and columns:

        * ``Ticker`` (str)
        * ``Issuer Name`` (str) — e.g. ``"NVIDIA Corp"``
        * ``Units`` (Float64) — shares held
        * ``% of Net Assets`` (Float64) — decimal form (``0.0876`` = 8.76%)
        * ``CUSIP`` (str)
        * ``Security Type`` (str) — e.g. ``"Common Stock"``
        * ``Currency`` (str) — e.g. ``"USD"``

    Raises:
        ValueError: If *ticker* isn't QQQ, or the JSON layout has
            changed.
        httpx.HTTPError: If the request fails.
    """
    upper = ticker.upper()
    if upper not in _SUPPORTED_TICKERS:
        msg = (
            f"Invesco provider only supports {sorted(_SUPPORTED_TICKERS)} "
            f"right now (got {ticker!r}).  Other Invesco ETFs sit behind "
            "a bot-protected backend that this module can't reach yet."
        )
        raise ValueError(msg)

    url = _URL_TEMPLATE.format(ticker=upper)
    logger.info("Fetching %s holdings from %s", upper, url)
    with httpx.Client(
        follow_redirects=True,
        timeout=timeout,
        headers={
            "User-Agent": _USER_AGENT,
            "Referer": _REFERER,
            "Origin": "https://www.invesco.com",
            "Accept": "application/json,*/*",
        },
    ) as client:
        response = client.get(url)
        response.raise_for_status()
    return _parse_holdings_json(response.content)


def get_holdings_table(
    ticker: str = "QQQ",
    *,
    timeout: float = 30.0,
) -> GT:
    """Return :func:`get_holdings` wrapped in a styled GT."""
    df = get_holdings(ticker, timeout=timeout)
    subtitle = f"{df.height} holdings"
    return (
        GT(df)
        .tab_header(title=f"{ticker.upper()} Holdings", subtitle=subtitle)
        .fmt_number(columns=["Units"], decimals=0)
        .fmt_percent(columns=["% of Net Assets"], decimals=2)
        .sub_missing(missing_text="—")
        .cols_align(align="right", columns=["Units", "% of Net Assets"])
        .cols_align(
            align="left",
            columns=[
                "Ticker",
                "Issuer Name",
                "CUSIP",
                "Security Type",
                "Currency",
            ],
        )
    )
