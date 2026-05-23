"""Project provider-specific holdings DataFrames to a canonical schema.

Each ETF issuer publishes a different column layout (VanEck reports
Market Value + FIGI, SPDR reports SEDOL + Sector, iShares reports
Quantity + Price, Roundhill exposes CUSIP + creation-unit info, etc.).
Cross-issuer analysis is easier when all providers' DataFrames share a
small uniform shape.

The canonical schema is:

* ``ticker`` (str, nullable — iShares bond ETFs report by name only)
* ``name``   (str)
* ``weight`` (Float64, decimal form — ``0.083`` for 8.3%)
* ``market_value`` (Float64, nullable — SPDR doesn't report it)
* ``issuer`` (str)

Use :func:`normalize_holdings` to project a DataFrame you already have,
or :func:`get_holdings_normalized` to dispatch + normalize in one call.
"""

import polars as pl

from bestee.etf.holdings import _resolve_provider

# Per-issuer mapping: canonical -> the column name carrying that field
# in that provider's native DataFrame.  Missing entries mean the
# provider doesn't report that field; the column is filled with nulls.
_COLUMN_MAP: dict[str, dict[str, str]] = {
    "VanEck": {
        "ticker": "Ticker",
        "name": "Holding Name",
        "weight": "% of Net Assets",
        "market_value": "Market Value (US$)",
    },
    "SPDR": {
        "ticker": "Ticker",
        "name": "Name",
        "weight": "Weight",
        # SPDR's xlsx omits Market Value.
    },
    "iShares": {
        "ticker": "Ticker",
        "name": "Name",
        "weight": "Weight (%)",
        "market_value": "Market Value",
    },
    "Roundhill": {
        "ticker": "StockTicker",
        "name": "SecurityName",
        "weight": "Weightings",
        "market_value": "MarketValue",
    },
    "Invesco": {
        "ticker": "Ticker",
        "name": "Issuer Name",
        "weight": "% of Net Assets",
        # Invesco's JSON doesn't include a market-value field.
    },
}

_CANONICAL_COLS: tuple[str, ...] = ("ticker", "name", "weight", "market_value")


def supported_issuers_for_normalization() -> list[str]:
    """Return the issuer names :func:`normalize_holdings` understands."""
    return sorted(_COLUMN_MAP)


def normalize_holdings(df: pl.DataFrame, issuer: str) -> pl.DataFrame:
    """Project *df* to the canonical (ticker, name, weight, market_value,
    issuer) schema using the provider's known column layout.

    Columns the issuer doesn't publish (e.g. SPDR's missing
    ``market_value``, or an iShares bond fund's missing ``Ticker``) are
    filled with nulls of the appropriate dtype.

    Args:
        df: The provider's native holdings DataFrame, as returned by
            :func:`bestee.etf.<provider>.get_holdings`.
        issuer: The canonical issuer name (one of the keys of
            :data:`_COLUMN_MAP`).

    Returns:
        A new DataFrame with columns ``ticker``, ``name``, ``weight``,
        ``market_value``, ``issuer``.  All original columns are dropped
        — use :func:`pl.DataFrame.join` if you need to reattach them.

    Raises:
        KeyError: If *issuer* isn't a known provider.
    """
    if issuer not in _COLUMN_MAP:
        msg = f"Unknown issuer {issuer!r}. Known: {sorted(_COLUMN_MAP)}."
        raise KeyError(msg)
    mapping = _COLUMN_MAP[issuer]

    exprs: list[pl.Expr] = []
    for canon in _CANONICAL_COLS:
        src = mapping.get(canon)
        if src is not None and src in df.columns:
            exprs.append(pl.col(src).alias(canon))
        else:
            # Field not provided — emit nulls of the canonical dtype.
            dtype = pl.Utf8 if canon in ("ticker", "name") else pl.Float64
            exprs.append(pl.lit(None, dtype=dtype).alias(canon))

    return df.select(exprs).with_columns(pl.lit(issuer).alias("issuer"))


def get_holdings_normalized(ticker: str) -> pl.DataFrame:
    """Dispatch and normalize in one call.

    Equivalent to::

        from bestee.etf import get_holdings, guess_etf_issuer
        df = get_holdings(ticker)
        issuer = guess_etf_issuer(ticker)
        normalize_holdings(df, issuer)

    but without re-running the issuer guess.

    Returns:
        DataFrame with the canonical schema.

    Raises:
        bestee.etf.IssuerNotSupportedError: If *ticker*'s issuer has no
            registered scraper.
    """
    issuer, provider = _resolve_provider(ticker)
    df = provider.get_holdings(ticker)
    return normalize_holdings(df, issuer)
