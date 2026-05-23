"""Scrape ETF holdings from iShares (BlackRock).

iShares serves daily-holdings CSVs from a per-portfolio document
endpoint that uses an immutable numeric ``portfolioId`` rather than a
ticker symbol::

    https://www.blackrock.com/varnish-api/blk-one01-product-data/product-data/
        api/v1/get-fund-document?
        appType=PRODUCT_PAGE&appSubType=ISHARES&targetSite=us-ishares
        &locale=en_US&portfolioId=<ID>&userType=individual&component=holdings

The numeric ID is the same one embedded in iShares' public product URLs
(``/us/products/<ID>/<slug>``).  iShares does not publish a programmatic
ticker→ID index, so this module ships a curated mapping
(:data:`_PRODUCT_IDS`) for popular ETFs.  Callers can pass
``product_id`` explicitly to bypass it.
"""

import io
import logging

import httpx
import polars as pl
from great_tables import GT

logger = logging.getLogger(__name__)

_DOCUMENT_URL = (
    "https://www.blackrock.com/varnish-api/blk-one01-product-data/"
    "product-data/api/v1/get-fund-document"
)
_REFERER = "https://www.ishares.com/"
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
)


# ── Ticker → portfolio-ID lookup ─────────────────────────────────────


# Verified by hitting each product page and confirming the title.
# Covers the top iShares US ETFs by AUM.  Add entries as needed; pass
# ``product_id=`` explicitly to override or for unlisted tickers.
_PRODUCT_IDS: dict[str, int] = {
    # Broad US equity
    "IVV": 239726,  # Core S&P 500
    "ITOT": 239724,  # Core S&P Total US Stock Market
    "IJR": 239774,  # Core S&P Small-Cap
    "IJH": 239763,  # Core S&P Mid-Cap
    "IWM": 239710,  # Russell 2000
    "IWF": 239706,  # Russell 1000 Growth
    "IWD": 239708,  # Russell 1000 Value
    # Factor / style
    "USMV": 239695,  # MSCI USA Min Vol Factor
    "MTUM": 251614,  # MSCI USA Momentum Factor
    "VLUE": 251616,  # MSCI USA Value Factor
    "HDV": 239563,  # Core High Dividend
    # Fixed income
    "AGG": 239458,  # Core US Aggregate Bond
    "TLT": 239454,  # 20+ Year Treasury Bond
    "IEF": 239456,  # 7-10 Year Treasury Bond
    "SHY": 239452,  # 1-3 Year Treasury Bond
    "LQD": 239566,  # iBoxx $ Investment Grade Corporate Bond
    "HYG": 239565,  # iBoxx $ High Yield Corporate Bond
    "TIP": 239467,  # TIPS Bond
    # International
    "IEFA": 244049,  # Core MSCI EAFE
    "IEMG": 244050,  # Core MSCI Emerging Markets
    "EFA": 239623,  # MSCI EAFE
    "EEM": 239637,  # MSCI Emerging Markets
}


# ── CSV parsing ──────────────────────────────────────────────────────


_NUMERIC_COLS = (
    # Equity ETF columns
    "Market Value",
    "Weight (%)",
    "Notional Value",
    "Quantity",
    "Price",
    "FX Rate",
    # Bond ETF columns
    "Par Value",
    "Duration",
    "YTM (%)",
    "Coupon (%)",
    "Mod. Duration",
    "Yield to Call (%)",
    "Yield to Worst (%)",
    "Real Duration",
    "Real YTM (%)",
)


def _parse_holdings_csv(content: bytes) -> pl.DataFrame:
    """Parse an iShares holdings CSV into a typed DataFrame.

    The published CSV starts with several metadata rows (fund name, "as
    of" date, asset-class summary), then a blank line, then a standard
    column-header row beginning with ``Ticker,``.

    Numeric columns are stored as quoted strings with thousands
    separators (e.g. ``"1,234.56"``); they're cleaned and cast to
    Float64.  ``Weight (%)`` is converted from percent units (e.g.
    ``8.34``) to decimal form (``0.0834``) so the column matches the
    schema used by :mod:`bestee.etf.vaneck` and :mod:`bestee.etf.spdr`.

    Raises:
        ValueError: If the file has no recognizable ``Ticker,`` header
            row.
    """
    text = content.decode("utf-8", errors="replace")
    lines = text.splitlines()
    # The header row is the first line that has both ``Name,`` and
    # ``Weight (%),`` — equity ETFs lead with ``Ticker,`` but bond ETFs
    # start the header at ``Name,`` since bonds have no tickers.
    header_idx = next(
        (
            i
            for i, line in enumerate(lines)
            if "Weight (%)" in line and ("Ticker," in line or line.startswith("Name,"))
        ),
        None,
    )
    if header_idx is None:
        msg = "iShares holdings CSV has no recognizable header row — layout changed?"
        raise ValueError(msg)

    body = "\n".join(lines[header_idx:])
    df = pl.read_csv(
        io.BytesIO(body.encode("utf-8")),
        infer_schema_length=0,  # treat every column as String initially
        ignore_errors=True,
        truncate_ragged_lines=True,
    )

    # Drop fully-empty trailing rows (iShares sometimes appends them).
    # Use ``Ticker`` for equity funds and fall back to ``Name`` for bonds.
    key_col = "Ticker" if "Ticker" in df.columns else "Name"
    df = df.filter(
        pl.col(key_col).is_not_null() & (pl.col(key_col).str.strip_chars() != "")
    )

    # Coerce numeric columns: strip commas, cast to Float64.
    expressions = [
        pl.col(c).str.replace_all(",", "").cast(pl.Float64, strict=False).alias(c)
        for c in _NUMERIC_COLS
        if c in df.columns
    ]
    if expressions:
        df = df.with_columns(expressions)

    # Normalize Weight (%) to decimal form.
    if "Weight (%)" in df.columns:
        df = df.with_columns((pl.col("Weight (%)") / 100.0).alias("Weight (%)"))

    logger.info("Parsed %d iShares holdings from CSV", df.height)
    return df


# ── Public API ───────────────────────────────────────────────────────


def get_holdings(
    ticker: str = "IVV",
    *,
    product_id: int | None = None,
    timeout: float = 30.0,
) -> pl.DataFrame:
    """Download and parse the current holdings of an iShares ETF.

    Args:
        ticker: ETF ticker (case-insensitive).  Defaults to ``"IVV"``.
        product_id: Override the looked-up iShares portfolio ID.  Useful
            when *ticker* isn't in the bundled mapping
            (:data:`_PRODUCT_IDS`).  Find the ID in the public iShares
            URL ``/us/products/<id>/<slug>``.
        timeout: HTTP timeout in seconds.

    Returns:
        Polars DataFrame with one row per holding.  Columns:
        ``Ticker``, ``Name``, ``Sector``, ``Asset Class``,
        ``Market Value`` (Float64), ``Weight (%)`` (Float64, decimal
        form — ``0.0834`` for 8.34%), ``Notional Value`` (Float64),
        ``Quantity`` (Float64), ``Price`` (Float64), ``Location``,
        ``Exchange``, ``Currency``, ``FX Rate`` (Float64),
        ``Market Currency``, ``Accrual Date``.

    Raises:
        ValueError: If the ticker has no known ``product_id`` and none
            was passed in, or the response has an unexpected layout.
        httpx.HTTPError: If the download fails.
    """
    pid = product_id if product_id is not None else _PRODUCT_IDS.get(ticker.upper())
    if pid is None:
        msg = (
            f"No iShares product_id known for {ticker!r}. "
            f"Pass product_id= explicitly (find it in the iShares URL "
            f"'/us/products/<id>/<slug>'), or add it to _PRODUCT_IDS. "
            f"Known tickers: {sorted(_PRODUCT_IDS)}."
        )
        raise ValueError(msg)

    params = {
        "appType": "PRODUCT_PAGE",
        "appSubType": "ISHARES",
        "targetSite": "us-ishares",
        "locale": "en_US",
        "portfolioId": str(pid),
        "userType": "individual",
        "asOfDate": "",
        "component": "holdings",
    }
    logger.info("Fetching %s holdings (portfolioId=%d)", ticker.upper(), pid)
    with httpx.Client(
        follow_redirects=True,
        timeout=timeout,
        headers={"User-Agent": _USER_AGENT, "Referer": _REFERER},
    ) as client:
        response = client.get(_DOCUMENT_URL, params=params)
        response.raise_for_status()
    return _parse_holdings_csv(response.content)


def get_holdings_table(
    ticker: str = "IVV",
    *,
    product_id: int | None = None,
    timeout: float = 30.0,
) -> GT:
    """Return :func:`get_holdings` wrapped in a styled GT."""
    df = get_holdings(ticker, product_id=product_id, timeout=timeout)
    subtitle = f"{df.height} holdings"
    return (
        GT(df)
        .tab_header(title=f"{ticker.upper()} Holdings", subtitle=subtitle)
        .fmt_currency(columns=["Market Value"], compact=True, decimals=2)
        .fmt_percent(columns=["Weight (%)"], decimals=2)
        .fmt_number(columns=["Quantity"], decimals=0)
        .fmt_number(columns=["Price"], decimals=2)
        .sub_missing(missing_text="—")
        .cols_align(
            align="right",
            columns=["Market Value", "Weight (%)", "Quantity", "Price"],
        )
        .cols_align(
            align="left",
            columns=["Ticker", "Name", "Sector", "Asset Class"],
        )
    )
