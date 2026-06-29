"""Scrape ETF holdings from the VanEck website.

The VanEck site offers an xlsx download per ETF at
``/us/en/etf/equity/{ticker}/holdings/download/xlsx/``.  We fetch and
parse it with stdlib (zipfile + xml.etree) so no extra dependency is
required.
"""

import io
import logging
import zipfile

import httpx
import polars as pl
from great_tables import GT

from bestee_compute.etf._xlsx import parse_sheet, read_shared_strings

logger = logging.getLogger(__name__)

_URL_TEMPLATE = (
    "https://www.vaneck.com/us/en/etf/equity/{ticker}/holdings/download/xlsx/"
)
_REFERER = "https://www.vaneck.com/us/en/"
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
)


# ── Cell-value coercion ──────────────────────────────────────────────


def _parse_shares(s: str) -> int | None:
    s = s.strip().replace(",", "")
    if not s or s == "--":
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _parse_money(s: str) -> float | None:
    s = s.strip().replace("$", "").replace(",", "")
    if not s or s == "--":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_percent(s: str) -> float | None:
    """Parse ``"16.70%"`` to ``0.167``."""
    s = s.strip().rstrip("%").replace(",", "")
    if not s or s == "--":
        return None
    try:
        return float(s) / 100.0
    except ValueError:
        return None


def _parse_holdings_xlsx(content: bytes) -> pl.DataFrame:
    """Parse a VanEck-format holdings xlsx into a typed DataFrame.

    The expected layout is:

    * One or two title/summary rows at the top.
    * A header row containing at least the ``Ticker`` column.
    * One row per holding, ending with a blank-ticker row that begins
      the disclaimer/footer.

    Raises:
        ValueError: If the workbook layout doesn't match.
    """
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        shared = read_shared_strings(zf)
        rows = parse_sheet(zf, shared)

    header_idx = next(
        (i for i, row in enumerate(rows) if "Ticker" in row),
        None,
    )
    if header_idx is None:
        msg = "Holdings xlsx has no row containing 'Ticker' — layout changed?"
        raise ValueError(msg)
    headers = [h.strip() for h in rows[header_idx]]
    logger.debug("Detected header row %d: %s", header_idx, headers)

    data_rows: list[dict[str, str]] = []
    for row in rows[header_idx + 1 :]:
        padded = list(row) + [""] * (len(headers) - len(row))
        record = dict(zip(headers, padded, strict=False))
        if not record.get("Ticker", "").strip():
            break
        data_rows.append(record)

    if not data_rows:
        msg = "Holdings xlsx contained no data rows"
        raise ValueError(msg)
    logger.info("Parsed %d holdings from xlsx", len(data_rows))

    return pl.DataFrame(data_rows).with_columns(
        pl.col("Shares")
        .map_elements(_parse_shares, return_dtype=pl.Int64)
        .alias("Shares"),
        pl.col("Market Value (US$)")
        .map_elements(_parse_money, return_dtype=pl.Float64)
        .alias("Market Value (US$)"),
        pl.col("% of Net Assets")
        .map_elements(_parse_percent, return_dtype=pl.Float64)
        .alias("% of Net Assets"),
    )


# ── Public API ───────────────────────────────────────────────────────


def get_holdings(
    ticker: str = "SMH",
    *,
    timeout: float = 30.0,
) -> pl.DataFrame:
    """Download and parse the current holdings of a VanEck equity ETF.

    Hits the daily-holdings xlsx published at
    ``https://www.vaneck.com/us/en/etf/equity/{ticker}/holdings/download/xlsx/``.

    Args:
        ticker: ETF ticker (case-insensitive).  Defaults to ``"SMH"``.
        timeout: HTTP timeout in seconds.

    Returns:
        A :class:`polars.DataFrame` with one row per holding.  Columns:

        * ``Number`` (str)
        * ``Ticker`` (str)
        * ``Holding Name`` (str)
        * ``Identifier (FIGI)`` (str)
        * ``Shares`` (Int64, nullable)
        * ``Asset Class`` (str)
        * ``Market Value (US$)`` (Float64, nullable)
        * ``Notional Value`` (str — ``--`` preserved as-is)
        * ``% of Net Assets`` (Float64, decimal form — ``0.167`` for 16.7%)

    Raises:
        httpx.HTTPError: If the download request fails.
        ValueError: If the xlsx layout doesn't match the expected schema.
    """
    url = _URL_TEMPLATE.format(ticker=ticker.lower())
    logger.info("Fetching %s holdings from %s", ticker.upper(), url)
    with httpx.Client(
        follow_redirects=True,
        timeout=timeout,
        headers={"User-Agent": _USER_AGENT, "Referer": _REFERER},
    ) as client:
        response = client.get(url)
        response.raise_for_status()
    return _parse_holdings_xlsx(response.content)


def get_holdings_table(
    ticker: str = "SMH",
    *,
    timeout: float = 30.0,
) -> GT:
    """Return :func:`get_holdings` wrapped in a styled GT.

    The DataFrame columns are formatted for display: ``Shares`` and
    ``Market Value (US$)`` get thousands separators (the latter in
    compact form, e.g. ``$11.0B``); ``% of Net Assets`` is rendered as a
    percentage.  Missing values show as ``—``.
    """
    df = get_holdings(ticker, timeout=timeout)
    total_value = df["Market Value (US$)"].sum()
    subtitle = f"{df.height} holdings"
    if total_value is not None:
        subtitle += f" · AUM ${total_value:,.0f}"
    return (
        GT(df)
        .tab_header(title=f"{ticker.upper()} Holdings", subtitle=subtitle)
        .fmt_integer(columns=["Shares"])
        .fmt_currency(columns=["Market Value (US$)"], compact=True, decimals=2)
        .fmt_percent(columns=["% of Net Assets"], decimals=2)
        .sub_missing(missing_text="—")
        .cols_align(
            align="right",
            columns=["Shares", "Market Value (US$)", "% of Net Assets"],
        )
        .cols_align(
            align="left",
            columns=["Number", "Ticker", "Holding Name", "Identifier (FIGI)"],
        )
    )
