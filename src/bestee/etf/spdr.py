"""Scrape ETF holdings from State Street's SSGA / SPDR website.

SSGA publishes daily-holdings xlsx files at a clean per-ticker URL:
``https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-{ticker}.xlsx``.

The workbook layout is:

* Row 0 — ``Fund Name: <full name>``
* Row 1 — ``Ticker Symbol: <ticker>``
* Row 2 — ``Holdings: As of <date>``
* Row 3 — column headers: Name, Ticker, Identifier, SEDOL, Weight,
  Sector, Shares Held, Local Currency
* Rows 4..N — one holding per row
* Trailing blank rows + disclaimer below

Weights are reported as plain floats in **percent units** (e.g.
``8.348315`` means 8.348315%); the parser converts to decimals so the
output schema matches :mod:`bestee.etf.vaneck`.
"""

import io
import logging
import zipfile

import httpx
import polars as pl
from great_tables import GT

from bestee.etf._xlsx import parse_sheet, read_shared_strings

logger = logging.getLogger(__name__)

_URL_TEMPLATE = (
    "https://www.ssga.com/library-content/products/fund-data/etfs/us/"
    "holdings-daily-us-en-{ticker}.xlsx"
)
_REFERER = "https://www.ssga.com/"
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
)


# ── Cell-value coercion ──────────────────────────────────────────────


def _parse_float(s: str) -> float | None:
    s = s.strip().replace(",", "")
    if not s or s == "-":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_shares(s: str) -> int | None:
    """SPDR stores shares as scientific-notation floats (e.g. ``2.91237232E8``)."""
    v = _parse_float(s)
    if v is None:
        return None
    try:
        return int(round(v))
    except ValueError, OverflowError:
        return None


def _parse_weight_percent(s: str) -> float | None:
    """Convert a plain-percent float (``8.348315``) to decimal (``0.08348``)."""
    v = _parse_float(s)
    if v is None:
        return None
    return v / 100.0


def _parse_holdings_xlsx(content: bytes) -> pl.DataFrame:
    """Parse an SSGA SPDR holdings xlsx into a typed DataFrame.

    Raises:
        ValueError: If the workbook layout doesn't match.
    """
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        shared = read_shared_strings(zf)
        rows = parse_sheet(zf, shared)

    header_idx = next(
        (i for i, row in enumerate(rows) if "Ticker" in row and "Weight" in row),
        None,
    )
    if header_idx is None:
        msg = "SPDR holdings xlsx has no header row — layout changed?"
        raise ValueError(msg)
    headers = [h.strip() for h in rows[header_idx]]
    logger.debug("Detected header row %d: %s", header_idx, headers)

    data_rows: list[dict[str, str]] = []
    for row in rows[header_idx + 1 :]:
        padded = list(row) + [""] * (len(headers) - len(row))
        record = dict(zip(headers, padded, strict=False))
        # The data table ends with all-blank rows.  Stop at the first
        # blank-Ticker row (matches the trailing-rows shape we see).
        if not record.get("Ticker", "").strip():
            break
        data_rows.append(record)

    if not data_rows:
        msg = "SPDR holdings xlsx contained no data rows"
        raise ValueError(msg)
    logger.info("Parsed %d SPDR holdings from xlsx", len(data_rows))

    return pl.DataFrame(data_rows).with_columns(
        pl.col("Weight")
        .map_elements(_parse_weight_percent, return_dtype=pl.Float64)
        .alias("Weight"),
        pl.col("Shares Held")
        .map_elements(_parse_shares, return_dtype=pl.Int64)
        .alias("Shares Held"),
    )


# ── Public API ───────────────────────────────────────────────────────


def get_holdings(
    ticker: str = "SPY",
    *,
    timeout: float = 30.0,
) -> pl.DataFrame:
    """Download and parse the current holdings of a State Street SPDR ETF.

    Args:
        ticker: ETF ticker (case-insensitive).  Defaults to ``"SPY"``.
        timeout: HTTP timeout in seconds.

    Returns:
        A :class:`polars.DataFrame` with one row per holding and columns:

        * ``Name`` (str) — full holding name (e.g. ``"NVIDIA CORP"``)
        * ``Ticker`` (str)
        * ``Identifier`` (str) — CUSIP
        * ``SEDOL`` (str)
        * ``Weight`` (Float64) — decimal form (``0.0835`` for 8.35%)
        * ``Sector`` (str) — ``"-"`` if not classified
        * ``Shares Held`` (Int64)
        * ``Local Currency`` (str)

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
    ticker: str = "SPY",
    *,
    timeout: float = 30.0,
) -> GT:
    """Return :func:`get_holdings` wrapped in a styled GT."""
    df = get_holdings(ticker, timeout=timeout)
    subtitle = f"{df.height} holdings"
    return (
        GT(df)
        .tab_header(title=f"{ticker.upper()} Holdings", subtitle=subtitle)
        .fmt_integer(columns=["Shares Held"])
        .fmt_percent(columns=["Weight"], decimals=2)
        .sub_missing(missing_text="—")
        .cols_align(align="right", columns=["Weight", "Shares Held"])
        .cols_align(
            align="left",
            columns=["Name", "Ticker", "Identifier", "SEDOL", "Sector"],
        )
    )
