"""Scrape ETF holdings from Roundhill Investments.

Roundhill publishes a single dated master holdings file covering every
fund in the family at::

    https://www.roundhillinvestments.com/assets/data/
        FilepointRoundhill.40RU.RU_Holdings_<MMDDYYYY>.csv

The fund's own per-ETF pages don't expose a stable per-ticker download —
the "Download CSV" button is built client-side from the master file —
so this module hits the master directly and filters by the ``Account``
column to produce single-ETF views.

The site's own JavaScript walks today's date backwards by up to 15 days
to find the most recently published file; this module does the same so
the call still works on weekends and holidays.
"""

import io
import logging
from datetime import UTC, date, datetime, timedelta

import httpx
import polars as pl
from great_tables import GT

logger = logging.getLogger(__name__)

_URL_TEMPLATE = (
    "https://www.roundhillinvestments.com/assets/data/"
    "FilepointRoundhill.40RU.RU_Holdings_{date}.csv"
)
_REFERER = "https://www.roundhillinvestments.com/"
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
)
# Match the fund site's own retry budget when walking back from "today".
_MAX_LOOKBACK_DAYS = 15


# ── CSV parsing ──────────────────────────────────────────────────────


def _parse_weight_percent(s: str | None) -> float | None:
    """Parse ``"5.53%"`` to ``0.0553``."""
    if s is None:
        return None
    s = s.strip().rstrip("%").replace(",", "")
    if not s or s == "-":
        return None
    try:
        return float(s) / 100.0
    except ValueError:
        return None


def _parse_holdings_csv(content: bytes) -> pl.DataFrame:
    """Parse the Filepoint master holdings CSV into a typed DataFrame.

    Columns published by Filepoint: ``Date``, ``Account``,
    ``StockTicker``, ``CUSIP``, ``SecurityName``, ``Shares``, ``Price``,
    ``MarketValue``, ``Weightings``, ``NetAssets``, ``SharesOutstanding``,
    ``CreationUnits``, ``MoneyMarketFlag``.

    ``Weightings`` is reported as a ``"5.53%"`` string; it is converted
    to decimal form (``0.0553``) so the schema matches the other
    providers in this package.

    Raises:
        ValueError: If the file has no ``Account`` column, indicating a
            layout change.
    """
    df = pl.read_csv(
        io.BytesIO(content),
        infer_schema_length=0,  # treat every column as String initially
    )
    if "Account" not in df.columns:
        msg = (
            "Roundhill holdings CSV has no 'Account' column — layout changed? "
            f"Got: {df.columns}"
        )
        raise ValueError(msg)

    numeric_cols = (
        "Shares",
        "Price",
        "MarketValue",
        "NetAssets",
        "SharesOutstanding",
        "CreationUnits",
    )
    expressions = [
        pl.col(c).str.replace_all(",", "").cast(pl.Float64, strict=False).alias(c)
        for c in numeric_cols
        if c in df.columns
    ]
    if "Weightings" in df.columns:
        expressions.append(
            pl.col("Weightings")
            .map_elements(_parse_weight_percent, return_dtype=pl.Float64)
            .alias("Weightings")
        )
    if expressions:
        df = df.with_columns(expressions)

    logger.info(
        "Parsed %d Roundhill holdings rows across %d ETFs",
        df.height,
        df["Account"].n_unique(),
    )
    return df


# ── Master-file download (with weekend/holiday walk-back) ────────────


def _candidate_dates(start: date) -> list[str]:
    """Return ``MMDDYYYY`` strings for *start* and the prior 15 days."""
    return [
        (start - timedelta(days=offset)).strftime("%m%d%Y")
        for offset in range(_MAX_LOOKBACK_DAYS + 1)
    ]


_CSV_HEADER_PREFIX = b"Date,Account,"


def _looks_like_csv(content: bytes) -> bool:
    """True if *content* starts with the expected Filepoint header.

    The Roundhill site responds 200 with an HTML 404 page (not a real
    HTTP error) when a dated file doesn't exist yet, so the walk-back
    has to validate the body, not just the status code.
    """
    return content.lstrip().startswith(_CSV_HEADER_PREFIX)


def _fetch_master_csv(
    client: httpx.Client,
    *,
    start: date,
) -> tuple[str, bytes]:
    """Find the most recent published master holdings CSV.

    Walks back from *start* day-by-day until a request returns CSV
    content (matching the fund site's own retry loop, but with body
    validation since the server hands out a 200-status HTML 404 page
    when a dated file is missing).  Returns the date string used (e.g.
    ``"05222026"``) and the raw CSV bytes.

    Raises:
        httpx.HTTPError: If every request in the lookback window fails
            with a real HTTP error.
        ValueError: If every response in the window returned 200 but
            with a non-CSV body — indicates the file naming/path has
            changed or the master simply isn't published yet.
    """
    last_http_exc: httpx.HTTPError | None = None
    tried: list[str] = []
    for stamp in _candidate_dates(start):
        url = _URL_TEMPLATE.format(date=stamp)
        tried.append(stamp)
        try:
            response = client.get(url)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            last_http_exc = exc
            logger.debug("Roundhill master HTTP error for %s (%s)", stamp, exc)
            continue
        if not _looks_like_csv(response.content):
            logger.debug("Roundhill master for %s returned non-CSV body", stamp)
            continue
        logger.info("Roundhill master holdings dated %s", stamp)
        return stamp, response.content

    if last_http_exc is not None:
        raise last_http_exc
    msg = (
        "No Roundhill master holdings CSV found within the "
        f"{_MAX_LOOKBACK_DAYS}-day lookback window starting at {start}. "
        f"Tried: {tried}."
    )
    raise ValueError(msg)


# ── Public API ───────────────────────────────────────────────────────


def get_all_holdings(
    *,
    as_of: date | None = None,
    timeout: float = 30.0,
) -> pl.DataFrame:
    """Return current holdings for **every** Roundhill ETF in one frame.

    Roundhill publishes one master file per business day covering the
    entire fund family.  This is the natural shape of their data, so
    fetching all-at-once is much cheaper than scraping each fund
    individually.

    Args:
        as_of: Date to start looking from.  Defaults to today (UTC).
            The site usually publishes a file by mid-morning ET; on
            weekends/holidays the function walks back up to 15 days to
            find the most recent one — matching the fund site's own
            retry loop.
        timeout: HTTP timeout in seconds.

    Returns:
        A :class:`polars.DataFrame` with one row per (ETF, holding)
        pair.  Columns:

        * ``Date`` (str, e.g. ``"05/26/2026"``)
        * ``Account`` (str) — ETF ticker the holding belongs to
        * ``StockTicker`` (str) — constituent ticker / Bloomberg code
        * ``CUSIP`` (str)
        * ``SecurityName`` (str)
        * ``Shares`` (Float64)
        * ``Price`` (Float64) — in the security's local currency
        * ``MarketValue`` (Float64) — in USD
        * ``Weightings`` (Float64) — decimal form (``0.0553`` = 5.53%)
        * ``NetAssets`` (Float64) — fund AUM in USD
        * ``SharesOutstanding`` (Float64) — ETF shares outstanding
        * ``CreationUnits`` (Float64)
        * ``MoneyMarketFlag`` (str)

    Raises:
        httpx.HTTPError: If every request in the lookback window
            errored at the HTTP layer.
        ValueError: If the CSV layout has changed, or no master file
            was found within the 15-day window (the server returns a
            200-status HTML 404 page when a file is missing, so the
            walk-back validates the body, not just the status code).
    """
    start = as_of if as_of is not None else datetime.now(UTC).date()
    with httpx.Client(
        follow_redirects=True,
        timeout=timeout,
        headers={"User-Agent": _USER_AGENT, "Referer": _REFERER},
    ) as client:
        _, content = _fetch_master_csv(client, start=start)
    return _parse_holdings_csv(content)


def list_etfs(
    *,
    as_of: date | None = None,
    timeout: float = 30.0,
) -> list[str]:
    """Return the tickers of every Roundhill ETF in the latest master file."""
    df = get_all_holdings(as_of=as_of, timeout=timeout)
    return sorted(df["Account"].unique().to_list())


def get_holdings(
    ticker: str = "CHAT",
    *,
    as_of: date | None = None,
    timeout: float = 30.0,
) -> pl.DataFrame:
    """Return the current holdings for a single Roundhill ETF.

    Loads the master file via :func:`get_all_holdings` and filters by
    the ``Account`` column.  Callers fetching many tickers at once
    should prefer :func:`get_all_holdings` directly to avoid redundant
    network round-trips.

    Args:
        ticker: Roundhill ETF ticker (case-insensitive).  Defaults to
            ``"CHAT"``.
        as_of: Date to start looking from — see :func:`get_all_holdings`.
        timeout: HTTP timeout in seconds.

    Returns:
        Same schema as :func:`get_all_holdings`, filtered to *ticker*.

    Raises:
        ValueError: If the ticker isn't present in the master file.
        httpx.HTTPError: If the master file can't be downloaded.
    """
    df = get_all_holdings(as_of=as_of, timeout=timeout)
    upper = ticker.upper()
    filtered = df.filter(pl.col("Account") == upper)
    if filtered.height == 0:
        available = sorted(df["Account"].unique().to_list())
        msg = (
            f"No Roundhill holdings found for {ticker!r}. "
            f"Available tickers: {available}."
        )
        raise ValueError(msg)
    return filtered


def get_holdings_table(
    ticker: str = "CHAT",
    *,
    as_of: date | None = None,
    timeout: float = 30.0,
) -> GT:
    """Return :func:`get_holdings` wrapped in a styled GT."""
    df = get_holdings(ticker, as_of=as_of, timeout=timeout)
    as_of_str = df["Date"][0] if df.height else ""
    subtitle = f"{df.height} holdings · As of {as_of_str}"
    return (
        GT(df.drop(["Date", "Account"]))
        .tab_header(title=f"{ticker.upper()} Holdings", subtitle=subtitle)
        .fmt_number(columns=["Shares", "SharesOutstanding"], decimals=0)
        .fmt_currency(columns=["MarketValue", "NetAssets"], compact=True, decimals=2)
        .fmt_number(columns=["Price"], decimals=2)
        .fmt_percent(columns=["Weightings"], decimals=2)
        .sub_missing(missing_text="—")
        .cols_align(
            align="right",
            columns=[
                "Shares",
                "Price",
                "MarketValue",
                "Weightings",
                "NetAssets",
                "SharesOutstanding",
                "CreationUnits",
            ],
        )
        .cols_align(
            align="left",
            columns=["StockTicker", "CUSIP", "SecurityName", "MoneyMarketFlag"],
        )
    )
